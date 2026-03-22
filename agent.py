from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from datetime import datetime, timezone
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openai_module import OpenAIClient, OpenAIConfigurationError, OpenAIResponseError

LOGGER = logging.getLogger(__name__)

PAGE_TIMEOUT_SECONDS = 20
MAX_TEXT_LENGTH = 12000
COMMON_TLDS = ("ru", "com", "org", "net", "rf", "su")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PageExtractionError(RuntimeError):
    """Raised when a page cannot be downloaded or converted to text."""


class FriendlyRequestError(RuntimeError):
    """Raised when a low-level request error is converted to a user-friendly message."""


def normalize_url(url: str) -> str:
    """Normalize a user-supplied URL and add a default scheme if needed."""
    normalized_url = url.strip()
    if not normalized_url:
        raise PageExtractionError("Укажите URL страницы.")

    if any(character.isspace() for character in normalized_url):
        raise PageExtractionError("URL не должен содержать пробелы.")

    if "://" not in normalized_url:
        normalized_url = f"https://{normalized_url}"

    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme not in {"http", "https"}:
        raise PageExtractionError("URL должен начинаться с http:// или https://.")

    if parsed_url.username or parsed_url.password:
        raise PageExtractionError("URL с логином и паролем не поддерживается.")

    hostname = parsed_url.hostname
    if not parsed_url.netloc or not hostname:
        raise PageExtractionError("Некорректный URL страницы.")

    if not _is_valid_hostname(hostname):
        raise PageExtractionError("Некорректное доменное имя или IP-адрес в URL.")

    return normalized_url


def _is_valid_hostname(hostname: str) -> bool:
    """Validate a hostname, localhost, or IP address."""
    if hostname == "localhost":
        return True

    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass

    if len(hostname) > 253 or "." not in hostname:
        return False

    labels = hostname.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not all(character.isalnum() or character == "-" for character in label):
            return False

    return True


def build_friendly_request_error(url: str, error: Exception) -> FriendlyRequestError:
    """Convert low-level request exceptions into user-friendly messages."""
    message = str(error)
    if "NameResolutionError" in message or "Failed to resolve" in message:
        suggestion = suggest_url(url)
        friendly_message = "Не удалось найти сайт. Проверьте домен."
        if suggestion:
            friendly_message = (
                f"{friendly_message} Возможно, вы имели в виду {suggestion}"
            )
        return FriendlyRequestError(friendly_message)

    return FriendlyRequestError(message)


def suggest_url(url: str) -> str | None:
    """Build a best-effort URL suggestion for common domain typos."""
    try:
        normalized_url = normalize_url(url)
    except PageExtractionError:
        return None

    parsed_url = urlparse(normalized_url)
    hostname = parsed_url.hostname
    if not hostname or hostname == "localhost":
        return None

    parts = hostname.split(".")
    if len(parts) < 2:
        return None

    domain = ".".join(parts[:-1])
    tld = parts[-1]

    if len(tld) == 1:
        suffix_candidates = [candidate for candidate in COMMON_TLDS if candidate.endswith(tld)]
        if suffix_candidates:
            preferred_candidate = sorted(
                suffix_candidates,
                key=lambda candidate: (candidate != "ru", len(candidate), candidate),
            )[0]
            suggested_host = f"{domain}.{preferred_candidate}"
            return parsed_url._replace(netloc=suggested_host).geturl()

    if tld not in COMMON_TLDS:
        candidates = difflib.get_close_matches(tld, COMMON_TLDS, n=1, cutoff=0.0)
        if candidates:
            suggested_host = f"{domain}.{candidates[0]}"
            return parsed_url._replace(netloc=suggested_host).geturl()

    if len(tld) <= 2:
        candidates = difflib.get_close_matches(tld, COMMON_TLDS, n=1, cutoff=0.0)
        if candidates and candidates[0] != tld:
            suggested_host = f"{domain}.{candidates[0]}"
            return parsed_url._replace(netloc=suggested_host).geturl()

    return None


def configure_logging() -> None:
    """Configure the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((requests.RequestException, PageExtractionError)),
)
def fetch_page_text(url: str) -> str:
    """Download a page and extract readable text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UserQuestionsAgent/1.0; +https://openai.com)"
        ),
    }
    try:
        response = requests.get(url, headers=headers, timeout=PAGE_TIMEOUT_SECONDS)
    except requests.exceptions.SSLError:
        LOGGER.warning(
            "SSL verification failed for %s. Retrying without certificate verification.",
            url,
        )
        response = requests.get(
            url,
            headers=headers,
            timeout=PAGE_TIMEOUT_SECONDS,
            verify=False,
        )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag_name in ("script", "style", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        raise PageExtractionError("Не удалось извлечь текст со страницы.")

    return cleaned_text[:MAX_TEXT_LENGTH]


def generate_questions(url: str) -> list[str]:
    """Fetch page content and generate five user questions."""
    normalized_url = normalize_url(url)
    try:
        page_text = fetch_page_text(normalized_url)
    except requests.RequestException as error:
        raise build_friendly_request_error(normalized_url, error) from error
    client = OpenAIClient()
    return client.generate_user_questions(page_text)

def render_html(
    *,
    url: str = "",
    questions: list[str] | None = None,
    error_message: str | None = None,
) -> bytes:
    """Render the web interface HTML."""
    questions_html = ""
    if questions:
        items = "\n".join(f"<li>{escape(question)}</li>" for question in questions)
        questions_html = f"""
        <section>
            <h2>Вопросы пользователей</h2>
            <ol>{items}</ol>
        </section>
        """

    error_html = ""
    if error_message:
        error_html = f'<p class="error">{escape(error_message)}</p>'

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>User Questions Agent</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f3efe7;
            --card: #fffdf8;
            --accent: #146356;
            --accent-dark: #0b4138;
            --text: #1f2933;
            --muted: #52606d;
            --error: #b42318;
            --border: #d9d3c7;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            background:
                radial-gradient(circle at top right, rgba(20, 99, 86, 0.16), transparent 32%),
                linear-gradient(180deg, #f6f0e4 0%, var(--bg) 100%);
            color: var(--text);
        }}
        main {{
            max-width: 840px;
            margin: 48px auto;
            padding: 0 20px;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 18px 40px rgba(31, 41, 51, 0.08);
        }}
        h1 {{
            margin-top: 0;
            font-size: clamp(2rem, 5vw, 3rem);
            color: var(--accent-dark);
        }}
        p {{
            color: var(--muted);
            line-height: 1.6;
        }}
        form {{
            display: grid;
            gap: 14px;
            margin-top: 24px;
        }}
        input[type="url"] {{
            width: 100%;
            padding: 14px 16px;
            border: 1px solid var(--border);
            border-radius: 12px;
            font-size: 1rem;
        }}
        button {{
            width: fit-content;
            padding: 12px 18px;
            border: none;
            border-radius: 999px;
            background: var(--accent);
            color: white;
            cursor: pointer;
            font-size: 1rem;
        }}
        button:hover {{
            background: var(--accent-dark);
        }}
        h2 {{
            margin-top: 28px;
            color: var(--accent-dark);
        }}
        ol {{
            padding-left: 20px;
            line-height: 1.7;
        }}
        .error {{
            color: var(--error);
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <main>
        <section class="card">
            <h1>Генератор вопросов по сайту</h1>
            <p>Введите URL страницы, и агент сформирует 5 логичных вопросов, которые мог бы задать пользователь после чтения текста.</p>
            {error_html}
            <form method="post">
                <label for="url">URL страницы</label>
                <input
                    id="url"
                    name="url"
                    type="text"
                    inputmode="url"
                    placeholder="example.com или https://example.com"
                    value="{escape(url)}"
                    required
                >
                <button type="submit">Сгенерировать вопросы</button>
            </form>
            {questions_html}
        </section>
    </main>
</body>
</html>"""
    return html.encode("utf-8")


class QuestionsWebHandler(BaseHTTPRequestHandler):
    """Minimal web interface for the questions agent."""

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "user-questions-agent",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        if parsed_url.path == "/api/questions":
            query_params = parse_qs(parsed_url.query)
            url = query_params.get("url", [""])[0].strip()
            self._handle_api_request(url)
            return

        self._send_html(render_html())

    def do_POST(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/api/questions":
            self._handle_api_post()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(body)
        url = form_data.get("url", [""])[0].strip()

        if not url:
            self._send_html(
                render_html(error_message="Укажите URL страницы."),
                status_code=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            questions = generate_questions(url)
        except (
            FriendlyRequestError,
            PageExtractionError,
            requests.RequestException,
            RetryError,
            OpenAIConfigurationError,
            OpenAIResponseError,
        ) as error:
            LOGGER.exception("Web request failed for URL %s", url)
            self._send_html(
                render_html(
                    url=url,
                    error_message=f"Ошибка при обработке страницы: {error}",
                ),
                status_code=HTTPStatus.BAD_GATEWAY,
            )
            return

        self._send_html(render_html(url=url, questions=questions))

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _handle_api_post(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")

        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_json(
                    {"error": "Некорректный JSON в теле запроса."},
                    status_code=HTTPStatus.BAD_REQUEST,
                )
                return
            url = str(payload.get("url", "")).strip()
        else:
            form_data = parse_qs(body)
            url = form_data.get("url", [""])[0].strip()

        self._handle_api_request(url)

    def _handle_api_request(self, url: str) -> None:
        if not url:
            self._send_json(
                {"error": "Укажите параметр url."},
                status_code=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            questions = generate_questions(url)
        except (
            PageExtractionError,
            requests.RequestException,
            RetryError,
            OpenAIConfigurationError,
            OpenAIResponseError,
        ) as error:
            LOGGER.exception("API request failed for URL %s", url)
            self._send_json(
                {
                    "error": "Ошибка при обработке страницы.",
                    "details": str(error),
                    "url": url,
                },
                status_code=HTTPStatus.BAD_GATEWAY,
            )
            return

        self._send_json({"url": url, "questions": questions})

    def _send_html(
        self,
        html: bytes,
        status_code: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _send_json(
        self,
        payload: dict[str, Any],
        status_code: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_web_server(host: str, port: int) -> None:
    """Start the built-in web server."""
    server = ThreadingHTTPServer((host, port), QuestionsWebHandler)
    LOGGER.info("Web app is running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Web app stopped by user.")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Генерирует 5 вопросов пользователя по содержимому веб-страницы. "
            "CLI: python agent.py https://example.com "
            "или веб-режим: python agent.py --web"
        ),
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL страницы для анализа.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Запустить встроенное веб-приложение.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Хост для веб-приложения. По умолчанию 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для веб-приложения. По умолчанию 8000.",
    )
    return parser


def main() -> int:
    """Application entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.web:
        run_web_server(args.host, args.port)
        return 0

    if not args.url:
        parser.print_help()
        return 1

    try:
        questions = generate_questions(args.url)
    except (
        FriendlyRequestError,
        PageExtractionError,
        requests.RequestException,
        RetryError,
        OpenAIConfigurationError,
        OpenAIResponseError,
    ) as error:
        LOGGER.error("Ошибка: %s", error)
        return 1

    print(json.dumps(questions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
