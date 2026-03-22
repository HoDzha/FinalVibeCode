from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 600
SYSTEM_PROMPT = (
    "Ты в роли пользователя. "
    "Прочитай текст страницы и верни ровно 5 логичных, содержательных вопросов, "
    "которые мог бы задать пользователь после чтения. "
    "Отвечай только JSON-массивом из 5 строк без пояснений."
)

load_dotenv()


class OpenAIConfigurationError(RuntimeError):
    """Raised when OpenAI client configuration is invalid."""


class OpenAIResponseError(RuntimeError):
    """Raised when the model response cannot be parsed."""


class OpenAIClient:
    """Centralized wrapper around OpenAI-compatible requests."""

    def __init__(self) -> None:
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigurationError(
                "Не найден API ключ. Укажите API_KEY или OPENAI_API_KEY в .env."
            )

        base_url = os.getenv("BASE_URL") or None
        model = os.getenv("MODEL", DEFAULT_MODEL)
        max_tokens_raw = os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))

        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as error:
            raise OpenAIConfigurationError(
                "MAX_TOKENS в .env должен быть целым числом."
            ) from error

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model
        self._max_tokens = max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def generate_user_questions(self, page_text: str) -> list[str]:
        """Generate five user questions from page text."""
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.7,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Текст страницы:\n"
                        f"{page_text}\n\n"
                        "Верни JSON-объект вида "
                        '{"questions":["вопрос 1","вопрос 2","вопрос 3","вопрос 4","вопрос 5"]}.'
                    ),
                },
            ],
        )

        content = self._extract_content(response)
        return self._parse_questions(content)

    @staticmethod
    def _extract_content(response: Any) -> str:
        message = response.choices[0].message
        content = message.content
        if not content:
            raise OpenAIResponseError("Модель вернула пустой ответ.")
        return content

    @staticmethod
    def _parse_questions(content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise OpenAIResponseError(
                "Не удалось разобрать JSON из ответа модели."
            ) from error

        if isinstance(payload, list):
            questions = payload
        else:
            questions = payload.get("questions")

        if not isinstance(questions, list):
            raise OpenAIResponseError("Ответ модели не содержит список questions.")

        cleaned_questions = [
            question.strip()
            for question in questions
            if isinstance(question, str) and question.strip()
        ]

        if len(cleaned_questions) != 5:
            raise OpenAIResponseError("Модель должна вернуть ровно 5 вопросов.")

        return cleaned_questions
