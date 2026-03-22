# User Questions Agent

Агент загружает текст страницы по URL и генерирует 5 логичных вопросов, которые мог бы задать пользователь после чтения сайта.

Проект поддерживает:
- запуск из командной строки
- встроенное веб-приложение
- JSON API для интеграций

## Стек

- Python 3.12+
- `openai`
- `requests`
- `beautifulsoup4`
- `tenacity`
- `python-dotenv`

## Структура

```text
user_questions/
├── agent.py
├── openai_module.py
├── requirements.txt
├── .env.example
└── README.md
```

## Быстрый старт

1. Создайте виртуальное окружение.
2. Установите зависимости.
3. Создайте `.env` на основе `.env.example`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Пример `.env`:

```env
BASE_URL=https://routerai.ru/api/v1
API_KEY=your_api_key_here
MODEL=mistralai/mistral-small-2603
MAX_TOKENS=4096
```

## CLI

Запуск с полным URL:

```bash
python agent.py "https://example.com"
```

Запуск только с доменом:

```bash
python agent.py "example.com"
```

Скрипт автоматически добавит `https://`, если схема не указана.

## Веб-приложение

```bash
python agent.py --web
```

Этот режим удобен для локальной разработки и быстрых проверок.

По умолчанию приложение будет доступно по адресу `http://127.0.0.1:8000`.

Можно указать свой хост и порт:

```bash
python agent.py --web --host 0.0.0.0 --port 8080
```

## Docker

Локальный запуск через Docker Compose:

```bash
docker compose up --build
```

После старта приложение будет доступно по адресу `http://127.0.0.1:8000`.

Остановка:

```bash
docker compose down
```

Если нужен только образ:

```bash
docker build -t user-questions-agent .
docker run --rm -p 8000:8000 --env-file .env user-questions-agent
```

Контейнер по умолчанию запускает production-сервер через `gunicorn` + `uvicorn` worker:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 agent:app
```

## Production server

Для production-режима приложение теперь экспортирует ASGI app:

```bash
agent:app
```

Локальный запуск через `uvicorn`:

```bash
uvicorn agent:app --host 0.0.0.0 --port 8000
```

Запуск через `gunicorn`:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 agent:app
```

## JSON API

Доступные эндпоинты:

```text
GET  /api/health
GET  /api/questions?url=https://example.com
POST /api/questions
```

Проверка состояния:

```bash
curl "http://127.0.0.1:8000/api/health"
```

Запрос вопросов через `GET`:

```bash
curl "http://127.0.0.1:8000/api/questions?url=example.com"
```

Запрос вопросов через `POST`:

```bash
curl -X POST "http://127.0.0.1:8000/api/questions" ^
  -H "Content-Type: application/json" ^
  -d "{\"url\":\"example.com\"}"
```

Пример ответа:

```json
{
  "url": "example.com",
  "questions": [
    "Вопрос 1",
    "Вопрос 2",
    "Вопрос 3",
    "Вопрос 4",
    "Вопрос 5"
  ]
}
```

## Как работает

1. Агент получает URL.
2. Загружает HTML страницы.
3. Извлекает чистый текст через `BeautifulSoup`.
4. Передает текст в модель через `openai_module.py`.
5. Возвращает 5 вопросов.

## Обработка ошибок

- Повторные попытки при временных HTTP-сбоях и ошибках модели.
- Проверка валидности URL до сетевого запроса.
- Понятные сообщения при пустой странице, неправильном `.env` и некорректном ответе модели.
- Дружелюбная подсказка, если домен не найден.

## Публикация на GitHub

- Не публикуйте реальный `.env`.
- Используйте `.env.example` как шаблон.
- Перед публикацией публичного репозитория лучше перевыпустить API-ключ, если он уже хранился локально в рабочей папке.
