FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py openai_module.py ./

EXPOSE 8000

CMD ["python", "agent.py", "--web", "--host", "0.0.0.0", "--port", "8000"]
