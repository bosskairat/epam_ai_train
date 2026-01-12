FROM python:3.12-slim

# Системные зависимости для Playwright
RUN apt-get update && apt-get install -y \
    curl wget gnupg ca-certificates libnss3 libatk1.0-0 libcups2 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем Playwright браузеры
RUN python -m playwright install chromium

# Копируем весь код в контейнер
COPY ./app ./app

# Устанавливаем PYTHONPATH, чтобы Python видел модуль
ENV PYTHONPATH=/app

# expose port
EXPOSE 8000

# запускаем FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
