FROM python:3.11-slim

WORKDIR /app

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Делаем скрипт исполняемым
RUN chmod +x wait-for-postgres.sh

# Используем скрипт ожидания как точку входа
ENTRYPOINT ["./wait-for-postgres.sh", "postgres"]