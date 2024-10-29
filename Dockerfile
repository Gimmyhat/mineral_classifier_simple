FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY *.py ./

# Копируем директории
COPY templates/ templates/
COPY static/ static/
COPY data/ data/

# Создаем необходимые директории
RUN mkdir -p static results templates logs

# Открываем порт
EXPOSE 8001

CMD ["python", "main.py"]
