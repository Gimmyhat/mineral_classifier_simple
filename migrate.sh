#!/bin/bash

# Ждем, пока PostgreSQL будет готов
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done

echo "PostgreSQL started"

# Инициализируем базу данных
python init_db.py

# Запускаем миграцию
python migrate_data.py

echo "Migration completed" 