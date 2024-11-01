#!/bin/bash
# wait-for-postgres.sh

set -e

host="$1"
shift

echo "Checking PostgreSQL connection..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

>&2 echo "PostgreSQL is up - waiting for it to be fully ready..."
sleep 10  # Увеличиваем задержку до 10 секунд

>&2 echo "PostgreSQL is up - initializing database"
python init_db.py

>&2 echo "Running data migration"
python migrate_data.py

>&2 echo "Starting web application"
exec python -m uvicorn main:app --host 0.0.0.0 --port 8001