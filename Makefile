.PHONY: clean-db rebuild-db

clean-db:
	docker-compose exec web python clean_db.py

rebuild-db: clean-db
	docker-compose exec web python migrate_data.py

reset-all:
	docker-compose down -v
	docker-compose up -d --build
	sleep 5  # Ждем, пока база данных запустится
	docker-compose exec web python clean_db.py
	docker-compose exec web python migrate_data.py 