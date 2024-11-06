from database import Base, engine
import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_database():
    """Удаляет все таблицы из базы данных"""
    try:
        # Получаем список всех таблиц
        with engine.connect() as connection:
            # Отключаем внешние ключи на время удаления
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            
            # Удаляем все таблицы
            Base.metadata.drop_all(bind=engine)
            
            # Включаем обратно внешние ключи
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            
        logger.info("All tables have been dropped successfully")
        
        # Создаем таблицы заново
        Base.metadata.create_all(bind=engine)
        logger.info("New tables have been created")
        
        return True
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        return False

if __name__ == "__main__":
    clean_database() 