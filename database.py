from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, Table, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import logging
from typing import List, Dict, Optional
import json
import time
from sqlalchemy.exc import OperationalError

# Создаем базовый класс для моделей
Base = declarative_base()

# Создаем соединение с базой данных
DATABASE_URL = "postgresql://user:password@postgres:5432/minerals"

def create_db_engine():
    """Создает engine с повторными попытками подключения"""
    retries = 5
    while retries > 0:
        try:
            engine = create_engine(DATABASE_URL)
            # Проверяем подключение
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except OperationalError as e:
            if retries > 1:
                retries -= 1
                logging.warning(f"Failed to connect to database. Retrying... ({retries} attempts left)")
                time.sleep(2)
            else:
                logging.error("Failed to connect to database after all retries")
                raise
        except Exception as e:
            logging.error(f"Unexpected error while connecting to database: {e}")
            raise

engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine)

class Dictionary(Base):
    """Основной словарь"""
    __tablename__ = "dictionary"

    id = Column(Integer, primary_key=True)
    unique_key = Column(String, unique=True, index=True)
    normalized_name = Column(String)
    gbz_name = Column(String)
    group_name = Column(String)
    measurement_unit = Column(String)
    measurement_unit_alt = Column(String)
    
    # Связь с вариациями
    variations = relationship("Variation", back_populates="dictionary_entry")

class Variation(Base):
    """Вариации написания"""
    __tablename__ = "variations"

    id = Column(Integer, primary_key=True)
    variant = Column(String, unique=True, index=True)
    dictionary_id = Column(Integer, ForeignKey('dictionary.id'))
    
    # Связь с основным словарем
    dictionary_entry = relationship("Dictionary", back_populates="variations")

class UnclassifiedTerm(Base):
    """Неклассифицированные термины"""
    __tablename__ = "unclassified_terms"

    id = Column(Integer, primary_key=True)
    term = Column(String, unique=True)

class DatabaseManager:
    def __init__(self):
        self.SessionLocal = SessionLocal
        Base.metadata.create_all(bind=engine)

    def get_db(self):
        """Получение сессии базы данных"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def add_dictionary_entry(self, entry_data: Dict) -> Optional[Dictionary]:
        """Добавление или обновление записи в основном словаре"""
        try:
            with self.SessionLocal() as db:
                # Проверяем существование записи
                existing_entry = db.query(Dictionary).filter_by(
                    unique_key=entry_data['unique_key']
                ).first()
                
                if existing_entry:
                    # Обновляем существующую запись
                    existing_entry.normalized_name = entry_data['normalized_name']
                    existing_entry.gbz_name = entry_data['gbz_name']
                    existing_entry.group_name = entry_data['group_name']
                    existing_entry.measurement_unit = entry_data['measurement_unit']
                    existing_entry.measurement_unit_alt = entry_data.get('measurement_unit_alt', '')
                    db.commit()
                    db.refresh(existing_entry)
                    return existing_entry
                else:
                    # Создаем новую запись
                    dictionary_entry = Dictionary(
                        unique_key=entry_data['unique_key'],
                        normalized_name=entry_data['normalized_name'],
                        gbz_name=entry_data['gbz_name'],
                        group_name=entry_data['group_name'],
                        measurement_unit=entry_data['measurement_unit'],
                        measurement_unit_alt=entry_data.get('measurement_unit_alt', '')
                    )
                    db.add(dictionary_entry)
                    db.commit()
                    db.refresh(dictionary_entry)
                    return dictionary_entry
        except Exception as e:
            logging.error(f"Error adding/updating dictionary entry: {e}")
            db.rollback()
            return None

    def add_variation(self, variant: str, dictionary_id: int) -> Optional[Variation]:
        """Добавление или обновление вариации"""
        try:
            with self.SessionLocal() as db:
                # Проверяем существование вариации
                existing_variation = db.query(Variation).filter_by(
                    variant=variant.lower()
                ).first()
                
                if existing_variation:
                    # Обновляем существующую вариацию
                    existing_variation.dictionary_id = dictionary_id
                    db.commit()
                    db.refresh(existing_variation)
                    return existing_variation
                else:
                    # Создаем новую вариацию
                    variation = Variation(
                        variant=variant.lower(),
                        dictionary_id=dictionary_id
                    )
                    db.add(variation)
                    db.commit()
                    db.refresh(variation)
                    return variation
        except Exception as e:
            logging.error(f"Error adding/updating variation: {e}")
            db.rollback()
            return None

    def get_mapping(self, variant: str) -> Optional[Dict]:
        """Получение маппинга по вариации"""
        try:
            with self.SessionLocal() as db:
                variation = db.query(Variation).filter(
                    Variation.variant == variant.lower()
                ).first()
                
                if variation and variation.dictionary_entry:
                    entry = variation.dictionary_entry
                    return {
                        'normalized_name_for_display': entry.normalized_name,
                        'pi_name_gbz_tbz': entry.gbz_name,
                        'pi_group_is_nedra': entry.group_name,
                        'pi_measurement_unit': entry.measurement_unit,
                        'pi_measurement_unit_alternative': entry.measurement_unit_alt
                    }
                return None
        except Exception as e:
            logging.error(f"Error getting mapping: {e}")
            return None

    def add_unclassified_term(self, term: str) -> bool:
        """Добавление неклассифицированного термина"""
        try:
            with self.SessionLocal() as db:
                # Проверяем, нет ли уже такого термина
                existing = db.query(UnclassifiedTerm).filter(
                    UnclassifiedTerm.term == term
                ).first()
                
                if not existing:
                    unclassified = UnclassifiedTerm(term=term)
                    db.add(unclassified)
                    db.commit()
                return True
        except Exception as e:
            logging.error(f"Error adding unclassified term: {e}")
            return False

    def get_unclassified_terms(self) -> List[str]:
        """Получение списка неклассифицированных терминов"""
        try:
            with self.SessionLocal() as db:
                terms = db.query(UnclassifiedTerm).all()
                return [term.term for term in terms]
        except Exception as e:
            logging.error(f"Error getting unclassified terms: {e}")
            return []

    def get_classification_options(self) -> Dict:
        """Получение доступных вариантов классификации"""
        try:
            with self.SessionLocal() as db:
                entries = db.query(Dictionary).all()
                
                options = {
                    'normalized_names': sorted(list(set(e.normalized_name for e in entries))),
                    'gbz_names': sorted(list(set(e.gbz_name for e in entries))),
                    'groups': sorted(list(set(e.group_name for e in entries))),
                    'measurement_units': sorted(list(set(e.measurement_unit for e in entries)))
                }
                
                return options
        except Exception as e:
            logging.error(f"Error getting classification options: {e}")
            return {
                'normalized_names': [],
                'gbz_names': [],
                'groups': [],
                'measurement_units': []
            }

    def remove_unclassified_term(self, term: str) -> bool:
        """Удаление термина из неклассифицированных"""
        try:
            with self.SessionLocal() as db:
                db.query(UnclassifiedTerm).filter(
                    UnclassifiedTerm.term == term
                ).delete()
                db.commit()
                return True
        except Exception as e:
            logging.error(f"Error removing unclassified term: {e}")
            return False

    def check_health(self) -> bool:
        """Проверка состояния подключения к базе данных"""
        try:
            with self.SessionLocal() as db:
                db.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logging.error(f"Database health check failed: {e}")
            return False