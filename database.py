from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, Table, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import logging
from typing import List, Dict, Optional
import json
import time
from sqlalchemy.exc import OperationalError
import pandas as pd
from pathlib import Path

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

# Новые таблицы для нормализованных данных
class NormalizedName(Base):
    """Справочник нормализованных названий"""
    __tablename__ = "normalized_names"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    
    # Связь с основным словарем
    dictionary_entries = relationship("Dictionary", back_populates="normalized_name_ref")

class GbzName(Base):
    """Справочник названий ГБЗ"""
    __tablename__ = "gbz_names"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    
    # Связь с основным словарем
    dictionary_entries = relationship("Dictionary", back_populates="gbz_name_ref")

class GroupName(Base):
    """Справочник групп недр"""
    __tablename__ = "group_names"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    
    # Связь с основным словарем
    dictionary_entries = relationship("Dictionary", back_populates="group_name_ref")

class MeasurementUnit(Base):
    """Справочник единиц измерения"""
    __tablename__ = "measurement_units"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    
    # Связи с основным словарем
    primary_units = relationship("Dictionary", foreign_keys="Dictionary.measurement_unit_id", back_populates="measurement_unit_ref")
    alternative_units = relationship("Dictionary", foreign_keys="Dictionary.measurement_unit_alt_id", back_populates="measurement_unit_alt_ref")

class Dictionary(Base):
    """Основной словарь"""
    __tablename__ = "dictionary"

    id = Column(Integer, primary_key=True)
    unique_key = Column(String, unique=True, index=True)
    
    # Внешние ключи
    normalized_name_id = Column(Integer, ForeignKey('normalized_names.id'), nullable=False)
    gbz_name_id = Column(Integer, ForeignKey('gbz_names.id'), nullable=False)
    group_name_id = Column(Integer, ForeignKey('group_names.id'), nullable=False)
    measurement_unit_id = Column(Integer, ForeignKey('measurement_units.id'), nullable=False)
    measurement_unit_alt_id = Column(Integer, ForeignKey('measurement_units.id'), nullable=True)
    
    # Связи с справочными таблицами
    normalized_name_ref = relationship("NormalizedName", back_populates="dictionary_entries")
    gbz_name_ref = relationship("GbzName", back_populates="dictionary_entries")
    group_name_ref = relationship("GroupName", back_populates="dictionary_entries")
    measurement_unit_ref = relationship("MeasurementUnit", foreign_keys=[measurement_unit_id], back_populates="primary_units")
    measurement_unit_alt_ref = relationship("MeasurementUnit", foreign_keys=[measurement_unit_alt_id], back_populates="alternative_units")
    
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
        self.Dictionary = Dictionary
        self.NormalizedName = NormalizedName
        self.GbzName = GbzName
        self.GroupName = GroupName
        self.MeasurementUnit = MeasurementUnit
        self.Variation = Variation
        self.UnclassifiedTerm = UnclassifiedTerm
        Base.metadata.create_all(bind=engine)

    def get_connection(self):
        """Контекстный менеджер для соединения с базой данных"""
        try:
            conn = engine.connect()
            return conn
        except Exception as e:
            logging.error(f"Error getting database connection: {e}")
            raise

    def get_db(self):
        """Получение сессии базы данных"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def add_dictionary_entry(self, entry_data: Dict) -> Optional[Dictionary]:
        """Добавление или обновение записи в основном словаре"""
        try:
            with self.SessionLocal() as db:
                # Получаем или создаем записи в справочных таблицах
                normalized_name = self._get_or_create(db, NormalizedName, 
                    name=entry_data['normalized_name'])
                gbz_name = self._get_or_create(db, GbzName, 
                    name=entry_data['gbz_name'])
                group_name = self._get_or_create(db, GroupName, 
                    name=entry_data['group_name'])
                measurement_unit = self._get_or_create(db, MeasurementUnit, 
                    name=entry_data['measurement_unit'])
                measurement_unit_alt = None
                if entry_data.get('measurement_unit_alt'):
                    measurement_unit_alt = self._get_or_create(db, MeasurementUnit, 
                        name=entry_data['measurement_unit_alt'])

                # Проверяем существование записи
                existing_entry = db.query(Dictionary).filter_by(
                    unique_key=entry_data['unique_key']
                ).first()
                
                if existing_entry:
                    # Обновляем существющую запись
                    existing_entry.normalized_name_id = normalized_name.id
                    existing_entry.gbz_name_id = gbz_name.id
                    existing_entry.group_name_id = group_name.id
                    existing_entry.measurement_unit_id = measurement_unit.id
                    existing_entry.measurement_unit_alt_id = measurement_unit_alt.id if measurement_unit_alt else None
                    db.commit()
                    db.refresh(existing_entry)
                    return existing_entry
                else:
                    # Создаем новую запись
                    dictionary_entry = Dictionary(
                        unique_key=entry_data['unique_key'],
                        normalized_name_id=normalized_name.id,
                        gbz_name_id=gbz_name.id,
                        group_name_id=group_name.id,
                        measurement_unit_id=measurement_unit.id,
                        measurement_unit_alt_id=measurement_unit_alt.id if measurement_unit_alt else None
                    )
                    db.add(dictionary_entry)
                    db.commit()
                    db.refresh(dictionary_entry)
                    return dictionary_entry
        except Exception as e:
            logging.error(f"Error adding/updating dictionary entry: {e}")
            db.rollback()
            return None

    def _get_or_create(self, session, model, **kwargs):
        """Получает существующую запись или создает новую"""
        instance = session.query(model).filter_by(**kwargs).first()
        if not instance:
            instance = model(**kwargs)
            session.add(instance)
            session.flush()
        return instance

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
                        'normalized_name_for_display': entry.normalized_name_ref.name,
                        'pi_name_gbz_tbz': entry.gbz_name_ref.name,
                        'pi_group_is_nedra': entry.group_name_ref.name,
                        'pi_measurement_unit': entry.measurement_unit_ref.name,
                        'pi_measurement_unit_alternative': entry.measurement_unit_alt_ref.name if entry.measurement_unit_alt_ref else ''
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
                normalized_names = db.query(NormalizedName.name).distinct().all()
                gbz_names = db.query(GbzName.name).distinct().all()
                group_names = db.query(GroupName.name).distinct().all()
                measurement_units = db.query(MeasurementUnit.name).distinct().all()
                
                options = {
                    'normalized_names': sorted([r[0] for r in normalized_names]),
                    'gbz_names': sorted([r[0] for r in gbz_names]),
                    'groups': sorted([r[0] for r in group_names]),
                    'measurement_units': sorted([r[0] for r in measurement_units])
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

    def get_dictionary_entries(self) -> List[Dict]:
        """Получение всех записей справочника"""
        try:
            with self.SessionLocal() as db:
                entries = db.query(Dictionary).all()
                return [
                    {
                        'unique_key': entry.unique_key,
                        'normalized_name': entry.normalized_name_ref.name,
                        'gbz_name': entry.gbz_name_ref.name,
                        'group_name': entry.group_name_ref.name,
                        'measurement_unit': entry.measurement_unit_ref.name,
                        'measurement_unit_alt': entry.measurement_unit_alt_ref.name if entry.measurement_unit_alt_ref else ''
                    }
                    for entry in entries
                ]
        except Exception as e:
            logging.error(f"Error getting dictionary entries: {e}")
            return []

    def update_dictionary_entry(self, entry_data: Dict) -> bool:
        """Обновление записи в справочнике"""
        try:
            with self.SessionLocal() as db:
                # Получаем или создаем записи в справочных таблицах
                normalized_name = self._get_or_create(db, self.NormalizedName, 
                    name=entry_data['normalized_name'])
                gbz_name = self._get_or_create(db, self.GbzName, 
                    name=entry_data['gbz_name'])
                group_name = self._get_or_create(db, self.GroupName, 
                    name=entry_data['group_name'])
                measurement_unit = self._get_or_create(db, self.MeasurementUnit, 
                    name=entry_data['measurement_unit'])
                measurement_unit_alt = None
                if entry_data.get('measurement_unit_alt'):
                    measurement_unit_alt = self._get_or_create(db, self.MeasurementUnit, 
                        name=entry_data['measurement_unit_alt'])

                entry = db.query(self.Dictionary).filter_by(
                    unique_key=entry_data['unique_key']
                ).first()
                
                if entry:
                    # Обновляем связи
                    entry.normalized_name_id = normalized_name.id
                    entry.gbz_name_id = gbz_name.id
                    entry.group_name_id = group_name.id
                    entry.measurement_unit_id = measurement_unit.id
                    entry.measurement_unit_alt_id = measurement_unit_alt.id if measurement_unit_alt else None
                    
                    # Обновляем связанные вариации
                    for variation in entry.variations:
                        variation.dictionary_id = entry.id
                    
                    db.commit()
                    return True
                return False
        except Exception as e:
            logging.error(f"Error updating dictionary entry: {e}")
            db.rollback()
            return False

    def delete_dictionary_entry(self, unique_key: str) -> bool:
        """Удаление записи из справочника"""
        try:
            with self.SessionLocal() as db:
                # Сначала удаляем все связанные вариации
                entry = db.query(Dictionary).filter_by(unique_key=unique_key).first()
                if entry:
                    # SQLAlchemy автоматически удалит связанные вариации благодаря cascade
                    db.delete(entry)
                    db.commit()
                    return True
                return False
        except Exception as e:
            logging.error(f"Error deleting dictionary entry: {e}")
            db.rollback()
            return False

    def export_dictionary(self, filename: str) -> str:
        """Экспорт справочника в Excel"""
        try:
            entries = self.get_dictionary_entries()
            df = pd.DataFrame(entries)
            
            # Создаем директорию для экспорта, если её нет
            export_dir = Path('exports')
            export_dir.mkdir(exist_ok=True)
            
            # Формируем путь к файлу
            filepath = export_dir / filename
            
            # Сохраняем в Excel
            df.to_excel(filepath, index=False)
            
            return str(filepath)
        except Exception as e:
            logging.error(f"Error exporting dictionary: {e}")
            raise

    def get_dictionary_entries_count(self) -> int:
        """Возвращает общее количество записей в словаре"""
        try:
            with self.SessionLocal() as session:
                return session.query(Dictionary).count()
        except Exception as e:
            logging.error(f"Error getting dictionary entries count: {e}")
            raise

    def get_dictionary_entries_paginated(self, page: int = 1, limit: int = 50) -> List[Dict]:
        """Возвращает записи словаря с пагинацией"""
        try:
            offset = (page - 1) * limit
            with self.SessionLocal() as session:
                entries = session.query(Dictionary).order_by(Dictionary.normalized_name)\
                    .offset(offset).limit(limit).all()
                
                return [{
                    'id': entry.id,
                    'term': entry.unique_key,
                    'normalized_name': entry.normalized_name,
                    'gbz_name': entry.gbz_name,
                    'group_name': entry.group_name,
                    'measurement_unit': entry.measurement_unit,
                    'measurement_unit_alt': entry.measurement_unit_alt or ''
                } for entry in entries]
        except Exception as e:
            logging.error(f"Error getting paginated dictionary entries: {e}")
            raise

    def remove_all_unclassified_terms(self) -> bool:
        """Удаляет все неклассифицированные термины из базы данных"""
        try:
            with self.SessionLocal() as db:
                # Используем SQLAlchemy ORM вместо прямого SQL
                db.query(UnclassifiedTerm).delete()
                db.commit()
                return True
        except Exception as e:
            logging.error(f"Error removing all unclassified terms: {e}")
            return False

    def get_all_terms(self) -> List[str]:
        """Получает все термины из базы данных"""
        try:
            with self.SessionLocal() as session:
                # Получаем термины из таблицы вариаций
                variations = session.query(Variation.variant).distinct().all()
                terms = [row[0] for row in variations]
                
                # Получаем также нормализованные названия
                normalized = session.query(NormalizedName.name).distinct().all()
                terms.extend([row[0] for row in normalized])
                
                # Убираем дубликаты и возвращаем
                return list(set(terms))
        except Exception as e:
            logging.error(f"Error getting all terms: {str(e)}")
            return []