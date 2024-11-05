import logging
from database import (
    DatabaseManager, Dictionary, NormalizedName, GbzName, 
    GroupName, MeasurementUnit, Variation
)
import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

logging.basicConfig(level=logging.DEBUG)

def migrate_data(dictionary_path: str, variations_path: str):
    """Миграция данных из Excel в новую структуру PostgreSQL"""
    try:
        db_manager = DatabaseManager()
        
        # Загружаем данные из Excel
        df_dict = pd.read_excel(dictionary_path)
        df_var = pd.read_excel(variations_path)
        
        with db_manager.SessionLocal() as session:
            # Сначала очищаем все таблицы
            session.execute(text('TRUNCATE TABLE variations CASCADE'))
            session.execute(text('TRUNCATE TABLE dictionary CASCADE'))
            session.execute(text('TRUNCATE TABLE normalized_names CASCADE'))
            session.execute(text('TRUNCATE TABLE gbz_names CASCADE'))
            session.execute(text('TRUNCATE TABLE group_names CASCADE'))
            session.execute(text('TRUNCATE TABLE measurement_units CASCADE'))
            session.commit()
            
            # Создаем справочники
            for _, row in df_dict.iterrows():
                try:
                    # Проверяем наличие всех необходимых колонок
                    required_columns = ['unique_key', 'normalized_name', 'gbz_name', 
                                         'group_name', 'measurement_unit']
                    if not all(col in row.index for col in required_columns):
                        logging.error(f"Missing required columns in row: {row}")
                        continue

                    # Получаем или создаем записи в справочных таблицах
                    normalized_name = db_manager._get_or_create(session, NormalizedName, 
                        name=str(row['normalized_name']))
                    gbz_name = db_manager._get_or_create(session, GbzName, 
                        name=str(row['gbz_name']))
                    group_name = db_manager._get_or_create(session, GroupName, 
                        name=str(row['group_name']))
                    measurement_unit = db_manager._get_or_create(session, MeasurementUnit, 
                        name=str(row['measurement_unit']))
                    measurement_unit_alt = None
                    if 'measurement_unit_alt' in row and pd.notna(row['measurement_unit_alt']):
                        measurement_unit_alt = db_manager._get_or_create(session, MeasurementUnit, 
                            name=str(row['measurement_unit_alt']))

                    # Создаем запись в словаре
                    dictionary_entry = Dictionary(
                        unique_key=str(row['unique_key']),
                        normalized_name_id=normalized_name.id,
                        gbz_name_id=gbz_name.id,
                        group_name_id=group_name.id,
                        measurement_unit_id=measurement_unit.id,
                        measurement_unit_alt_id=measurement_unit_alt.id if measurement_unit_alt else None
                    )
                    session.add(dictionary_entry)
                    session.flush()

                    # Добавляем вариации
                    variations = df_var[df_var['unique_key'] == row['unique_key']]
                    for _, var_row in variations.iterrows():
                        if pd.notna(var_row['variant']):
                            variation = Variation(
                                variant=str(var_row['variant']).lower(),
                                dictionary_id=dictionary_entry.id
                            )
                            session.add(variation)

                except IntegrityError as e:
                    session.rollback()
                    logging.warning(f"Duplicate entry for {row['unique_key']}: {e}")
                    continue
                except Exception as e:
                    session.rollback()
                    logging.error(f"Error processing entry {row['unique_key']}: {e}")
                    continue

            try:
                session.commit()
                logging.info("Migration completed successfully")
            except Exception as e:
                session.rollback()
                logging.error(f"Error committing changes: {e}")
                raise

    except Exception as e:
        logging.error(f"Error during migration: {e}")
        raise

if __name__ == "__main__":
    migrate_data('data/dictionary.xlsx', 'data/variations.xlsx')