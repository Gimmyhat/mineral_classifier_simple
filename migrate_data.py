import logging
from database import DatabaseManager, Dictionary, Variation
import pandas as pd
from sqlalchemy.exc import IntegrityError

logging.basicConfig(level=logging.DEBUG)

def migrate_data(dictionary_path: str, variations_path: str):
    """Миграция данных из Excel в PostgreSQL"""
    try:
        db_manager = DatabaseManager()
        
        # Загружаем данные из Excel
        df_dict = pd.read_excel(dictionary_path)
        df_var = pd.read_excel(variations_path)
        
        # Создаем сессию
        with db_manager.SessionLocal() as session:
            # Загружаем основной словарь
            for _, row in df_dict.iterrows():
                try:
                    # Проверяем существование записи
                    existing_entry = session.query(Dictionary).filter_by(
                        unique_key=row['unique_key']
                    ).first()
                    
                    if not existing_entry:
                        # Создаем новую запись
                        dictionary_entry = Dictionary(
                            unique_key=row['unique_key'],
                            normalized_name=row['normalized_name'],
                            gbz_name=row['gbz_name'],
                            group_name=row['group_name'],
                            measurement_unit=row['measurement_unit'],
                            measurement_unit_alt=row['measurement_unit_alt'] if pd.notna(row['measurement_unit_alt']) else ''
                        )
                        session.add(dictionary_entry)
                        try:
                            session.flush()  # Пробуем сохранить запись
                            
                            # Добавляем вариации для этой записи
                            variations = df_var[df_var['unique_key'] == row['unique_key']]
                            for _, var_row in variations.iterrows():
                                # Проверяем существование вариации
                                existing_variation = session.query(Variation).filter_by(
                                    variant=var_row['variant'].lower()
                                ).first()
                                
                                if not existing_variation:
                                    variation = Variation(
                                        variant=var_row['variant'].lower(),
                                        dictionary_id=dictionary_entry.id
                                    )
                                    session.add(variation)
                            
                        except IntegrityError as e:
                            session.rollback()
                            logging.warning(f"Duplicate entry for {row['unique_key']}: {e}")
                            continue
                    else:
                        logging.debug(f"Entry already exists for {row['unique_key']}")
                        
                        # Обновляем вариации для существующей записи
                        variations = df_var[df_var['unique_key'] == row['unique_key']]
                        for _, var_row in variations.iterrows():
                            existing_variation = session.query(Variation).filter_by(
                                variant=var_row['variant'].lower()
                            ).first()
                            
                            if not existing_variation:
                                variation = Variation(
                                    variant=var_row['variant'].lower(),
                                    dictionary_id=existing_entry.id
                                )
                                session.add(variation)
                
                except Exception as e:
                    logging.error(f"Error processing entry {row['unique_key']}: {e}")
                    session.rollback()
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