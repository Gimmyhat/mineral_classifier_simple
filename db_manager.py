import sqlite3
from typing import Dict, List
import logging

class DatabaseManager:
    def __init__(self, db_path: str = "mineral_classifier.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mineral_mappings (
                    id INTEGER PRIMARY KEY,
                    variant TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    gbz_name TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    measurement_unit TEXT,
                    measurement_unit_alt TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS synonyms (
                    id INTEGER PRIMARY KEY,
                    word TEXT NOT NULL,
                    base_form TEXT NOT NULL
                )
            """)
            
            # Индексы для быстрого поиска
            conn.execute("CREATE INDEX IF NOT EXISTS idx_variant ON mineral_mappings(variant)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_word ON synonyms(word)")

    def add_mapping(self, variant: str, normalized_name: str, gbz_name: str, 
                   group_name: str, unit: str, unit_alt: str):
        """Добавление нового маппинга"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO mineral_mappings 
                (variant, normalized_name, gbz_name, group_name, measurement_unit, measurement_unit_alt)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (variant.lower(), normalized_name, gbz_name, group_name, unit, unit_alt))

    def add_synonym(self, word: str, base_form: str):
        """Добавление нового синонима"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO synonyms (word, base_form)
                VALUES (?, ?)
            """, (word.lower(), base_form.lower()))

    def get_mapping(self, variant: str) -> Dict:
        """Поиск маппинга по варианту"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT normalized_name, gbz_name, group_name, measurement_unit, measurement_unit_alt
                FROM mineral_mappings
                WHERE variant = ?
            """, (variant.lower(),))
            result = cursor.fetchone()
            
            if result:
                return {
                    'normalized_name_for_display': result[0],
                    'pi_name_gbz_tbz': result[1],
                    'pi_group_is_nedra': result[2],
                    'pi_measurement_unit': result[3],
                    'pi_measurement_unit_alternative': result[4]
                }
            return None

    def get_base_form(self, word: str) -> str:
        """Получение базовой формы слова"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT base_form FROM synonyms WHERE word = ?
            """, (word.lower(),))
            result = cursor.fetchone()
            return result[0] if result else word
