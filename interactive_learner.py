import pandas as pd
import logging
from typing import List, Dict, Optional
from database import DatabaseManager

class InteractiveLearner:
    def __init__(self, db):
        self.db = db

    def get_classification_options(self) -> Dict:
        """Получение доступных вариантов классификации"""
        return self.db.get_classification_options()

    def add_classification(self, term: str, classification: Dict) -> bool:
        """Добавляет новую классификацию"""
        try:
            # Создаем уникальный ключ
            unique_key = f"{classification['normalized_name']}|{classification['gbz_name']}|{classification['group_name']}"
            
            # Добавляем запись в словарь
            dictionary_entry = self.db.add_dictionary_entry({
                'unique_key': unique_key,
                'normalized_name': classification['normalized_name'],
                'gbz_name': classification['gbz_name'],
                'group_name': classification['group_name'],
                'measurement_unit': classification['measurement_unit'],
                'measurement_unit_alt': classification.get('measurement_unit_alt', '')
            })
            
            if dictionary_entry:
                # Добавляем вариацию
                self.db.add_variation(term, dictionary_entry.id)
                # Удаляем термин из неклассифицированных
                self.db.remove_unclassified_term(term)
                return True
            return False
        except Exception as e:
            logging.error(f"Error adding classification: {e}")
            return False

    def get_unclassified_terms(self) -> List[str]:
        """Получение списка неклассифицированных терминов"""
        return self.db.get_unclassified_terms()

    def add_unclassified_term(self, term: str) -> bool:
        """Добавление термина в список неклассифицированных"""
        return self.db.add_unclassified_term(term)

    def suggest_classification(self, term: str) -> Optional[Dict]:
        """Предлагает классификацию для термина"""
        try:
            with self.db.SessionLocal() as session:
                # Ищем похожие записи в словаре
                similar_entries = session.query(self.db.Dictionary)\
                    .join(self.db.NormalizedName)\
                    .filter(self.db.NormalizedName.name.ilike(f"%{term}%"))\
                    .all()

                if similar_entries:
                    # Берем первую похожую запись
                    entry = similar_entries[0]
                    return {
                        'normalized_name': entry.normalized_name_ref.name,
                        'gbz_name': entry.gbz_name_ref.name,
                        'group_name': entry.group_name_ref.name,
                        'measurement_unit': entry.measurement_unit_ref.name,
                        'measurement_unit_alt': entry.measurement_unit_alt_ref.name if entry.measurement_unit_alt_ref else ''
                    }
                return None
        except Exception as e:
            logging.error(f"Error suggesting classification: {e}")
            return None

    def export_unclassified(self, filename: str):
        """Экспортирует неклассифицированные термины в Excel"""
        try:
            terms = self.get_unclassified_terms()
            df = pd.DataFrame({
                'term': terms,
                'suggested_classification': [
                    self.suggest_classification(term) for term in terms
                ]
            })
            df.to_excel(filename, index=False)
            logging.info(f"Exported {len(terms)} unclassified terms to {filename}")
        except Exception as e:
            logging.error(f"Error exporting unclassified terms: {e}")