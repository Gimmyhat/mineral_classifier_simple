import pandas as pd
import logging
from typing import List, Dict, Optional
from database import DatabaseManager

class InteractiveLearner:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.known_classifications = self._load_known_classifications()

    def _load_known_classifications(self) -> Dict:
        """Загружает известные классификации из базы данных"""
        try:
            with self.db.SessionLocal() as session:
                entries = session.query(self.db.Dictionary).all()
                classifications = {}
                for entry in entries:
                    classifications[entry.normalized_name] = {
                        'gbz_name': entry.gbz_name,
                        'group_name': entry.group_name,
                        'measurement_unit': entry.measurement_unit,
                        'measurement_unit_alt': entry.measurement_unit_alt
                    }
                return classifications
        except Exception as e:
            logging.error(f"Error loading classifications: {e}")
            return {}

    def add_unclassified_term(self, term: str):
        """Добавляет неклассифицированный термин"""
        self.db.add_unclassified_term(term)

    def get_unclassified_terms(self) -> List[str]:
        """Возвращает список неклассифицированных терминов"""
        return self.db.get_unclassified_terms()

    def get_classification_options(self) -> Dict:
        """Возвращает доступные варианты классификации"""
        return self.db.get_classification_options()

    def add_classification(self, term: str, classification: Dict) -> bool:
        """Добавляет новую классификацию"""
        try:
            logging.debug(f"Adding classification for term: {term} with data: {classification}")
            
            # Создаем уникальный ключ для новой записи
            unique_key = f"{classification['normalized_name']}|{classification['gbz_name']}|{classification['group_name']}"
            
            # Добавляем или обновляем запись в словаре
            dictionary_entry = self.db.add_dictionary_entry({
                'unique_key': unique_key,
                'normalized_name': classification['normalized_name'],
                'gbz_name': classification['gbz_name'],
                'group_name': classification['group_name'],
                'measurement_unit': classification['measurement_unit'],
                'measurement_unit_alt': classification.get('measurement_unit_alt', '')
            })
            
            if dictionary_entry:
                # Добавляем вариации
                term_lower = term.lower()
                variation = self.db.add_variation(term_lower, dictionary_entry.id)
                if not variation:
                    logging.error("Failed to add variation")
                    return False
                
                # Добавляем вариант без контекста в скобках
                if '(' in term_lower:
                    base_term = term_lower.split('(')[0].strip()
                    self.db.add_variation(base_term, dictionary_entry.id)
                
                # Если есть дефис, добавляем вариант с пробелом
                if '-' in term_lower:
                    space_variant = term_lower.replace('-', ' ')
                    self.db.add_variation(space_variant, dictionary_entry.id)
                    
                    # Также для варианта без контекста
                    if '(' in space_variant:
                        base_space_variant = space_variant.split('(')[0].strip()
                        self.db.add_variation(base_space_variant, dictionary_entry.id)
                
                # Обновляем локальный кэш
                self.known_classifications[classification['normalized_name']] = {
                    'gbz_name': classification['gbz_name'],
                    'group_name': classification['group_name'],
                    'measurement_unit': classification['measurement_unit'],
                    'measurement_unit_alt': classification.get('measurement_unit_alt', '')
                }
                
                # Удаляем из неклассифицированных
                self.db.remove_unclassified_term(term)
                
                logging.info(f"Successfully added new classification for term: {term}")
                return True
            
            logging.error("Failed to add dictionary entry")
            return False
            
        except Exception as e:
            logging.error(f"Error adding classification for {term}: {e}")
            return False

    def suggest_classification(self, term: str) -> Optional[Dict]:
        """Предлагает возможную классификацию на основе похожих терминов"""
        try:
            # Простой поиск похожих слов
            term_words = set(term.lower().split())
            
            best_match = None
            best_score = 0
            
            for known_term, classification in self.known_classifications.items():
                known_words = set(known_term.lower().split())
                # Считаем количество общих слов
                common_words = term_words & known_words
                score = len(common_words) / max(len(term_words), len(known_words))
                
                if score > best_score:
                    best_score = score
                    best_match = classification
            
            if best_score > 0.5:  # Порог схожести
                return best_match
                
            return None
            
        except Exception as e:
            logging.error(f"Error suggesting classification for {term}: {e}")
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