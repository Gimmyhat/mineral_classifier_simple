import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Union
from database import DatabaseManager
from text_cleaner import TextCleaner
from morphology_processor import MorphologyProcessor
import pandas as pd
from interactive_learner import InteractiveLearner

class MineralClassifier:
    def __init__(self, file_path):
        try:
            self.db = DatabaseManager()
            # Удалить эти строки:
            # self.dictionary_splitter = DictionarySplitter(file_path)
            
            # Удалить проверку файлов словарей:
            # dict_path = Path('data/dictionary.xlsx')
            # var_path = Path('data/variations.xlsx')
            
            # if not dict_path.exists() or not var_path.exists():
            #     self.dictionary_splitter.process_file()
            #     self.dictionary_splitter.save_files()
            # else:
            #     self.dictionary_splitter.process_file()
            
            # Инициализируем обработчики
            self.text_cleaner = TextCleaner()
            self.morphology_processor = MorphologyProcessor()
            
            # Инициализируем BatchProcessor
            self.batch_processor = BatchProcessor(self)
            
            # Изменить лог:
            logging.debug("MineralClassifier initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing classifier: {str(e)}")
            raise

    def classify_mineral(self, mineral_name: str) -> Dict[str, Any]:
        """Классифицирует минерал"""
        try:
            logging.debug(f"Starting classification for: {mineral_name}")
            
            # 1. Очищаем текст
            cleaned_text = self.text_cleaner.clean_text(mineral_name)
            
            # 2. Извлекаем основной термин и контексты
            main_term, contexts = self.text_cleaner.extract_context(cleaned_text)
            
            # 3. Пробуем точное соответствие
            direct_result = self.db.get_mapping(cleaned_text)
            if direct_result:
                logging.debug(f"Found direct match for: {cleaned_text}")
                return direct_result
            
            # 4. Генерируем варианты для поиска
            search_variants = self._generate_search_variants(main_term, contexts)
            
            # 5. Проверяем варианты
            for variant in search_variants:
                result = self.db.get_mapping(variant)
                if result:
                    logging.debug(f"Found match for variant: {variant}")
                    return result
            
            # 6. Если не нашли, пробуем морфологический анализ
            normalized_variants = self._generate_morphological_variants(main_term)
            for variant in normalized_variants:
                result = self.db.get_mapping(variant)
                if result:
                    logging.debug(f"Found match for normalized variant: {variant}")
                    return result
            
            # 7. Если не удалось классифицировать, добавляем в неклассифицированные
            logging.debug(f"No matches found for: {mineral_name}")
            try:
                self.db.add_unclassified_term(mineral_name)
                logging.debug(f"Added '{mineral_name}' to unclassified terms")
            except Exception as e:
                logging.error(f"Error adding to unclassified terms: {e}")
                
            return self._get_unknown_result()
            
        except Exception as e:
            logging.error(f"Error in classify_mineral: {str(e)}")
            return self._get_unknown_result()

    def _generate_search_variants(self, main_term: str, contexts: List[str]) -> List[str]:
        """Генерирует варианты для поиска"""
        variants = []
        
        # Добавляем основной термин
        variants.append(main_term)
        
        # Добавляем варианты с каждым контекстом
        for context in contexts:
            variants.append(f"{main_term} ({context})")
            variants.append(f"{main_term} {context}")
        
        # Если есть несколько контекстов, добавляем полный вариант
        if len(contexts) > 1:
            full_context = ', '.join(contexts)
            variants.append(f"{main_term} ({full_context})")
        
        # Обрабатываем составные темины
        components = self.morphology_processor.split_compound_word(main_term)
        variants.extend(components)
        
        # Удаляем дубликаты с сохранением порядка
        return self.text_cleaner.remove_duplicates(variants)

    def _generate_morphological_variants(self, term: str) -> List[str]:
        """Генерирует морфологические варианты"""
        variants = []
        
        # Получаем все формы слова
        word_forms = self.morphology_processor.get_word_forms(term)
        variants.extend(word_forms)
        
        # Для составных терминов обрабатываем каждую часть
        if ' ' in term or '-' in term:
            components = self.morphology_processor.split_compound_word(term)
            for component in components:
                if self.morphology_processor.is_valid_word(component):
                    variants.extend(self.morphology_processor.get_word_forms(component))
        
        return self.text_cleaner.remove_duplicates(variants)

    def _get_unknown_result(self) -> Dict[str, str]:
        """Возвращает стандартный результат для неизвестного значения"""
        return {
            'normalized_name_for_display': 'неизвестно',
            'pi_name_gbz_tbz': 'неизвестно',
            'pi_group_is_nedra': 'неизвестно',
            'pi_measurement_unit': '',
            'pi_measurement_unit_alternative': ''
        }

class BatchProcessor:
    def __init__(self, classifier: MineralClassifier):
        self.classifier = classifier
        self.learner = InteractiveLearner(classifier.db)

    def _process_mineral(self, mineral: str) -> dict:
        """Обработка одного минерала"""
        if pd.isna(mineral):
            return None
        result = self.classifier.classify_mineral(str(mineral))
        result['original_name'] = mineral
        return result

    def process_excel(self, input_file: str, output_file: str = None) -> str:
        """Обрабатывает входной Excel файл."""
        try:
            logging.debug(f"Starting processing of file: {input_file}")
            df_input = pd.read_excel(input_file, usecols=[0])
            
            if df_input.empty:
                raise ValueError("Input file is empty")

            minerals = df_input.iloc[:, 0].tolist()
            total_records = len(minerals)
            logging.debug(f"Total minerals to process: {total_records}")

            # Обрабатываем минералы
            all_results = []
            from main import processing_progress
            
            for i, mineral in enumerate(minerals, 1):
                result = self._process_mineral(mineral)
                if result:
                    all_results.append(result)

                # Обновляем прогресс
                if hasattr(self, 'process_id'):
                    current_progress = int((i / total_records) * 100)
                    processing_progress[self.process_id] = {
                        'current': i,
                        'total': total_records,
                        'progress': current_progress,
                        'status': f'Обработано {i} из {total_records} записей'
                    }
                    logging.debug(f"Progress updated: {current_progress}%")

                if i % 10 == 0:  # Логируем каждые 10 минералов
                    logging.debug(f"Processed {i}/{total_records} minerals")

            # Создаем DataFrame с результатами
            df_results = pd.DataFrame(all_results)
            
            columns_order = [
                'original_name',
                'normalized_name_for_display',
                'pi_name_gbz_tbz',
                'pi_group_is_nedra',
                'pi_measurement_unit',
                'pi_measurement_unit_alternative'
            ]
            df_results = df_results[columns_order]

            if output_file is None:
                base_name = input_file.rsplit('.', 1)[0]
                output_file = f"{base_name}_classified.xlsx"

            logging.debug(f"Saving results to: {output_file}")
            df_results.to_excel(output_file, index=False)
            logging.debug("File processing completed")

            # Добавляем неклассифицированные термины в learner
            for result in all_results:
                if result['normalized_name_for_display'] == 'неизвестно':
                    self.learner.add_unclassified_term(result['original_name'])

            return output_file

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            if hasattr(self, 'process_id'):
                processing_progress[self.process_id]['status'] = f'Ошибка: {str(e)}'
            raise

    def get_unclassified_terms(self) -> List[str]:
        """Возвращает список неклассифицированных терминов"""
        return self.learner.get_unclassified_terms()

    def get_classification_options(self) -> Dict:
        """Возвращает доступные варианты классификации"""
        return self.learner.get_classification_options()

    def add_classification(self, term: str, classification: Dict) -> bool:
        """Добавляет новую классификацию"""
        return self.learner.add_classification(term, classification)
