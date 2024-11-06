import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Union
from database import DatabaseManager
from text_cleaner import TextCleaner
from morphology_processor import MorphologyProcessor
import pandas as pd
from interactive_learner import InteractiveLearner
from rapidfuzz import fuzz, process

class MineralClassifier:
    def __init__(self, file_path):
        try:
            self.db = DatabaseManager()
            self.text_cleaner = TextCleaner()
            self.morphology_processor = MorphologyProcessor()
            self.batch_processor = BatchProcessor(self)
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
            
            # 2. Извлекаем контекст из полного термина
            main_text = cleaned_text
            context_part = ""
            if '(' in cleaned_text and ')' in cleaned_text:
                context_start = cleaned_text.find('(')
                context_part = cleaned_text[context_start:]
                main_text = cleaned_text[:context_start].strip()
                logging.debug(f"Extracted context: {context_part} from main text: {main_text}")
            
            # 3. Проверяем разделители в основном тексте
            if "," in main_text or " и " in main_text:
                terms = []
                # Сначала разбиваем по запятой
                parts = [part.strip() for part in main_text.split(',')]
                
                # Обрабатываем каждую часть на наличие союза "и"
                for part in parts:
                    if " и " in part:
                        terms.extend([t.strip() for t in part.split(" и ")])
                    else:
                        terms.append(part)
                
                logging.debug(f"Split terms: {terms}")
                
                results = []
                # Обрабатываем каждый термин с контекстом
                for term in terms:
                    # Добавляем контекст к термину
                    full_term = f"{term} {context_part}".strip()
                    logging.debug(f"Processing split term with context: {full_term}")
                    
                    # Пробуем прямое соответствие с контекстом
                    direct_result = self.db.get_mapping(full_term)
                    if direct_result:
                        logging.debug(f"Found direct match for term with context: {full_term}")
                        results.append(direct_result)
                        continue
                    
                    # Пробуем без контекста
                    direct_result = self.db.get_mapping(term)
                    if direct_result:
                        logging.debug(f"Found direct match for term: {term}")
                        results.append(direct_result)
                        continue
                    
                    # Если прямого соответствия нет, пробуем классифицировать термин отдельно
                    term_result = self._classify_single_term(term, context_part)
                    if term_result and term_result['normalized_name_for_display'] != 'неизвестно':
                        logging.debug(f"Found classification for term: {term}")
                        results.append(term_result)
                
                # Если нашли хотя бы один результат, возвращаем первый
                if results:
                    logging.debug(f"Returning first result from {len(results)} results")
                    return results[0]
                
                # Если не нашли результатов для отдельных терминов,
                # пробуем обработать весь текст как единый термин
                logging.debug(f"Trying to process as single term: {main_text}")
                return self._classify_single_term(main_text, context_part)
            
            # 4. Если нет разделителей, обрабатываем как единый термин
            return self._classify_single_term(main_text, context_part)
            
        except Exception as e:
            logging.error(f"Error in classify_mineral: {str(e)}")
            return self._get_unknown_result()

    def _classify_single_term(self, term: str, context: str = "") -> Dict[str, Any]:
        """Классифицирует отдельный термин"""
        try:
            # Нормализуем написание термина
            normalized_term = self.morphology_processor.normalize_spelling(term)
            
            # Сначала проверяем полный термин с контекстом
            if context:
                full_term = f"{normalized_term} {context}".strip()
                direct_result = self.db.get_mapping(full_term)
                if direct_result:
                    logging.debug(f"Found direct match for full term with context: {full_term}")
                    return direct_result
            
            # Проверяем нормализованный термин
            direct_result = self.db.get_mapping(normalized_term)
            if direct_result:
                logging.debug(f"Found direct match for normalized term: {normalized_term}")
                return direct_result
                
            # Обработка составных терминов
            compound_terms = self._split_compound_terms(term)
            if compound_terms:
                logging.debug(f"Generated compound terms: {compound_terms}")
                for sub_term in compound_terms:
                    # Проверяем каждый подтермин с контекстом
                    if context:
                        full_sub_term = f"{sub_term} {context}".strip()
                        result = self.db.get_mapping(full_sub_term)
                        if result:
                            logging.debug(f"Found match for compound term with context: {full_sub_term}")
                            return result
                    
                    # Проверяем просто подтермин
                    result = self.db.get_mapping(sub_term)
                    if result:
                        logging.debug(f"Found match for compound term: {sub_term}")
                        return result
            
            # Получаем все морфологические формы
            normalized_variants = self._generate_morphological_variants(term)
            logging.debug(f"Generated morphological variants: {normalized_variants}")
            
            for variant in normalized_variants:
                result = self.db.get_mapping(variant)
                if result:
                    logging.debug(f"Found match for morphological variant: {variant}")
                    return result
            
            # Пробуем нечеткий поиск только если морфологический анализ не помог
            try:
                fuzzy_result = self._fuzzy_search(term)
                if fuzzy_result:
                    logging.debug(f"Found fuzzy match for term: {term}")
                    return fuzzy_result
            except Exception as e:
                logging.error(f"Error in fuzzy search: {str(e)}")
            
            logging.debug(f"No matches found for term: {term}")
            return self._get_unknown_result()
            
        except Exception as e:
            logging.error(f"Error in _classify_single_term: {str(e)}")
            return self._get_unknown_result()

    def _split_compound_terms(self, term: str) -> List[str]:
        """Разбивает составной термин на компоненты с учетом множественных комбинаций"""
        results = set()
        
        # Разбиваем по всем возможным разделителям
        separators = ['-', ' ']
        parts = term
        for sep in separators:
            parts = ' '.join(parts.split(sep))
        parts = [p.strip() for p in parts.split()]
        
        # Добавляем исходный термин
        results.add(term)
        
        # Добавляем отдельные компоненты
        results.update(parts)
        
        # Генерируем все возможные последовательные комбинации
        for i in range(len(parts)):
            for j in range(i + 2, len(parts) + 1):
                # Комбинация через дефис
                hyphen_combo = '-'.join(parts[i:j])
                results.add(hyphen_combo)
                
                # Комбинация через пробел
                space_combo = ' '.join(parts[i:j])
                results.add(space_combo)
        
        # Генерируем специальные комбинации для материалов
        if 'материал' in parts:
            material_index = parts.index('материал')
            prefix_parts = parts[:material_index]
            
            # Создаем комбинации для префиксной части
            for i in range(len(prefix_parts)):
                for j in range(i + 1, len(prefix_parts) + 1):
                    current_parts = prefix_parts[i:j]
                    # Добавляем варианты с "материал"
                    hyphen_combo = f"{'-'.join(current_parts)}-материал"
                    space_combo = f"{' '.join(current_parts)} материал"
                    results.add(hyphen_combo)
                    results.add(space_combo)
        
        # Добавляем специальные правила для известных шаблонов
        known_patterns = {
            'песчано': ['песок'],
            'гравийно': ['гравий'],
            'валунный': ['валуны'],
            'щебеночный': ['щебень']
        }
        
        for part in parts:
            if part in known_patterns:
                results.update(known_patterns[part])
        
        return list(results)

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
        """Взвращает стандартный результат для неизвестного значения"""
        return {
            'normalized_name_for_display': 'неизвестно',
            'pi_name_gbz_tbz': 'неизвестно',
            'pi_group_is_nedra': 'неизвестно',
            'pi_measurement_unit': '',
            'pi_measurement_unit_alternative': ''
        }

    def _fuzzy_search(self, term: str, min_ratio: int = 85) -> Dict[str, Any]:
        """Выполняет нечеткий поиск по базе данных"""
        try:
            all_terms = self.db.get_all_terms()
            if not all_terms:
                return None
                
            # process.extractOne возвращает кортеж (match, score, index)
            result = process.extractOne(
                term.lower(),
                all_terms,
                scorer=fuzz.ratio,
                score_cutoff=min_ratio
            )
            
            if result:
                best_match = result[0]  # Берем только совпадение, игнорируем score и index
                logging.debug(f"Fuzzy match found: {best_match} for term: {term}")
                return self.db.get_mapping(best_match)
            return None
        except Exception as e:
            logging.error(f"Error in fuzzy search: {str(e)}")
            return None

    def _process_context(self, context: str) -> Set[str]:
        """Обрабатывает контекстную информацию"""
        contexts = set()
        
        # Удаляем скобки и разделяем по запятой
        clean_context = context.replace('(', '').replace(')', '')
        parts = [p.strip() for p in clean_context.split(',')]
        
        for part in parts:
            # Добавляем оригинальный контекст
            contexts.add(part)
            
            # Добавляем нормализованные варианты
            normalized = self.morphology_processor.normalize(part)
            if normalized:
                contexts.add(normalized)
                
            # Обрабатываем составные контексты
            if ' ' in part:
                sub_parts = part.split()
                for sub in sub_parts:
                    normalized_sub = self.morphology_processor.normalize(sub)
                    if normalized_sub:
                        contexts.add(normalized_sub)
        
        return contexts

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
