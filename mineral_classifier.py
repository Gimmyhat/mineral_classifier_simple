import pandas as pd
import logging
from typing import Dict, Any, List
import redis
from redis_manager import RedisManager
from pathlib import Path
from split_dictionary import DictionarySplitter
import re

logging.basicConfig(level=logging.DEBUG)

class MineralClassifier:
    def __init__(self, file_path):
        try:
            self.db = RedisManager()

            # Создаем экземпляр DictionarySplitter
            self.dictionary_splitter = DictionarySplitter(file_path)

            # Проверяем наличие файлов словарей
            dict_path = Path('data/dictionary.xlsx')
            var_path = Path('data/variations.xlsx')

            if not dict_path.exists() or not var_path.exists():
                # Если файлов нет, создаем их
                self.dictionary_splitter.process_file()
                self.dictionary_splitter.save_files()
            else:
                # Даже если файлы существуют, нам нужно заполнить known_minerals
                self.dictionary_splitter.process_file()

            # Загружаем словари
            self.db.load_dictionaries(
                dictionary_path='data/dictionary.xlsx',
                variations_path='data/variations.xlsx'
            )

            logging.debug(f"Initialized with {len(self.dictionary_splitter.known_minerals)} known minerals")

        except Exception as e:
            logging.error(f"Error initializing classifier: {str(e)}")
            raise

    def classify_mineral(self, mineral_name: str) -> Dict[str, Any]:
        """Классифицирует минерал по имени"""
        mineral_name = mineral_name.lower().strip()

        try:
            # Сначала проверяем точное соответствие в Redis
            exact_match = self.db.get_exact_mapping(mineral_name)
            if exact_match:
                logging.debug(f"Found exact match in Redis for {mineral_name}: {exact_match}")
                return exact_match

            # Разбиваем на компоненты и анализируем каждый
            components = self._split_input(mineral_name)
            logging.debug(f"Split into components: {components}")

            # Создаем словарь для хранения результатов с их приоритетами
            results_with_priority = {}

            for component in components:
                # Проверяем точное соответствие компонента
                result = self.db.get_exact_mapping(component)
                if result:
                    priority = self._calculate_match_priority(component, result, is_exact=True)
                    results_with_priority[priority] = result
                    continue

                # Проверяем частичные соответствия
                partial_matches = self._find_partial_matches(component)
                for match, match_result in partial_matches.items():
                    priority = self._calculate_match_priority(component, match_result, is_exact=False)
                    results_with_priority[priority] = match_result

            # Если нашли результаты, возвращаем с наивысшим приоритетом
            if results_with_priority:
                best_priority = max(results_with_priority.keys())
                logging.debug(f"Found best match with priority {best_priority}")
                return results_with_priority[best_priority]

            # Если ничего не нашли
            unknown_result = {
                'normalized_name_for_display': 'неизвестно',
                'pi_name_gbz_tbz': 'неизвестно',
                'pi_group_is_nedra': 'неизвестно',
                'pi_measurement_unit': '',
                'pi_measurement_unit_alternative': ''
            }
            logging.debug(f"No result found, returning: {unknown_result}")
            return unknown_result

        except Exception as e:
            logging.error(f"Error in classify_mineral: {str(e)}")
            raise

    def _calculate_match_priority(self, component: str, result: Dict[str, Any], is_exact: bool) -> float:
        """Вычисляет приоритет соответствия"""
        priority = 0.0

        # Базовый приоритет для точных совпадений
        if is_exact:
            priority += 10.0

        # Приоритет за ключевые слова
        keyword_weights = {
            'песчано-гравийный': 6.0,  # Повышенный вес для составных терминов
            'песчано': 5.0,
            'гравийный': 5.0,
            'материал': 4.0,
            'песок': 5.0,
            'гравий': 5.0,
            'керамический': 4.0,
            'керамика': 4.0,
            'глина': 4.0,
            'щебень': 3.0,
            'супесь': 3.0,
            'строительный': 2.0,
            'природный': 1.5,
            'отсев': 2.0
        }

        for keyword, weight in keyword_weights.items():
            if keyword in component:
                priority += weight

        # Приоритет за длину компонента (более длинные совпадения обычно более точные)
        priority += len(component) * 0.1

        # Приоритет за наличие уточнений в скобках
        if '(' in component and ')' in component:
            priority += 2.0

        # Приоритет за соответствие категории
        if 'керамический' in component or 'керамика' in component:
            if 'керамическое сырье' in result['pi_name_gbz_tbz'].lower():
                priority += 5.0

        return priority

    def _find_partial_matches(self, component: str) -> Dict[str, Dict[str, Any]]:
        """Ищет частичные соответствия для компонента"""
        matches = {}

        # Проверяем базовые слова
        base_words = ['песок', 'глина', 'щебень', 'супесь']
        for word in base_words:
            if word in component:
                result = self.db.get_mapping(word)
                if result:
                    matches[word] = result

        # Проверяем комбинации с уточнениями
        if 'керамический' in component or 'керамика' in component:
            for word in base_words:
                if word in component:
                    ceramic_key = f"{word} керамический"
                    result = self.db.get_mapping(ceramic_key)
                    if result:
                        matches[ceramic_key] = result

        return matches

    def _split_input(self, text: str) -> List[str]:
        """Разбиение входного текста на компоненты с сохранением контекста"""
        components = []
        text = text.lower().strip()

        # Добавляем полный текст
        components.append(text)

        # Нормализуем текст, заменяя последовательности пробелов на один пробел
        text = ' '.join(text.split())

        # Заменяем запятую с пробелом на запятую для унификации
        text = text.replace(' , ', ',').replace(', ', ',').replace(' ,', ',')

        # Извлекаем основной термин (до скобок) и разбиваем по запятым
        main_part = text.split('(')[0].strip()
        main_terms = [term.strip() for term in main_part.split(',')]

        # Добавляем основные термины
        for term in main_terms:
            if term and term not in components:
                components.append(term)

        # Извлекаем термины в скобках
        bracket_terms = re.findall(r'\((.*?)\)', text)
        for term in bracket_terms:
            term = term.strip()
            if term:
                components.append(term)
                # Разбиваем по запятым внутри скобок
                for subterm in term.split(','):
                    subterm = subterm.strip()
                    if subterm and subterm not in components:
                        components.append(subterm)

        # Дополнительно разбиваем составные термины
        compound_terms = []
        for component in components[:]:  # Используем копию списка для итерации
            if ' ' in component:
                parts = component.split()
                for part in parts:
                    if len(part) > 2 and part not in components:  # Игнорируем короткие части
                        compound_terms.append(part)

        # Добавляем составные части в конец списка
        components.extend(compound_terms)

        logging.debug(f"Split '{text}' into components: {components}")
        return components

    def load_data(self, file_path):
        """Загрузка данных из Excel в Redis"""
        try:
            df = pd.read_excel(
                file_path,
                header=3,
                usecols=[2,3,4,5,6,7]
            )

            df.columns = [
                "pi_variants", "normalized_name_for_display", "pi_name_gbz_tbz",
                "pi_group_is_nedra", "pi_measurement_unit", "pi_measurement_unit_alternative"
            ]

            # Заполняем пустые значения
            df = df.fillna('')

            # Подготавливаем данные для Redis
            mappings = []
            for _, row in df.iterrows():
                mapping = {
                    'variant': str(row['pi_variants']),
                    'normalized_name': str(row['normalized_name_for_display']),
                    'gbz_name': str(row['pi_name_gbz_tbz']),
                    'group_name': str(row['pi_group_is_nedra']),
                    'measurement_unit': str(row['pi_measurement_unit']),
                    'measurement_unit_alt': str(row['pi_measurement_unit_alternative'])
                }
                mappings.append(mapping)

            # Массовое добавление данных
            self.db.bulk_add_mappings(mappings)
            logging.debug(f"Loaded {len(mappings)} mappings into Redis")

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def analyze_unknown_term(self, term: str) -> Dict[str, Any]:
        """Анализ неизвестного термина"""
        analysis = self.dictionary_splitter._analyze_term(term)

        if analysis['base_mineral']:
            # Ище похожие термины в известных связях
            if analysis['base_mineral'] in self.dictionary_splitter.term_relationships:
                relationships = self.dictionary_splitter.term_relationships[analysis['base_mineral']]

                # Находим наиболе подходящий контекст
                best_match = max(relationships,
                               key=lambda x: len(set(x['context']) & set(analysis['context'])))

                if best_match['confidence'] > 0.5:
                    return self.db.get_mapping(best_match['normalized_name'])

        return {
            'normalized_name_for_display': 'неизвестно',
            'pi_name_gbz_tbz': 'неизвестно',
            'pi_group_is_nedra': 'неизвестно',
            'pi_measurement_unit': '',
            'pi_measurement_unit_alternative': ''
        }

    def _analyze_chemical_formula(self, text: str) -> Dict[str, Any]:
        """Анализирует химические формулы в тексте"""
        formula_pattern = r'[A-Z][a-z]?\d*'
        formulas = re.findall(formula_pattern, text)

        if formulas:
            # Проверяем наличие формул в базе данных
            for formula in formulas:
                result = self.db.get_mapping(formula)
                if result:
                    return result

        return None

class BatchProcessor:
    def __init__(self, classifier: MineralClassifier):
        self.classifier = classifier

    def _process_mineral(self, mineral: str) -> dict:
        """Обработка одного минерала"""
        if pd.isna(mineral):
            return None
        result = self.classifier.classify_mineral(str(mineral))
        result['original_name'] = mineral
        return result

    def process_excel(self, input_file: str, output_file: str = None) -> str:
        """
        Обрабатывает входной Excel файл.
        """
        try:
            logging.debug(f"Starting processing of file: {input_file}")
            df_input = pd.read_excel(
                input_file,
                usecols=[0],
                engine='openpyxl'
            )

            if df_input.empty:
                raise ValueError("Input file is empty")

            minerals = df_input.iloc[:, 0].tolist()
            total_minerals = len(minerals)
            logging.debug(f"Total minerals to process: {total_minerals}")

            # Обрабатываем минералы
            all_results = []
            for i, mineral in enumerate(minerals, 1):
                result = self._process_mineral(mineral)
                if result:
                    all_results.append(result)
                if i % 10 == 0:  # Логируем каждые 10 минералов
                    logging.debug(f"Processed {i}/{total_minerals} minerals")

            logging.debug("Creating DataFrame with results")
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

            return output_file

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            raise
