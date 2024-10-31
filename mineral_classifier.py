import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Set, Union
import redis
from redis_manager import RedisManager
from pathlib import Path
from split_dictionary import DictionarySplitter
import re
import pymorphy3
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.DEBUG)

class MineralClassifier:
    def __init__(self, file_path):
        try:
            self.db = RedisManager()
            self.morph = pymorphy3.MorphAnalyzer()  # Инициализируем морфологический анализатор

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
                # Даже если файлы сществуют, нам нужно заполнить known_minerals
                self.dictionary_splitter.process_file()

            # Загружаем словари
            self.db.load_dictionaries(
                dictionary_path='data/dictionary.xlsx',
                variations_path='data/variations.xlsx'
            )

            # Инициализируем улучшенный классификатор
            self.enhanced_classifier = EnhancedMineralClassifier(
                known_minerals=set(self.dictionary_splitter.known_minerals)
            )

            logging.debug(f"Initialized with {len(self.dictionary_splitter.known_minerals)} known minerals")

        except Exception as e:
            logging.error(f"Error initializing classifier: {str(e)}")
            raise

    def classify_mineral(self, mineral_name: str, return_multiple: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Классифицирует минерал с возможностью возврата нескольких вариантов"""
        try:
            # Очищаем входной текст от лишних пробелов
            mineral_name = ' '.join(mineral_name.split())
            
            # 1. Пробуем точное соответствие
            direct_result = self.db.get_exact_mapping(mineral_name)
            if direct_result:
                logging.debug(f"Found direct match for '{mineral_name}'")
                return direct_result

            # 2. Формируем варианты поиска от более специфичного к более общему
            search_variants = []
            
            # Основной термин и контекст
            main_term = mineral_name
            context = ""
            if '(' in mineral_name and ')' in mineral_name:
                main_term = mineral_name.split('(')[0].strip()
                context = mineral_name.split('(')[1].split(')')[0].strip()
                # Добавляем полный термин с контекстом
                search_variants.append(f"{main_term} ({context})")
            
            # Добавляем основной термин
            search_variants.append(main_term)
            
            # Разбиваем на слова
            words = main_term.split()
            stop_words = {'для', 'на', 'в', 'из', 'с', 'и', 'при', 'под', 'рудный', 'открытый', 'закрытый'}
            
            # Добавляем варианты без стоп-слов
            meaningful_words = [w for w in words if w.lower() not in stop_words]
            if len(meaningful_words) > 1:
                search_variants.append(' '.join(meaningful_words))
            
            # Добавляем каждое значимое слово отдельно
            search_variants.extend(meaningful_words)
            
            # 3. Проверяем каждый вариант
            logging.debug(f"Checking variants: {search_variants}")
            for variant in search_variants:
                # Очищаем вариант от знаков препинания
                cleaned_variant = re.sub(r'[.,!?:;()[\]{}«»""\'`]', '', variant)
                cleaned_variant = re.sub(r'\s+', ' ', cleaned_variant).strip()
                
                # Пробуем точное соответствие
                variant_result = self.db.get_exact_mapping(cleaned_variant)
                if variant_result:
                    logging.debug(f"Found match for variant '{cleaned_variant}'")
                    return variant_result
                
                # Пробуем нормализованную форму
                normalized = self._normalize_word(cleaned_variant)
                if normalized != cleaned_variant:
                    norm_result = self.db.get_exact_mapping(normalized)
                    if norm_result:
                        logging.debug(f"Found match for normalized variant '{normalized}'")
                        return norm_result
                
                # Если термин составной (с дефисом), проверяем варианты написания
                if '-' in cleaned_variant:
                    space_variant = cleaned_variant.replace('-', ' ')
                    space_result = self.db.get_exact_mapping(space_variant)
                    if space_result:
                        logging.debug(f"Found match for space variant '{space_variant}'")
                        return space_result

            # 4. Если ничего не нашли, возвращаем неизвестно
            logging.debug(f"No match found for '{mineral_name}' and its variants")
            return self._get_unknown_result()
                
        except Exception as e:
            logging.error(f"Error in classify_mineral: {str(e)}")
            return self._get_unknown_result()

    def _calculate_match_priority(self, component: str, result: Dict[str, Any], is_exact: bool) -> float:
        """Вычисляет приоритет соответствия"""
        priority = 0.0

        # Базовый приоритет для точных совпадений
        if is_exact:
            priority += 10.0

        # Расширенный словарь ключевых слов и их весов
        keyword_weights = {
            # Составные термины
            'песчано-гравийный': 8.0,
            'песчано-гравийно-валунный': 8.0,
            'песчано-гравийно-галечный': 8.0,
            'песчано-валунный': 7.0,
            'песчано-глинистый': 7.0,
            'дресвяно-щебнистый': 7.0,

            # Базовые материалы
            'песчаник': 6.0,
            'песок': 6.0,
            'гравий': 6.0,
            'щебень': 6.0,
            'глина': 6.0,
            'валун': 6.0,
            'галька': 6.0,
            'алевропесчаник': 6.0,
            'доломит': 6.0,

            # Характеристики
            'строительный': 4.0,
            'полимиктовый': 4.0,
            'пемзовый': 4.0,

            # Типы материалов
            'материал': 3.0,
            'отложения': 3.0,
            'породы': 3.0,

            # Контекстные маркеры
            'балластное сырье': 5.0,
            'строительные камни': 5.0,
            'стительные материалы': 5.0,
            'наполнители бетона': 5.0
        }

        # Проверяем контекст в скобках
        context_weights = {
            'балластное сырье': 3.0,
            'строительные камни': 3.0,
            'строительные материалы': 3.0,
            'наполнители бетона': 3.0
        }

        # Проверяем наличие контекста в скобках
        if '(' in component and ')' in component:
            context = component.split('(')[1].split(')')[0].lower().strip()
            for ctx, weight in context_weights.items():
                if ctx in context:
                    priority += weight

        # Проверяем варианты написания с дефисом и пробелом
        component_variants = [
            component,
            component.replace('-', ' '),
            *[part.strip() for part in component.split('-')],
            *[part.strip() for part in component.split(',')]
        ]

        # Проверяем все варианты на наличие ключевых слов
        for variant in component_variants:
            variant = variant.lower()
            for keyword, weight in keyword_weights.items():
                if keyword in variant:
                    priority += weight

        # Приоритет за длину компонента (для более специфичных терминов)
        priority += len(component) * 0.1

        # Дополнительный приоритет за составные термины
        if '-' in component:
            priority += 2.0
        if ',' in component:
            priority += 1.5

        logging.debug(f"Priority calculation for '{component}': {priority}")
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

        # Извлекаем контекст из скобок
        context = ""
        if '(' in text and ')' in text:
            context = text.split('(')[1].split(')')[0].strip()
            # Добавляем основной текст с контекстом
            main_with_context = f"{text.split('(')[0].strip()} ({context})"
            if main_with_context not in components:
                components.append(main_with_context)

        # Обработка основного термина
        main_part = text.split('(')[0].strip()

        # Разбиваем по запятым
        for part in main_part.split(','):
            part = part.strip()
            if part:
                # Добавляем часть
                components.append(part)
                # Добавляем часть с контекстом
                if context:
                    components.append(f"{part} ({context})")

                # Обработка дефисов
                if '-' in part:
                    # Вариант с пробелом
                    space_variant = part.replace('-', ' ')
                    components.append(space_variant)

                    # Отдельные части
                    subparts = [p.strip() for p in part.split('-')]
                    normalized_subparts = [self._normalize_word(p) for p in subparts]

                    # Добавляем нормализованные части
                    components.extend(normalized_subparts)

                    # Добавляем нормализованный составной термин
                    normalized_compound = '-'.join(normalized_subparts)
                    if normalized_compound != part:
                        components.append(normalized_compound)

        # Удаляем дубликаты и пустые строки, сохраняя порядок
        seen = set()
        unique_components = []
        for comp in components:
            if comp and comp not in seen and len(comp) > 1:
                seen.add(comp)
                unique_components.append(comp)
                # Добавляем нормализованную версию
                normalized = ' '.join(self._normalize_word(word) for word in comp.split())
                if normalized != comp and normalized not in seen:
                    seen.add(normalized)
                    unique_components.append(normalized)

        logging.debug(f"Split '{text}' into components: {unique_components}")
        return unique_components

    def _normalize_word(self, word: str) -> str:
        """Приводит слово к нормальной форме"""
        try:
            # Получаем все возможные разборы слова
            parses = self.morph.parse(word.lower())
            if not parses:
                return word.lower()

            # Ищем существительное среди разборов
            noun_parses = [p for p in parses if 'NOUN' in p.tag]
            if noun_parses:
                normalized = noun_parses[0].normal_form
                logging.debug(f"Normalized '{word}' to '{normalized}' (noun)")
                return normalized

            # Если существительное не найдено, берем первый разбор
            normalized = parses[0].normal_form
            logging.debug(f"Normalized '{word}' to '{normalized}' (other)")
            return normalized

        except Exception as e:
            logging.error(f"Error normalizing word '{word}': {e}")
            return word.lower()

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

            # Подготавлваем данные для Redis
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
            # Ищем похожие термины в известных связях
            if analysis['base_mineral'] in self.dictionary_splitter.term_relationships:
                relationships = self.dictionary_splitter.term_relationships[analysis['base_mineral']]

                # Находим наиболее подходящий контекст
                best_match = max(relationships,
                               key=lambda x: len(set(x['context']) & set(analysis['context'])))

                if best_match['confidence'] > 0.5:
                    return self.db.get_mapping(best_match['normalized_name'])

        return self._get_unknown_result()

    def _analyze_chemical_formula(self, text: str) -> Dict[str, Any]:
        """Анализирует химические формулы в тексте"""
        formula_pattern = r'[A-Z][a-z]?\d*'
        formulas = re.findall(formula_pattern, text)

        if formulas:
            # Проверяем наличие формул в базе даннх
            for formula in formulas:
                result = self.db.get_mapping(formula)
                if result:
                    return result

        return None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Вычисляет расстояние Левенштейна между двумя строками"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _find_similar_word(self, word: str, known_words: set, threshold: float = 0.65) -> str:
        """Находит похожее слово среди известных слов"""
        word = word.lower()
        best_match = None
        best_ratio = 0

        # Очищаем входное слово от повторяющихся букв
        word_cleaned = re.sub(r'(.)\1{2,}', r'\1', word)

        for known_word in known_words:
            known_word = known_word.lower()

            # Очищаем известное слово от повторяющихся букв
            known_cleaned = re.sub(r'(.)\1{2,}', r'\1', known_word)

            # Вычисляем расстояние Левенштейна
            distance = self._levenshtein_distance(word_cleaned, known_cleaned)
            max_len = max(len(word_cleaned), len(known_cleaned))
            ratio = 1 - (distance / max_len)

            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = known_word
                logging.debug(f"Found match: '{known_word}' for '{word}' with ratio {ratio}")

        return best_match

    def _clean_word(self, word: str) -> str:
        """Базовая очистка слова от ошибок"""
        # Приводим к нижнему регистру
        word = word.lower().strip()

        # Удаляем все знаки препинания
        word = re.sub(r'[.,!?:;()[\]{}«»""\'`]', '', word)

        # Удаляем повторяющиеся буквы (оставляем максимум две подряд)
        word = re.sub(r'(.)\1{2,}', r'\1\1', word)

        # Удаляем одиночные буквы в конце слова
        word = re.sub(r'[а-яё]$', '', word)

        return word.strip()

    def _get_word_prefix(self, word: str, min_length: int = 5) -> str:
        """Получает значимый префикс слова"""
        word = self._clean_word(word)
        if len(word) <= min_length:
            return word
        return word[:min_length]

    def _find_by_prefix(self, word: str) -> Optional[str]:
        """Поиск слова по префикс в известных минералах"""
        prefix = self._get_word_prefix(word)
        if not prefix:
            return None

        matches = []
        for known_word in self.dictionary_splitter.known_minerals:
            if known_word.lower().startswith(prefix):
                matches.append(known_word)

        if matches:
            # Выбираем самое короткое совпадение как наиболее вероятное
            return min(matches, key=len)
        return None

    def _clean_text(self, text: str) -> str:
        """Очищает текст от случайных символов и нормализует пробелы"""
        try:
            # Базовая очистка
            text = text.lower().strip()

            # Снаала пробуем найти точное соответствие
            exact_match = self.db.get_exact_mapping(text)
            if exact_match:
                return text

            # Если точное соответствие не найдено, очищаем текст
            words = text.split()
            cleaned_words = []

            for word in words:
                # Пропускаем короткие слова
                if len(word) <= 1 and not word in ['и', 'в', 'с', 'к', 'у', 'о']:
                    continue

                # Очищаем от знаков препинания
                cleaned_word = re.sub(r'[.,!?:;()[\]{}«»""\'`]', '', word)

                # Удаляем повторяющиеся буквы в конце
                if re.search(r'(.)\1{2,}$', cleaned_word):
                    cleaned_word = re.sub(r'(.)\1{2,}$', r'\1', cleaned_word)

                # Пробуем разные варианты поиска в порядке приоритета:

                # 1. Проверяем слово как есть
                if cleaned_word in self.dictionary_splitter.known_minerals:
                    cleaned_words.append(cleaned_word)
                    continue

                # 2. Пробуем нормализовать через морфологию
                normalized = self._normalize_word(cleaned_word)
                if normalized in self.dictionary_splitter.known_minerals:
                    cleaned_words.append(normalized)
                    continue

                # 3. Ищем похожие слова
                similar = self._find_similar_word(cleaned_word, self.dictionary_splitter.known_minerals)
                if similar:
                    cleaned_words.append(similar)
                    continue

                # 4. Проверяем нормализованную форму на похожие слова
                if normalized != cleaned_word:
                    similar_normalized = self._find_similar_word(normalized, self.dictionary_splitter.known_minerals)
                    if similar_normalized:
                        cleaned_words.append(similar_normalized)
                        continue

                # Если все проверки не сработали, добавляем нормализованное слово
                cleaned_words.append(normalized)

            # Собираем текст обратно
            result = ' '.join(cleaned_words)

            # Проверяем получившийся результат на точное соответствие
            final_match = self.db.get_exact_mapping(result)
            if final_match:
                return result

            logging.debug(f"Cleaned text '{text}' to '{result}'")
            return result

        except Exception as e:
            logging.error(f"Error cleaning text '{text}': {e}")
            return text

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

class EnhancedMineralClassifier:
    def __init__(self, known_minerals: Set[str]):
        self.morph = pymorphy3.MorphAnalyzer()
        self.known_minerals = known_minerals
        
        # Словарь типичных ошибок написания
        self.common_errors = {
            'йй': 'й',
            'нн': 'н',
            'тт': 'т',
            'сс': 'с',
            'лл': 'л',
            # Типичные замены букв
            'о': 'а',  # акающие диалекты
            'а': 'о',
            'е': 'и',
            'и': 'е',
            # Сложные случаи
            'тс': 'ц',
            'дс': 'ц',
            'дц': 'ц',
        }
        
        # Словарь морфем и их вариантов
        self.morpheme_variants = {
            'песчан': ['песчан', 'песчян', 'песчен'],
            'глинист': ['глинист', 'глинест'],
            'гравийн': ['гравийн', 'гравин'],
            'щебнист': ['щебнист', 'щебенист'],
        }
        
        # Контекстные маркеры с весами
        self.context_weights = {
            'стоительный': 1.5,
            'технический': 1.3,
            'ювелирный': 1.8,
            'поделочный': 1.6,
            'рудный': 1.4,
        }

    def classify(self, term: str, return_multiple: bool = False, confidence_threshold: float = 0.6) -> Union[Dict, List[Dict]]:
        """Классифицирует термин с возможностью возврата нескольких вариантов"""
        try:
            # Базовая очистка
            term = term.lower().strip()
            
            # Удаляем заки препинания и повторяющиеся буквы
            cleaned_term = re.sub(r'[.,!?:;()[\]{}«»""\'`]', '', term)
            cleaned_term = re.sub(r'(.)\1{2,}', r'\1', cleaned_term)
            
            # 1. Проверяем точное соотвтствие
            if cleaned_term in self.known_minerals:
                return {
                    'term': cleaned_term,
                    'confidence': 1.0,
                    'match_type': 'exact'
                }
            
            # 2. Проверяем морфологическую форму
            normalized = self._normalize_word(cleaned_term)
            if normalized in self.known_minerals:
                return {
                    'term': normalized,
                    'confidence': 0.95,
                    'match_type': 'morphological'
                }
            
            # 3. Проверяем перестановки букв
            transposed = self._find_similar_with_transpositions(cleaned_term)
            if transposed:
                return {
                    'term': transposed,
                    'confidence': 0.9,
                    'match_type': 'transposition'
                }
            
            # 4. Ищем похожие слова
            fuzzy_matches = self._find_fuzzy_matches(cleaned_term, min_ratio=75)
            if fuzzy_matches:
                # Берем лучшее совпадение
                best_match = max(fuzzy_matches, key=lambda x: x['confidence'])
                if best_match['confidence'] >= confidence_threshold:
                    return best_match
            
            # 5. Проверяем нормализованную форму на похожие слова
            if normalized != cleaned_term:
                fuzzy_matches_norm = self._find_fuzzy_matches(normalized, min_ratio=75)
                if fuzzy_matches_norm:
                    best_match = max(fuzzy_matches_norm, key=lambda x: x['confidence'])
                    if best_match['confidence'] >= confidence_threshold:
                        return best_match
            
            # Если ничего не нашли
            return self._get_unknown_result()
            
        except Exception as e:
            logging.error(f"Error in classify: {str(e)}")
            return self._get_unknown_result()

    def _normalize_word(self, word: str) -> str:
        """Приводит слово к нормальной форме"""
        try:
            # Получаем все возможные разборы слова
            parses = self.morph.parse(word.lower())
            if not parses:
                return word.lower()

            # Ищем существительное среди разборов
            noun_parses = [p for p in parses if 'NOUN' in p.tag]
            if noun_parses:
                normalized = noun_parses[0].normal_form
                logging.debug(f"Normalized '{word}' to '{normalized}' (noun)")
                return normalized

            # Если существительное не найдено, берем первый разбор
            normalized = parses[0].normal_form
            logging.debug(f"Normalized '{word}' to '{normalized}' (other)")
            return normalized

        except Exception as e:
            logging.error(f"Error normalizing word '{word}': {e}")
            return word.lower()

    def _find_similar_with_transpositions(self, word: str) -> Optional[str]:
        """Поиск похожих слов с учетом перестановок букв"""
        if len(word) < 3:
            return None
            
        # Генерируем варианты с перестановками
        variants = {word}
        chars = list(word)
        
        # Перестановки соседних букв
        for i in range(len(chars) - 1):
            chars_copy = chars.copy()
            chars_copy[i], chars_copy[i + 1] = chars_copy[i + 1], chars_copy[i]
            variants.add(''.join(chars_copy))
        
        # Проверяем все варианты
        for variant in variants:
            if variant in self.known_minerals:
                logging.debug(f"Found exact transposition match: '{word}' -> '{variant}'")
                return variant
            
            # Ищем похожие слова для каждого варианта
            for known_word in self.known_minerals:
                # Проверяем начало слова
                if known_word.startswith(variant[:3]):
                    ratio = fuzz.ratio(variant, known_word)
                    if ratio >= 85:
                        logging.debug(f"Found similar word for transposition: '{word}' -> '{known_word}' (ratio: {ratio})")
                        return known_word
        
        return None

    def _find_fuzzy_matches(self, term: str, min_ratio: int = 75) -> List[Dict]:
        """Поиск похожих терминов"""
        matches = []
        
        for known_term in self.known_minerals:
            # Проверяем начало слова
            if known_term.startswith(term[:3]) and len(term) >= 3:
                ratio = fuzz.ratio(term, known_term) * 1.2  # Повышаем вес дл совпадений в начале
            else:
                ratio = fuzz.ratio(term, known_term)
                
            if ratio >= min_ratio:
                matches.append({
                    'term': known_term,
                    'confidence': ratio / 100,
                    'match_type': 'fuzzy'
                })
        
        return matches

    def _get_unknown_result(self) -> Dict:
        """Возвращает результат для неизвестного термина"""
        return {
            'result': {
                'normalized_name_for_display': 'неизвестно',
                'pi_name_gbz_tbz': 'неизвестно',
                'pi_group_is_nedra': 'неизвестно',
                'pi_measurement_unit': '',
                'pi_measurement_unit_alternative': ''
            },
            'confidence': 0.0,
            'match_type': 'unknown'
        }

    def _find_exact_match(self, term: str) -> Optional[Dict]:
        """Поиск точного соответствия в известных минералах"""
        if term in self.known_minerals:
            return {
                'normalized_name_for_display': term,
                'confidence': 1.0,
                'match_type': 'exact'
            }
        return None

    def _analyze_compound_term(self, term: str) -> List[Dict]:
        """Анализ составного термина"""
        results = []
        
        # Разбиваем термин на части
        parts = []
        if '-' in term:
            parts.extend(term.split('-'))
        if ' ' in term:
            parts.extend(term.split())
        
        # Анализируем каждую часть
        for part in parts:
            # Нормализуем часть
            normalized = self._normalize_word(part)
            
            # Ищем точные совпадения
            if normalized in self.known_minerals:
                results.append({
                    'term': normalized,
                    'confidence': 0.9,  # Немного снижаем уверенность для частей
                    'match_type': 'compound_part'
                })
                
            # Ищем похожие термины
            fuzzy_matches = self._find_fuzzy_matches(normalized)
            if fuzzy_matches:
                # Добавляем лучшее совпадение
                best_match = max(fuzzy_matches, key=lambda x: x['confidence'])
                best_match['confidence'] *= 0.8  # Снижаем уверенность для нечетких совпадений
                results.append(best_match)
        
        return results

    def _calculate_context_weight(self, term: str, context: str) -> float:
        """Вычисляет вес контекста"""
        weight = 1.0
        
        # Проверяем контекстные маркеры
        for marker, marker_weight in self.context_weights.items():
            if marker in context:
                weight *= marker_weight
                
        # Учитываем специфические комбинации
        if 'рудный' in context and any(metal in term for metal in ['золото', 'серебро', 'платина']):
            weight *= 1.5
            
        if 'технический' in context and any(stone in term for stone in ['алмаз', 'корунд']):
            weight *= 1.4
            
        return weight

    def _analyze_with_context(self, term: str) -> List[Dict]:
        """Анализ термина с учетом контекста"""
        results = []
        
        # Извлекаем контекст из скобок
        main_term = term
        context = ""
        if '(' in term and ')' in term:
            main_term = term.split('(')[0].strip()
            context = term.split('(')[1].split(')')[0].strip()
        
        # Анализируем основной термин
        main_results = self._find_fuzzy_matches(main_term)
        
        # Если есть контекст, корректируем уверенность
        if context and main_results:
            context_weight = self._calculate_context_weight(main_term, context)
            for result in main_results:
                result['confidence'] *= context_weight
                result['context'] = context
        
        results.extend(main_results)
        return results

    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """Объединяет и фильтрует результаты"""
        if not results:
            return []
            
        # Группируем результаты по термину
        merged = {}
        for result in results:
            term = result['term']
            if term not in merged:
                merged[term] = result
            else:
                # Обновляем уверенность, если новый результат лучше
                if result['confidence'] > merged[term]['confidence']:
                    merged[term] = result
        
        # Сортируем по уверенности
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_results
