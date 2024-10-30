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
            
            # Если точного соответствия нет, проверяем базовое соответствие
            base_info = self.dictionary_splitter.get_base_mineral_info(mineral_name)
            if base_info:
                logging.debug(f"Found base mineral match for {mineral_name}: {base_info}")
                return base_info
            
            # Если не нашли базовое соответствие, ищем по компонентам
            components = self._split_input(mineral_name)
            best_result = None
            best_priority = -1
            
            for component in components:
                result = self.db.get_mapping(component)
                if result:
                    logging.debug(f"Found mapping for component {component}: {result}")
                    context_priority = self._get_context_priority(component)
                    if context_priority > best_priority:
                        best_result = result
                        best_priority = context_priority
            
            if best_result:
                logging.debug(f"Using best result: {best_result}")
                return best_result
            
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

    def _get_context_priority(self, component: str) -> int:
        """Возвращает приоритет для компонента на основе ключевых слов"""
        keyword_weights = {
            'силикат': 3,
            'бетон': 2,
            'строительный': 1,
            'балласт': 1
        }
        
        priority = 0
        for keyword, weight in keyword_weights.items():
            if keyword in component:
                priority += weight
        
        return priority

    def _split_input(self, text: str) -> List[str]:
        """Разбиение входного текста на компоненты с сохранением контекста"""
        components = []
        text = text.lower().strip()
        
        # Добавляем полный те��ст
        components.append(text)
        
        # Извлекаем основной термин (до скобок)
        main_term = text.split('(')[0].strip()
        if main_term and main_term != text:
            components.append(main_term)
        
        # Извлекаем термины в скобках
        bracket_terms = re.findall(r'\((.*?)\)', text)
        for term in bracket_terms:
            term = term.strip()
            if term:
                components.append(term)
                # Разбиваем по запятым
                for subterm in term.split(','):
                    subterm = subterm.strip()
                    if subterm and subterm not in components:
                        components.append(subterm)
        
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
            # Ище�� похожие термины в известных связях
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
