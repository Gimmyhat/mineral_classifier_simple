import pandas as pd
import logging
from pathlib import Path
import re
from typing import List, Dict
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

class DictionarySplitter:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.normalized_dict = {}  # Словарь для уникальных комбинаций
        self.variations_dict = {}  # Словарь вариантов написания
        
        # Контекстные маркеры для определения типа использования
        self.usage_markers = {
            'балластное сырье': 'Балластное сырье',
            'строительные камни': 'Камни строительные',
            'силикатные изделия': 'Пески для бетона и силикатных изделий',
            'наполнители бетона': 'Пески для бетона и силикатных изделий',
            'строительные материалы': 'Песчано-гравийные материалы'
        }
        
        # Стоп-слова остаются те же...

    def _create_unique_key(self, normalized_name: str, gbz_name: str, group_name: str) -> str:
        """Создание уникального ключа для комбинации"""
        return f"{normalized_name}|{gbz_name}|{group_name}"

    def _extract_usage_context(self, text: str) -> List[str]:
        """Извлечение контекста использования с приоритетами"""
        text = text.lower()
        contexts = []
        
        for marker, gbz in self.usage_markers.items():
            if marker in text:
                contexts.append((gbz, self._get_context_priority(marker)))
                
        # Сортируем контексты по приоритету
        contexts.sort(key=lambda x: x[1], reverse=True)
        return [context[0] for context in contexts]

    def _get_context_priority(self, marker: str) -> int:
        """Возвращает приоритет для контекста"""
        priority_map = {
            'силикатные изделия': 3,
            'наполнители бетона': 2,
            'строительные камни': 1,
            'балластное сырье': 1
        }
        return priority_map.get(marker, 0)

    def process_file(self):
        """Обработка файла с сохранением всех уникальных комбинаций"""
        df = pd.read_excel(
            self.input_file,
            header=3,
            usecols=[2,3,4,5,6,7]
        )
        
        df.columns = [
            "pi_variants", "normalized_name_for_display", "pi_name_gbz_tbz",
            "pi_group_is_nedra", "pi_measurement_unit", "pi_measurement_unit_alternative"
        ]
        
        df = df.fillna('')
        
        # Обрабатываем каждую строку
        for _, row in df.iterrows():
            normalized_name = row['normalized_name_for_display'].lower().strip()
            gbz_name = row['pi_name_gbz_tbz'].strip()
            group_name = row['pi_group_is_nedra'].strip()
            
            if not all([normalized_name, gbz_name, group_name]):
                continue
            
            # Создаем уникальный ключ для комбинации
            unique_key = self._create_unique_key(normalized_name, gbz_name, group_name)
            
            # Добавляем в словарь нормализованных названий
            if unique_key not in self.normalized_dict:
                self.normalized_dict[unique_key] = {
                    'normalized_name': normalized_name,
                    'gbz_name': gbz_name,
                    'group_name': group_name,
                    'measurement_unit': row['pi_measurement_unit'],
                    'measurement_unit_alt': row['pi_measurement_unit_alternative']
                }
            
            # Обрабатываем варианты написания
            variant_text = row['pi_variants'].lower().strip()
            if not variant_text:
                continue
            
            # Извлекаем контекст использования
            usage_contexts = self._extract_usage_context(variant_text)
            
            # Если нашли контекст использования, связываем вариант с соответствующей комбинацией
            if usage_contexts:
                for context in usage_contexts:
                    matching_key = self._create_unique_key(normalized_name, context, group_name)
                    if matching_key in self.normalized_dict:
                        self.variations_dict[variant_text] = matching_key
            else:
                # Если контекст не найден, связываем с текущей комбинацией
                self.variations_dict[variant_text] = unique_key
            
            # Добавляем также отдельные компоненты варианта
            components = self._split_components(variant_text)
            for component in components:
                if component and len(component) > 2:
                    if usage_contexts:
                        for context in usage_contexts:
                            matching_key = self._create_unique_key(normalized_name, context, group_name)
                            if matching_key in self.normalized_dict:
                                self.variations_dict[component] = matching_key
                    else:
                        self.variations_dict[component] = unique_key

    def _split_components(self, text: str) -> List[str]:
        """Разбиение текста на компоненты с сохранением контекста"""
        components = []
        
        # Извлекаем основной термин
        main_term = text.split('(')[0].strip()
        if main_term:
            components.append(main_term)
        
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
        
        return components

    def save_files(self, output_dir: str = 'data'):
        """Сохранение результатов"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Сохраняем словарь уникальных комбинаций
        dict_entries = []
        for unique_key, data in self.normalized_dict.items():
            entry = {
                'unique_key': unique_key,
                'normalized_name': data['normalized_name'],
                'gbz_name': data['gbz_name'],
                'group_name': data['group_name'],
                'measurement_unit': data['measurement_unit'],
                'measurement_unit_alt': data['measurement_unit_alt']
            }
            dict_entries.append(entry)
        
        df_dict = pd.DataFrame(dict_entries)
        dict_path = output_path / 'dictionary.xlsx'
        df_dict.to_excel(dict_path, index=False)
        logging.info(f"Saved dictionary with {len(df_dict)} entries to {dict_path}")
        
        # Сохраняем вариации
        var_entries = [
            {'variant': variant, 'unique_key': unique_key}
            for variant, unique_key in self.variations_dict.items()
        ]
        
        df_var = pd.DataFrame(var_entries)
        var_path = output_path / 'variations.xlsx'
        df_var.to_excel(var_path, index=False)
        logging.info(f"Saved variations with {len(df_var)} entries to {var_path}")

def main():
    input_file = 'Справочник_для_редактирования_09.10.2024.xlsx'
    splitter = DictionarySplitter(input_file)
    splitter.process_file()
    splitter.save_files()

if __name__ == "__main__":
    main()