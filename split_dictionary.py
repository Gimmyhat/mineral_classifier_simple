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
        self.known_minerals = set()  # Множество известных минералов и ПИ
        self.base_minerals = {}  # Словарь базовых минералов и их стандартных категорий

        # Контекстные маркеры для определения типа использования
        self.usage_markers = {
            'балластное сырье': 'Балластное сырье',
            'строительные камни': 'Камни строительные',
            'силикатные изделия': 'Пески для бетона и силикатных изделий',
            'наполнители бетона': 'Пески для бетона и силикатных изделий',
            'строиельные материалы': 'Песчано-гравийные материалы'
        }

        # Стоп-слова остаются те же...

    def _create_unique_key(self, normalized_name: str, gbz_name: str, group_name: str) -> str:
        """Создание уникального ключа для комбинации"""
        return f"{normalized_name}|{gbz_name}|{group_name}"

    def _extract_usage_context(self, text: str) -> List[str]:
        """Извлечение контек��та использования с приоритетами"""
        text = text.lower()
        contexts = []

        # Проверяем, является ли текст базовым названием минерала
        if text.strip() in self.base_minerals:
            return [self.base_minerals[text.strip()]['gbz_name']]

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
            usecols=[2, 3, 4, 5, 6, 7]
        )

        df.columns = [
            "pi_variants", "normalized_name_for_display", "pi_name_gbz_tbz",
            "pi_group_is_nedra", "pi_measurement_unit", "pi_measurement_unit_alternative"
        ]

        df = df.fillna('')

        # Создаем словарь для хранения базовых вариантов и их приоритетов
        base_variants = {}

        # Первый проход: собираем все варианты и их приоритеты
        for _, row in df.iterrows():
            variant_text = row['pi_variants'].lower().strip()
            normalized_name = row['normalized_name_for_display'].lower().strip()
            gbz_name = row['pi_name_gbz_tbz'].strip()
            group_name = row['pi_group_is_nedra'].strip()

            if not all([normalized_name, gbz_name, group_name]):
                continue

            # Создаем уникальный ключ
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

            if not variant_text:
                continue

            # Получаем базовое название (без скобок)
            base_name = variant_text.split('(')[0].strip()

            # Определяем приоритет варианта
            priority = self._get_variant_priority(variant_text, normalized_name)

            # Сохраняем вариант с его приоритетом
            if base_name not in base_variants or priority > base_variants[base_name]['priority']:
                base_variants[base_name] = {
                    'unique_key': unique_key,
                    'priority': priority
                }

            # Добавляем полный вариант в словарь вариаций
            self.variations_dict[variant_text] = unique_key

        # Второй проход: добавляем базовые варианты с учетом приоритетов
        for base_name, data in base_variants.items():
            self.variations_dict[base_name] = data['unique_key']

    def _get_variant_priority(self, variant: str, normalized_name: str) -> int:
        """Определяет приоритет варианта написания"""
        # Базовый приоритет
        priority = 0

        # Если это базовое название минерала (без скобок и уточнений)
        if '(' not in variant:
            priority += 2
            # Если это точное совпадение с нормализованным названием
            if variant == normalized_name:
                priority += 3

        # Если вариант содержит уточняющий контекст
        if '(' in variant:
            # Извлекаем основное название (до скобок)
            base_name = variant.split('(')[0].strip()

            # Если основное название совпадает с нормализованным
            if base_name == normalized_name:
                priority += 2

            # Анализируем контекст в скобках
            contexts = re.findall(r'\((.*?)\)', variant.lower())
            for context in contexts:
                # Приоритет для контекста, указывающего на тип/категорию
                if any(key in context for key in ['руда', 'сырье', 'металл', 'порода']):
                    priority += 1

                # Дополнительный приоритет для уточняющего контекста
                if normalized_name in context:
                    priority += 2

        return priority

    def _split_components(self, text: str) -> List[str]:
        """Разбиение текста на компоненты"""
        components = []

        # Извлекаем основной термин (до скобок)
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

    def contains_known_mineral(self, text: str) -> bool:
        """Проверяет, содержит ли текст известный минерал или ПИ"""
        text = text.lower().strip()
        terms = self._split_components(text)

        logging.debug(f"Checking terms {terms} against {len(self.known_minerals)} known minerals")

        # Проверяем каждый компонент текста
        for term in terms:
            if term in self.known_minerals:
                logging.debug(f"Found known mineral: {term}")
                return True

        logging.debug(f"No known minerals found in terms: {terms}")
        return False

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

    def get_base_mineral_info(self, mineral_name: str) -> Dict[str, str]:
        """Получает информацию о базовом минерале"""
        mineral_name = mineral_name.lower().strip()
        base_info = self.base_minerals.get(mineral_name)

        if base_info:
            # Преобразуем ключи в формат, ожидаемый фронтендом
            return {
                'normalized_name_for_display': base_info['normalized_name'],
                'pi_name_gbz_tbz': base_info['gbz_name'],
                'pi_group_is_nedra': base_info['group_name'],
                'pi_measurement_unit': base_info['measurement_unit'],
                'pi_measurement_unit_alternative': base_info['measurement_unit_alt']
            }
        return None


def main():
    input_file = 'Справочник_для_редактирования_09.10.2024.xlsx'
    splitter = DictionarySplitter(input_file)
    splitter.process_file()
    splitter.save_files()


if __name__ == "__main__":
    main()