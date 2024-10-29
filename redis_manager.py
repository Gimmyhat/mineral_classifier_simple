import redis
import json
import logging
from typing import Dict, Optional, List
from config import REDIS_CONFIG
import pandas as pd

class RedisManager:
    def __init__(self):
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.init_db()

    def init_db(self):
        """Инициализация базы данных"""
        try:
            # Проверяем соединение
            self.redis_client.ping()
            logging.debug("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise

    def add_mapping(self, variant: str, normalized_name: str, gbz_name: str, 
                   group_name: str, unit: str, unit_alt: str):
        """Добавление нового маппинга"""
        mapping = {
            'normalized_name': normalized_name,
            'gbz_name': gbz_name,
            'group_name': group_name,
            'measurement_unit': unit,
            'measurement_unit_alt': unit_alt
        }
        # Сохраняем маппинг
        self.redis_client.hset(
            f"mapping:{variant.lower()}", 
            mapping=json.dumps(mapping)
        )
        
        # Добавляем в индекс для поиска
        self.redis_client.sadd('variants', variant.lower())

    def add_synonym(self, word: str, base_form: str):
        """Добавление нового синонима"""
        self.redis_client.hset('synonyms', word.lower(), base_form.lower())

    def get_mapping(self, variant: str) -> Optional[Dict]:
        """Получение маппинга с учетом контекста"""
        try:
            variant = variant.lower().strip()
            
            # Получаем уникальный ключ для варианта
            unique_key = self.redis_client.hget('variations', variant)
            if unique_key:
                # Получаем данные по уникальному ключу
                data = self.redis_client.hget(f"dictionary:{unique_key}", 'mapping')
                if data:
                    parsed = json.loads(data)
                    return {
                        'normalized_name_for_display': parsed['normalized_name'],
                        'pi_name_gbz_tbz': parsed['gbz_name'],
                        'pi_group_is_nedra': parsed['group_name'],
                        'pi_measurement_unit': parsed['measurement_unit'],
                        'pi_measurement_unit_alternative': parsed['measurement_unit_alt']
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Error in get_mapping: {str(e)}")
            return None

    def _is_invalid_query(self, text: str) -> bool:
        """Проверка на невалидные запросы"""
        # Список стоп-слов и фраз
        stop_words = {
            'работы', 'исследования', 'разведка', 'поиски',
            'документация', 'отчет', 'проект', 'участок',
            'месторождение', 'площадь', 'объект', 'район',
            'территория', 'зона', 'блок', 'горизонт',
            'не связанное', 'не связанный', 'связанный',
            'прочие', 'другие', 'иные', 'разное'
        }
        
        # Минимальная длина валидного запроса
        if len(text.strip()) < 3:
            return True
        
        # Проверка на стоп-слова
        words = text.lower().split()
        if any(word in stop_words for word in words):
            return True
        
        # Проверка на наличие только служебных слов
        service_words = {'с', 'по', 'для', 'на', 'в', 'и', 'или'}
        if all(word in service_words for word in words):
            return True
        
        # Проверка на наличие цифр
        if any(char.isdigit() for char in text):
            return True
        
        return False

    def _get_dictionary_entry(self, normalized_name: str) -> Dict:
        """Получение записи из основного словаря"""
        data = self.redis_client.hget(f"dictionary:{normalized_name}", 'mapping')
        if data:
            parsed = json.loads(data)
            return {
                'normalized_name_for_display': normalized_name,
                'pi_name_gbz_tbz': parsed['gbz_name'],
                'pi_group_is_nedra': parsed['group_name'],
                'pi_measurement_unit': parsed['measurement_unit'],
                'pi_measurement_unit_alternative': '' if parsed['measurement_unit_alt'] in ['nan', 'None', ''] else parsed['measurement_unit_alt']
            }
        return None

    def get_base_form(self, word: str) -> str:
        """Получение базовой формы слова"""
        return self.redis_client.hget('synonyms', word.lower()) or word

    def bulk_add_mappings(self, mappings: list):
        """Массовое добавление маппингов"""
        pipe = self.redis_client.pipeline()
        for mapping in mappings:
            variant = mapping['variant'].lower()
            data = {
                'normalized_name': mapping['normalized_name'],
                'gbz_name': mapping['gbz_name'],
                'group_name': mapping['group_name'],
                'measurement_unit': mapping['measurement_unit'],
                'measurement_unit_alt': mapping['measurement_unit_alt']
            }
            # Исправляем формат команды hset
            pipe.hset(
                f"mapping:{variant}", 
                'mapping',  # ключ поля
                json.dumps(data)  # значение поля
            )
            pipe.sadd('variants', variant)
        pipe.execute()

    def _clean_variant(self, text: str) -> str:
        """Очищает текст от скобок и их содержимого"""
        import re
        # Удаляем содержимое скобок и сами скобки
        cleaned = re.sub(r'\([^)]*\)', '', text)
        # Очищаем от лишних пробелов
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    def _split_components(self, text: str) -> List[str]:
        """Разбивает текст на компоненты"""
        # Разделяем по скобкам и запятым
        import re
        # Сначала извлекаем основной термин (до первой скобки)
        main_term = text.split('(')[0].strip()
        
        # Затем извлекаем термины в скобках
        brackets_content = re.findall(r'\((.*?)\)', text)
        
        components = [main_term]
        for content in brackets_content:
            components.extend([term.strip() for term in content.split(',') if term.strip()])
            
        return [comp.lower() for comp in components if comp]

    def add_dictionary_entry(self, normalized_name: str, gbz_name: str, 
                           group_name: str, unit: str, unit_alt: str):
        """Добавление записи в основной словарь"""
        # Заменяем 'nan' на пустую строку перед сохранением
        unit_alt = '' if pd.isna(unit_alt) or unit_alt in ['nan', 'None'] else unit_alt
        
        data = {
            'gbz_name': gbz_name,
            'group_name': group_name,
            'measurement_unit': unit,
            'measurement_unit_alt': unit_alt  # Теперь здесь никогда не будет 'nan'
        }
        
        self.redis_client.hset(
            f"dictionary:{normalized_name.lower()}", 
            'mapping',  # ключ поля
            json.dumps(data)  # значение поля
        )

    def add_variation(self, variant: str, normalized_name: str):
        """Добавление вариации"""
        self.redis_client.hset('variations', variant.lower(), normalized_name.lower())

    def load_dictionaries(self, dictionary_path: str, variations_path: str):
        """Загрузка словарей с уникальными комбинациями"""
        try:
            # Загружаем основной словарь
            dict_df = pd.read_excel(dictionary_path)
            logging.debug(f"Loading dictionary from {dictionary_path}")
            
            # Загружаем каждую уникальную комбинацию
            for _, row in dict_df.iterrows():
                unique_key = row['unique_key']
                self.redis_client.hset(
                    f"dictionary:{unique_key}",
                    'mapping',
                    json.dumps({
                        'normalized_name': row['normalized_name'],
                        'gbz_name': row['gbz_name'],
                        'group_name': row['group_name'],
                        'measurement_unit': row['measurement_unit'],
                        'measurement_unit_alt': '' if pd.isna(row['measurement_unit_alt']) else row['measurement_unit_alt']
                    })
                )

            # Загружаем вариации
            var_df = pd.read_excel(variations_path)
            logging.debug(f"Loading variations from {variations_path}")
            
            # Загружаем связи вариантов с уникальными комбинациями
            for _, row in var_df.iterrows():
                self.redis_client.hset(
                    'variations',
                    row['variant'].lower(),
                    row['unique_key']
                )

            logging.debug(f"Loaded {len(dict_df)} dictionary entries and {len(var_df)} variations")

        except Exception as e:
            logging.error(f"Error loading dictionaries: {str(e)}")
            raise
