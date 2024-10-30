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

        # Проверка на наличе только служебных слов
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
                'pi_measurement_unit_alternative': '' if parsed['measurement_unit_alt'] in ['nan', 'None', ''] else
                parsed['measurement_unit_alt']
            }
        return None

    def get_base_form(self, word: str) -> str:
        """Получение базовой форм слова"""
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
        """Загрузка словарей в Redis"""
        try:
            # Загружаем основной словарь и вариации
            df_dict = pd.read_excel(dictionary_path)
            df_var = pd.read_excel(variations_path)

            # Очищаем существующие данные
            self.redis_client.flushdb()

            # Загружаем основной словарь
            for _, row in df_dict.iterrows():
                dict_key = f"dictionary:{row['unique_key']}"
                self.redis_client.hmset(dict_key, {
                    'normalized_name': row['normalized_name'],
                    'gbz_name': row['gbz_name'],
                    'group_name': row['group_name'],
                    'measurement_unit': row['measurement_unit'],
                    'measurement_unit_alt': row['measurement_unit_alt']
                })

            # Создаем словарь для хранения вариаций и их приоритетов
            variant_priorities = {}

            # Обрабатываем вариации и определяем их приоритеты
            for _, row in df_var.iterrows():
                variant = row['variant'].lower()
                unique_key = row['unique_key']

                # Получаем базовое название из варианта (без скобок)
                base_name = variant.split('(')[0].strip()

                # Определяем приоритет на основе контекста
                priority = 0

                # Если это точное совпадение с базовым названием (без дополнительного контекста)
                if variant == base_name:
                    priority = 1

                # Если вариант содержит уточняющий контекст в скобках
                if '(' in variant:
                    # Более высокий приоритет для вариантов с уточнением
                    priority = 2
                    # Особый приоритет для определенных контекстов
                    if 'бокситы' in variant.lower():
                        priority = 3

                # Сохраняем вариант и его приоритет
                if variant not in variant_priorities or priority > variant_priorities[variant]['priority']:
                    variant_priorities[variant] = {
                        'unique_key': unique_key,
                        'priority': priority
                    }

            # Загружаем вариации в Redis с учетом приоритетов
            for variant, data in variant_priorities.items():
                variation_key = f"variation:{variant}"
                self.redis_client.set(variation_key, data['unique_key'])

                # Если это базовое название, также сохраняем его отдельно
                base_name = variant.split('(')[0].strip()
                if base_name == variant:
                    base_key = f"variation:{base_name}"
                    self.redis_client.set(base_key, data['unique_key'])

            logging.info(f"Loaded {len(df_dict)} dictionary entries and {len(df_var)} variations")

        except Exception as e:
            logging.error(f"Error loading dictionaries: {str(e)}")
            raise

    def get_exact_mapping(self, term: str) -> Dict[str, str]:
        """Получает точное соответствие из Redis"""
        try:
            term = term.lower().strip()

            # Ищем точное соответствие в вариациях
            variation_key = f"variation:{term}"
            unique_key = self.redis_client.get(variation_key)

            logging.debug(f"Looking for variation key: {variation_key}")
            if unique_key:
                logging.debug(f"Found unique key: {unique_key}")

                # Декодируем bytes в строку, если это bytes
                if isinstance(unique_key, bytes):
                    unique_key = unique_key.decode('utf-8')

                # Получаем данные по уникальному ключу
                dict_key = f"dictionary:{unique_key}"
                logging.debug(f"Looking for dictionary key: {dict_key}")
                data = self.redis_client.hgetall(dict_key)

                if data:
                    # Преобразуем bytes в строки для всех значений
                    decoded_data = {}
                    for key, value in data.items():
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        decoded_data[key] = value

                    return {
                        'normalized_name_for_display': decoded_data['normalized_name'],
                        'pi_name_gbz_tbz': decoded_data['gbz_name'],
                        'pi_group_is_nedra': decoded_data['group_name'],
                        'pi_measurement_unit': decoded_data['measurement_unit'],
                        'pi_measurement_unit_alternative': decoded_data['measurement_unit_alt']
                    }

            return None

        except Exception as e:
            logging.error(f"Error in get_exact_mapping: {str(e)}")
            return None

    def check_health(self) -> bool:
        """Проверка состояния подключения к Redis"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logging.error(f"Redis health check failed: {e}")
            return False