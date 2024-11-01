import redis
import json
import logging
from typing import Dict, Optional, List, Union
from config import REDIS_CONFIG
import pandas as pd
import re


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
        # Сохраняем маппиг
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
        service_words = {'с', 'по', 'дл', 'на', 'в', 'и', 'ли'}
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
        components = []
        
        # Извлекаем основной термин (до скобок)
        main_term = text.split('(')[0].strip()
        components.append(main_term)
        
        # Если есть дефис, добавляем варианты
        if '-' in main_term:
            # Вариант с пробелами
            components.append(main_term.replace('-', ' '))
            # Отдельные части
            parts = [p.strip() for p in main_term.split('-')]
            components.extend(parts)
        
        # Если есть пробелы, добавляем отдельные слова
        if ' ' in main_term:
            components.extend([w.strip() for w in main_term.split()])
        
        # Удаляем дубликаты и пустые строки
        return [c for c in dict.fromkeys(components) if c]

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

            # Сначала з��гружаем основной словарь
            for _, row in df_dict.iterrows():
                dict_key = f"dictionary:{row['unique_key']}"
                mapping_data = {
                    'normalized_name': row['normalized_name'],
                    'gbz_name': row['gbz_name'],
                    'group_name': row['group_name'],
                    'measurement_unit': row['measurement_unit'],
                    'measurement_unit_alt': row['measurement_unit_alt']
                }
                # Сохраняем как JSON строку
                self.redis_client.hset(dict_key, 'mapping', json.dumps(mapping_data))

            # Создаем словарь для хранения вариаций и их приоритетов
            variant_priorities = {}

            # Затем загружаем вариации
            for _, row in df_var.iterrows():
                variant = row['variant'].lower().strip()
                unique_key = row['unique_key']
                
                # Определяем приоритет варианта
                priority = 0
                
                # Если это точное совпадение с вариантом из файла
                if '(' in variant:
                    priority = 3  # Высший приоритет для вариантов с контекстом
                    # Сохраняем полный вариант как есть
                    self.redis_client.hset('variations', variant, unique_key)
                    
                    # Извлекаем основной термин и контексты
                    main_term = variant.split('(')[0].strip()
                    context_part = variant[variant.find('(')+1:variant.find(')')].strip()
                    contexts = [ctx.strip() for ctx in context_part.split(',')]
                    
                    # Сохраняем варианты с каждым контекстом
                    for context in contexts:
                        context_variant = f"{main_term} ({context})"
                        self.redis_client.hset('variations', context_variant, unique_key)
                    
                    # Сохраняем ос��овной термин
                    if main_term not in variant_priorities or priority > variant_priorities[main_term]['priority']:
                        variant_priorities[main_term] = {'unique_key': unique_key, 'priority': 2}
                else:
                    # Для вариантов без контекста
                    if variant not in variant_priorities or priority > variant_priorities[variant]['priority']:
                        variant_priorities[variant] = {'unique_key': unique_key, 'priority': 1}
                
                # Если есть дефис, сохраняем вариант с пробелом
                if '-' in variant:
                    space_variant = variant.replace('-', ' ')
                    if '(' in space_variant:
                        self.redis_client.hset('variations', space_variant, unique_key)
                        main_term = space_variant.split('(')[0].strip()
                        if main_term not in variant_priorities or 2 > variant_priorities[main_term]['priority']:
                            variant_priorities[main_term] = {'unique_key': unique_key, 'priority': 2}
                    else:
                        if space_variant not in variant_priorities or 1 > variant_priorities[space_variant]['priority']:
                            variant_priorities[space_variant] = {'unique_key': unique_key, 'priority': 1}

            # Сохраняем все варианты с учетом приоритетов
            for variant, data in variant_priorities.items():
                self.redis_client.hset('variations', variant, data['unique_key'])

            logging.info(f"Loaded {len(df_dict)} dictionary entries and {len(df_var)} variations")

        except Exception as e:
            logging.error(f"Error loading dictionaries: {str(e)}")
            raise

    def _normalize_input(self, text: str) -> List[str]:
        """Нормализует входной текст и возвращает список возможных вариантов"""
        text = text.lower().strip()
        variants = set()
        
        # Добавляем исходный текст
        variants.add(text)
        
        # 1. Убираем повторяющиеся буквы в конце
        cleaned = re.sub(r'(.)\1+$', r'\1', text)
        variants.add(cleaned)
        
        # 2. Обработка окончаний
        # Список типичных окончаний существительных в русском языке
        endings = ['ий', 'ый', 'ая', 'яя', 'ое', 'ее', 'а', 'я', 'о', 'е', 'ом', 'ем', 
                  'ами', 'ями', 'ах', 'ях', 'у', 'ю', 'ы', 'и', 'ей', 'ов', 'ев']
        
        # Находим базовую форму, убирая возможные окончаня
        base_word = text
        for ending in sorted(endings, key=len, reverse=True):  # Сортируем по длине, чтобы сначала проверять длинные окончания
            if text.endswith(ending) and len(text) > len(ending) + 2:  # +2 чтобы оставалось хотя бы 2 буквы основы
                base_word = text[:-len(ending)]
                variants.add(base_word)
                break
        
        # 3. Получаем варианты из Redis
        # Проверяем, есть ли в базе похожие слова
        for variant in self.redis_client.scan_iter(match=f"variation:{base_word}*"):
            if isinstance(variant, bytes):
                variant = variant.decode('utf-8')
            # Убираем префикс 'variation:'
            clean_variant = variant.split(':', 1)[1]
            variants.add(clean_variant)
        
        # 4. Добавляем варианты с возможными опечатками
        for v in list(variants):  # list() чтобы избежать изменения set во время итерации
            # Замена повторяющихся букв
            cleaned_var = re.sub(r'(.)\1+', r'\1', v)
            variants.add(cleaned_var)
            
            # Обработка частых замен букв (е/ё, й/и в конце и т.д.)
            if v.endswith('й'):
                variants.add(v[:-1] + 'и')
            if v.endswith('и'):
                variants.add(v[:-1] + 'й')
        
        logging.debug(f"Normalized variants for '{text}': {variants}")
        return list(variants)

    def get_exact_mapping(self, term: str) -> Optional[Dict]:
        """Получает точное соответствие из Redis"""
        try:
            if self._is_invalid_query(term):
                logging.debug(f"Invalid query: {term}")
                return None
                
            term = term.lower().strip()
            
            # 1. Пробуем найти точное соответствие для полного термина
            unique_key = self.redis_client.hget('variations', term)
            if unique_key:
                return self._get_mapping_by_key(unique_key)
            
            # 2. Извлекаем основной термин и контексты
            main_term = term
            contexts = []
            if '(' in term and ')' in term:
                main_term = term.split('(')[0].strip()
                # Извлекаем контексты из скобок и разбиваем по запятой
                context_part = term[term.find('(')+1:term.find(')')].strip()
                contexts = [ctx.strip() for ctx in context_part.split(',')]
            
            # 3. Формируем все возможные варианты запроса
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
            
            # Если в основном термине есть дефис, добавляем вариант с пробелом
            if '-' in main_term:
                space_variant = main_term.replace('-', ' ')
                variants.append(space_variant)
                # Также добавляем варианты с контекстом
                for context in contexts:
                    variants.append(f"{space_variant} ({context})")
                    variants.append(f"{space_variant} {context}")
            
            logging.debug(f"Trying variants: {variants}")
            
            # 4. Проверяем все варианты
            for variant in variants:
                unique_key = self.redis_client.hget('variations', variant)
                if unique_key:
                    logging.debug(f"Found match for variant: {variant}")
                    return self._get_mapping_by_key(unique_key)
            
            logging.debug(f"No match found for '{term}' and its variants")
            return None

        except Exception as e:
            logging.error(f"Error in get_exact_mapping: {str(e)}")
            return None

    def _get_mapping_by_key(self, unique_key: Union[str, bytes]) -> Optional[Dict]:
        """Получает маппинг по ключу"""
        try:
            if isinstance(unique_key, bytes):
                unique_key = unique_key.decode('utf-8')
                
            dict_key = f"dictionary:{unique_key}"
            mapping_data = self.redis_client.hget(dict_key, 'mapping')
            
            if mapping_data:
                if isinstance(mapping_data, bytes):
                    mapping_data = mapping_data.decode('utf-8')
                
                parsed = json.loads(mapping_data)
                return {
                    'normalized_name_for_display': parsed['normalized_name'],
                    'pi_name_gbz_tbz': parsed['gbz_name'],
                    'pi_group_is_nedra': parsed['group_name'],
                    'pi_measurement_unit': parsed['measurement_unit'],
                    'pi_measurement_unit_alternative': parsed['measurement_unit_alt']
                }
            return None
        except Exception as e:
            logging.error(f"Error getting mapping by key: {e}")
            return None

    def check_health(self) -> bool:
        """Проверка состояния подключения к Redis"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logging.error(f"Redis health check failed: {e}")
            return False

    def save_dictionaries(self, dictionary_path: str, variations_path: str):
        """Сохраняет текущее состояние словарей в файлы"""
        try:
            # Собираем данные для основного словаря
            dict_entries = []
            for key in self.redis_client.scan_iter("dictionary:*"):
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                data = self.redis_client.hget(key, 'mapping')
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    parsed = json.loads(data)
                    unique_key = key.split(':')[1]
                    dict_entries.append({
                        'unique_key': unique_key,
                        'normalized_name': parsed['normalized_name'],
                        'gbz_name': parsed['gbz_name'],
                        'group_name': parsed['group_name'],
                        'measurement_unit': parsed['measurement_unit'],
                        'measurement_unit_alt': parsed['measurement_unit_alt']
                    })

            # Собираем данные для вариаций
            var_entries = []
            for key, value in self.redis_client.hgetall('variations').items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                var_entries.append({
                    'variant': key,
                    'unique_key': value
                })

            # Сохраняем в Excel файлы
            df_dict = pd.DataFrame(dict_entries)
            df_var = pd.DataFrame(var_entries)

            df_dict.to_excel(dictionary_path, index=False)
            df_var.to_excel(variations_path, index=False)

            logging.info(f"Saved {len(dict_entries)} dictionary entries and {len(var_entries)} variations")

        except Exception as e:
            logging.error(f"Error saving dictionaries: {e}")
            raise