import re
import logging
from typing import List, Tuple

class TextCleaner:
    def __init__(self):
        # Паттерны для очистки текста
        self.cleanup_patterns = {
            r'[.,!?:;]': '',  # Знаки препинания
            r'\s+': ' ',      # Множественные пробелы
            r'[^\w\s()-]': '' # Все кроме букв, цифр, пробелов, дефисов и скобок
        }
        
        # Замены букв
        self.char_replacements = {
            'йй': 'й',
            'нн': 'н',
            'тт': 'т',
            'сс': '��',
            'лл': 'л',
            'оо': 'о',
            'ее': 'е',
            'аа': 'а',
            'ии': 'и'
        }

    def clean_text(self, text: str) -> str:
        """Базовая очистка текста"""
        text = text.lower().strip()
        
        # Применяем паттерны очистки
        for pattern, replacement in self.cleanup_patterns.items():
            text = re.sub(pattern, replacement, text)
            
        # Применяем замены букв
        for old, new in self.char_replacements.items():
            text = text.replace(old, new)
            
        return text.strip()

    def extract_context(self, text: str) -> Tuple[str, List[str]]:
        """Извлечение основного термина и контекста из скобок"""
        main_term = text
        contexts = []
        
        if '(' in text and ')' in text:
            main_term = text.split('(')[0].strip()
            # Извлекаем все контексты из скобок
            context_parts = re.findall(r'\((.*?)\)', text)
            for context in context_parts:
                # Разбиваем контекст по запятым
                contexts.extend([c.strip() for c in context.split(',')])
                
        return main_term, contexts

    def remove_duplicates(self, words: List[str]) -> List[str]:
        """Удаление дубликатов с сохранением порядка"""
        seen = set()
        return [w for w in words if not (w in seen or seen.add(w))] 