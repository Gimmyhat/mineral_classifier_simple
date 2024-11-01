import logging
import pymorphy3
from typing import Set, List, Dict

class MorphologyProcessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Словарь исключений для нормализации
        self.normalization_exceptions = {
            # Базовые формы минералов
            'известняк': 'известняк',
            'алюминий': 'алюминий',
            'золото': 'золото',
            'песок': 'песок',
            
            # Падежные формы
            'известняка': 'известняк',
            'алюминия': 'алюминий',
            'золота': 'золото',
            'песка': 'песок',
            
            # Опечатки
            'известнак': 'известняк',
            'алюминийй': 'алюминий',
            'алюминий': 'алюминий',
            'пескок': 'песок',
        }
        
        # Словарь окончаний
        self.endings = {
            'существительные': ['а', 'я', 'ы', 'и', 'у', 'ю', 'ой', 'ей', 'ом', 'ем', 'е'],
            'прилагательные': ['ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие'],
            'причастия': ['щий', 'щая', 'щее', 'щие', 'вший', 'вшая', 'вшее', 'вшие']
        }

    def normalize_word(self, word: str) -> str:
        """Нормализация слова с учетом исключений и морфологии"""
        word = word.lower().strip()
        
        # Проверяем исключения
        if word in self.normalization_exceptions:
            return self.normalization_exceptions[word]
            
        # Убираем повторяющиеся буквы в конце
        while len(word) > 1 and word[-1] == word[-2]:
            word = word[:-1]
            
        # Проверяем исключения после очистки
        if word in self.normalization_exceptions:
            return self.normalization_exceptions[word]
            
        # Морфологический анализ
        parses = self.morph.parse(word)
        if not parses:
            return word
            
        # Приоритет существительным
        noun_parses = [p for p in parses if 'NOUN' in p.tag]
        if noun_parses:
            normalized = noun_parses[0].normal_form
            logging.debug(f"Normalized '{word}' to '{normalized}' (noun)")
            return normalized
            
        # Если существительное не найдено, берем первый разбор
        normalized = parses[0].normal_form
        logging.debug(f"Normalized '{word}' to '{normalized}' (other)")
        return normalized

    def get_word_forms(self, word: str) -> List[str]:
        """Получение всех возможных форм слова"""
        forms = set()
        forms.add(word)
        
        # Добавляем известные формы из исключений
        if word in self.normalization_exceptions:
            base_form = self.normalization_exceptions[word]
            forms.add(base_form)
            # Добавляем все известные формы для этого базового слова
            for exception, normalized in self.normalization_exceptions.items():
                if normalized == base_form:
                    forms.add(exception)
        
        # Получаем формы через морфологический анализатор
        parses = self.morph.parse(word)
        for parse in parses:
            forms.add(parse.normal_form)
            # Добавляем все возможные формы слова
            for form in parse.lexeme:
                forms.add(form.word)
                
        return list(forms)

    def is_valid_word(self, word: str) -> bool:
        """Проверка валидности слова"""
        # Минимальная длина
        if len(word) < 3:
            return False
            
        # Проверка на наличие цифр
        if any(char.isdigit() for char in word):
            return False
            
        # Проверка на наличие только служебных слов
        service_words = {'для', 'при', 'под', 'над', 'от', 'до', 'из', 'без'}
        if word.lower() in service_words:
            return False
            
        return True

    def split_compound_word(self, word: str) -> List[str]:
        """Разбиение составного слова на компоненты"""
        components = []
        
        # Разбиение по дефису
        if '-' in word:
            parts = [p.strip() for p in word.split('-')]
            components.extend(parts)
            # Добавляем вариант с пробелом
            components.append(' '.join(parts))
            
        # Разбиение по пробелу
        elif ' ' in word:
            components.extend([p.strip() for p in word.split()])
            
        # Если нет разделителей, возвращаем исходное слово
        else:
            components.append(word)
            
        return components 