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
            
            # Варианты написания минералов
            'аргиллит': 'аргиллит',
            'аргилит': 'аргиллит',
            'аргилитовый': 'аргиллит',
            'аргиллитовый': 'аргиллит',
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
        """Получает все формы слова, включая единственное и множественное число"""
        try:
            forms = set()
            # Нормализуем написание
            normalized_word = self.normalize_spelling(word)
            words = normalized_word.split()
            
            for single_word in words:
                parsed = self.morph.parse(single_word)
                if not parsed:
                    continue
                    
                for p in parsed:
                    # Добавляем начальную форму
                    forms.add(p.normal_form)
                    
                    # Для существительных генерируем все формы
                    if 'NOUN' in p.tag:
                        # Все падежи в единственном числе
                        for case in ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']:
                            form = p.inflect({'sing', case})
                            if form:
                                forms.add(form.word)
                        
                        # Все падежи во множественном числе
                        for case in ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']:
                            form = p.inflect({'plur', case})
                            if form:
                                forms.add(form.word)
                    
                    # Для прилагательных генерируем согласованные формы
                    elif 'ADJF' in p.tag:
                        for gender in ['masc', 'femn', 'neut']:
                            for case in ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']:
                                for number in ['sing', 'plur']:
                                    form = p.inflect({gender, case, number})
                                    if form:
                                        forms.add(form.word)
            
            # Генерируем составные формы
            if len(words) > 1:
                single_forms = list(forms)
                compound_forms = set()
                
                # Создаем все возможные комбинации форм слов
                for form1 in single_forms:
                    for form2 in single_forms:
                        if form1 != form2:
                            compound_forms.add(f"{form1} {form2}")
                            compound_forms.add(f"{form2} {form1}")
                
                forms.update(compound_forms)
            
            return list(forms)
            
        except Exception as e:
            logging.error(f"Error in get_word_forms: {str(e)}")
            return [word]
            
    def normalize(self, word: str) -> str:
        """Нормализует слово (приводит к начальной форме)"""
        try:
            parsed = self.morph.parse(word)
            if parsed:
                # Берем самый вероятный разбор
                return parsed[0].normal_form
            return word
        except Exception as e:
            logging.error(f"Error in normalize: {str(e)}")
            return word

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

    def normalize_spelling(self, word: str) -> str:
        """Нормализует написание слова с учетом известных вариаций"""
        # Словарь известных вариаций написания
        spelling_variations = {
            'аргилит': 'аргиллит',
            'агрилит': 'аргиллит',
            'аргилид': 'аргиллит',
            # Можно добавить другие известные вариации
        }
        
        word = word.lower()
        return spelling_variations.get(word, word)