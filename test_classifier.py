import logging
from mineral_classifier import MineralClassifier

logging.basicConfig(level=logging.DEBUG)

def test_classifier():
    classifier = MineralClassifier('Справочник_для_редактирования_09.10.2024.xlsx')
    
    # Тест 1: Базовая классификация
    test_cases = [
        # Точные совпадения
        "золото",
        "песок",
        "алюминий",
        
        # Морфологические формы
        "золота",
        "песков",
        "алюминиевый",
        
        # Опечатки
        "золотоо",
        "пессок",
        "алюминийй",
        
        # Перестановки букв
        "злоото",
        "псеок",
        "аюлминий",
        
        # Знаки препинания
        "золото.",
        "песок,",
        "алюминий!",
        
        # Составные термины
        "песчано-гравийный материал",
        "глинисто-песчаные отложения",
        "щебеночно-песчаная смесь",
        
        # Контекст в скобках
        "алмаз (технический)",
        "золото (рудное)",
        "песок (строительный)",
        
        # Сложные случаи
        "песчано-гравийно-галечные отложения",
        "глинисто-алевритовые породы",
        "щебеночно-гравийно-песчаная смесь"
    ]
    
    print("\nТестирование классификатора:")
    print("-" * 80)
    
    for term in test_cases:
        print(f"\nТестовый термин: {term}")
        
        # Получаем один результат
        result = classifier.classify_mineral(term)
        print(f"Одиночный результат: {result}")
        
        # Получаем множественные результаты
        results = classifier.classify_mineral(term, return_multiple=True)
        print("Множественные результаты:")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res}")
            
        print("-" * 80)

if __name__ == "__main__":
    test_classifier() 