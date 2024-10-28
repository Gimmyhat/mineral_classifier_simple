import pandas as pd
import re
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any
from multiprocessing import Pool, Manager
from functools import partial
import tqdm  # Для отображения прогресса
import os

logging.basicConfig(level=logging.DEBUG)

class DataLoader:
    @staticmethod
    def load_excel(file_path: str, sheet_name: int = 0, header: int = 3, start_column: int = 2) -> pd.DataFrame:
        logging.debug(f"Attempting to load data from {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        df = df.iloc[:, start_column:]
        logging.debug(f"Data loaded successfully. Shape: {df.shape}")
        return df

class DataPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        new_column_names = [
            "pi_variants", "normalized_name_for_display", "pi_name_gbz_tbz",
            "pi_group_is_nedra", "pi_measurement_unit", "pi_measurement_unit_alternative",
        ]
        df.columns = new_column_names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df = df.dropna(subset=['pi_variants', 'normalized_name_for_display', 'pi_name_gbz_tbz', 'pi_group_is_nedra'])
        df['pi_measurement_unit'] = df['pi_measurement_unit'].replace('-', '').fillna('')
        df['pi_measurement_unit_alternative'] = df['pi_measurement_unit_alternative'].replace('-', '').fillna('')
        df = df.drop_duplicates(subset=['pi_variants'], keep='first')
        df['pi_variants'] = df['pi_variants'].apply(DataPreprocessor.clean_text)
        return df

class ModelTrainer:
    @staticmethod
    def train_model(X, y, model_class, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return model, accuracy

class MineralClassifier:
    def __init__(self, file_path: str):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.df = self.load_and_preprocess_data(file_path)
        
        # Оптимизация памяти - изменим эту часть
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Сначала заполняем пустые значения
                self.df[col] = self.df[col].fillna('')
                # Затем преобразуем в категориальный тип
                self.df[col] = self.df[col].astype('category')
        
        # Оптимизация векторизатора
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Ограничиваем количество признаков
            dtype=np.float32  # Используем float32 вместо float64
        )
        self.X = self.vectorizer.fit_transform(self.df['pi_variants'])
        self.train_models()

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = self.data_loader.load_excel(file_path)
        return self.preprocessor.preprocess_dataframe(df)

    def train_models(self):
        self.models = {}
        self.label_encoders = {}

        for column in ['pi_group_is_nedra', 'pi_name_gbz_tbz', 'normalized_name_for_display', 
                      'pi_measurement_unit', 'pi_measurement_unit_alternative']:
            le = LabelEncoder()
            # Используем уже заполненные значения
            y = le.fit_transform(self.df[column])
            self.label_encoders[column] = le

            if column in ['pi_group_is_nedra', 'pi_name_gbz_tbz']:
                model, accuracy = ModelTrainer.train_model(self.X, y, RandomForestClassifier, random_state=42)
            elif column == 'normalized_name_for_display':
                model, accuracy = ModelTrainer.train_model(self.X, y, RidgeClassifier)
            else:
                model, accuracy = ModelTrainer.train_model(self.X, y, LogisticRegression)

            self.models[column] = model
            logging.debug(f"Accuracy for {column}: {accuracy}")

    def classify_mineral(self, mineral_name: str) -> Dict[str, Any]:
        cleaned_name = self.preprocessor.clean_text(mineral_name)
        vectorized_name = self.vectorizer.transform([cleaned_name])

        predictions = {}
        probabilities = {}

        for column, model in self.models.items():
            le = self.label_encoders[column]
            pred = le.inverse_transform(model.predict(vectorized_name))[0]
            predictions[column] = pred if pred in self.df[column].unique() else "неизвестно"

            if hasattr(model, 'predict_proba'):
                probabilities[column] = model.predict_proba(vectorized_name)[0]
            else:
                probabilities[column] = model.decision_function(vectorized_name)[0]

            logging.debug(f"Probabilities for {column}: {probabilities[column]}")

        logging.debug(f"Final predictions: {predictions}")
        return predictions

    def check_classification(self, mineral_name: str):
        result = self.classify_mineral(mineral_name)
        print(f"\nРезультат классификации для '{mineral_name}':")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        mineral_data = self.df[self.df['pi_variants'].str.contains(mineral_name, case=False)]
        if not mineral_data.empty:
            print("\nДанные из исходного датасета:")
            print(mineral_data)
        else:
            print(f"\nМинерал '{mineral_name}' не найден в исходном датасете.")

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

    def process_excel(self, input_file: str, output_file: str = None, chunk_size: int = 100) -> str:
        """
        Обрабатывает входной Excel файл последовательно.
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
            
            # Обрабатываем минералы последовательно
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

if __name__ == "__main__":
    classifier = MineralClassifier('Справочник_для_редактирования_09.10.2024.xlsx')
    classifier.check_classification("авантюрин")
