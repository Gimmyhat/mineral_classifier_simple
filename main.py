from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import logging
import os
from mineral_classifier import MineralClassifier, BatchProcessor
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import uuid
from functools import lru_cache, wraps
from time import time
from database import DatabaseManager

logging.basicConfig(level=logging.DEBUG)

# Создаем директории
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Определяем lifespan ДО создания приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный енеджер жизненного цикла приложения"""
    # Код инициализации при запуске
    cleanup_old_results(results_dir)
    asyncio.create_task(cleanup_old_tasks())

    # Инициализируем классификатор в отдельной задаче
    def init_classifier():
        return get_classifier()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_classifier)
    
    logging.info("Application startup complete")
    
    yield  # Здесь приложение работает
    
    # Код очистки при выключении
    logging.info("Application shutdown")
    # Можно добавить очистку ресурсов при необходимости

# ПОСЛЕ определения lifespan создаем приложение
app = FastAPI(
    title="Mineral Classifier API",
    description="API для классификации олезных ископаемых",
    version="1.0.0",
    lifespan=lifespan
)

# Создаем экземпляры менеджеров
db_manager = DatabaseManager()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Mineral Classifier API",
        version="1.0.0",
        description="API для классификации олезных ископаемых",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Теперь устанавливаем custom_openapi после создания приложения
app.openapi = custom_openapi

# Добавляем поддержку CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Убедимся, что директория templates существует
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# В начале файла добавим:
# Изменим инициализацию классификатора
classifier = None
batch_processor = None


@lru_cache(maxsize=1)
def get_classifier():
    global classifier, batch_processor
    if classifier is None:
        logging.debug("Initializing MineralClassifier")
        classifier = MineralClassifier('Справочник_для_редактирования_09.10.2024.xlsx')
        batch_processor = BatchProcessor(classifier)
        logging.debug("MineralClassifier and BatchProcessor initialized")
    return classifier


def get_batch_processor():
    if batch_processor is None:
        get_classifier()  # Это инициализирует оба объекта
    return batch_processor


# Добавьте после инициализаци FastAPI:
processing_tasks = {}


class ProcessingStatus:
    def __init__(self, total_records: int):
        self.total = total_records  # Общее количество записей
        self.processed = 0  # Обработано записей
        self.status = 'processing'  # Статус обработки


@app.get("/", response_class=HTMLResponse, tags=["Pages"])
async def home(request: Request):
    """
    Главная страница с навигацией
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse, tags=["Pages"])
async def upload_form(request: Request):
    """
    Страница для загрузки Excel файла
    """
    logging.debug("Accessing upload form page")
    try:
        return templates.TemplateResponse("upload.html", {"request": request})
    except Exception as e:
        logging.error(f"Error rendering upload template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_file", tags=["File Processing"])
async def process_file(file: UploadFile = File(...)):
    try:
        # Добавляем таймаут для операции чтения файла
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, file.filename)

        # Читаем файл с таймаутом
        async with asyncio.timeout(30):  # 30 секунд таймаут
            with open(input_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

        df = pd.read_excel(input_path, usecols=[0])
        total_rows = len(df)

        processing_id = str(uuid.uuid4())
        processing_tasks[processing_id] = ProcessingStatus(total_rows)

        # Запускаем с обраоткой ошибок
        task = asyncio.create_task(process_file_background(processing_id, input_path, temp_dir))
        task.add_done_callback(lambda t: logging.error(f"Task failed: {t.exception()}") if t.exception() else None)

        return {"processing_id": processing_id}

    except asyncio.TimeoutError:
        logging.error("Timeout while processing file")
        raise HTTPException(status_code=504, detail="Processing timeout")
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_file_background(processing_id: str, input_path: str, temp_dir: str):
    try:
        classifier = get_classifier()
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"classified_{timestamp}_{os.path.basename(input_path)}"
        output_path = results_dir / output_filename

        logging.debug(f"Starting background processing for ID: {processing_id}")

        # Читаем данные
        df_input = pd.read_excel(input_path, usecols=[0])
        minerals = df_input.iloc[:, 0].tolist()
        total = len(minerals)
        logging.debug(f"Total minerals to process: {total}")

        # Обновляем общее количество
        processing_tasks[processing_id].total = total

        # Обрабатываем минералы с обновлением прогресса
        all_results = []
        for i, mineral in enumerate(minerals):
            if pd.notna(mineral):
                # Передаем управление каждые 5 операций
                if i % 5 == 0:
                    await asyncio.sleep(0.01)

                result = classifier.classify_mineral(str(mineral))
                result['original_name'] = mineral
                all_results.append(result)

            processing_tasks[processing_id].processed = i + 1

            # Логируем прогресс
            if (i + 1) % 10 == 0:
                logging.debug(f"Processed {i + 1}/{total} minerals")

        # Сохраняем результаты
        df_results = pd.DataFrame(all_results)
        columns_order = [
            'original_name', 'normalized_name_for_display', 'pi_name_gbz_tbz',
            'pi_group_is_nedra', 'pi_measurement_unit', 'pi_measurement_unit_alternative'
        ]
        df_results = df_results[columns_order]
        df_results.to_excel(output_path, index=False)

        # Обновляем статус и путь к файлу
        processing_tasks[processing_id].status = 'completed'
        processing_tasks[processing_id].output_file = output_path
        logging.debug(f"Processing completed for ID: {processing_id}")

    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
        processing_tasks[processing_id].status = 'error'
    finally:
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/progress/{processing_id}", tags=["File Processing"])
async def get_progress(processing_id: str):
    if processing_id not in processing_tasks:
        logging.error(f"Processing task not found: {processing_id}")
        raise HTTPException(status_code=404, detail="Processing task not found")

    task = processing_tasks[processing_id]
    progress_data = {
        "total": task.total,
        "processed": task.processed,
        "status": task.status
    }
    logging.debug(f"Progress for {processing_id}: {progress_data}")
    return progress_data


@app.get("/download/{processing_id}", tags=["File Processing"])
async def download_result(processing_id: str):
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Processing task not found")

    task = processing_tasks[processing_id]
    if task.status != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed")

    return FileResponse(
        path=str(task.output_file),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=os.path.basename(task.output_file)
    )


def cleanup_old_results(directory: Path, max_age_hours: int = 24):
    """Удаляет файлы результатов старше указного возраста"""
    try:
        current_time = datetime.now()
        for file_path in directory.glob("classified_*"):
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(hours=max_age_hours):
                file_path.unlink()
                logging.debug(f"Deleted old result file: {file_path}")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")


# Функция очистки старых задач
async def cleanup_old_tasks():
    while True:
        try:
            current_time = datetime.now()
            to_remove = []

            for task_id, task in processing_tasks.items():
                if (task.status == 'completed' and
                        hasattr(task, 'completion_time') and
                        current_time - task.completion_time > timedelta(hours=1)):
                    to_remove.append(task_id)

            for task_id in to_remove:
                del processing_tasks[task_id]

            await asyncio.sleep(3600)  # Проверяем раз в час

        except Exception as e:
            logging.error(f"Error in cleanup_old_tasks: {e}")
            await asyncio.sleep(60)


@app.get("/health")
async def health_check():
    try:
        db_healthy = db_manager.check_health()
        
        if db_healthy:
            return {"status": "healthy"}
            
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Database connection failed"}
        )
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )


class ClassificationInput(BaseModel):
    term: str
    normalized_name: str
    gbz_name: str
    group_name: str
    measurement_unit: str
    measurement_unit_alt: str = ""

def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_decorator(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time() + seconds

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time() > func.expiration:
                func.cache_clear()
                func.expiration = time() + func.lifetime
            return func(*args, **kwargs)

        wrapped_func.cache_info = func.cache_info
        wrapped_func.cache_clear = func.cache_clear
        return wrapped_func

    return wrapper_decorator

@app.get("/unclassified", response_class=HTMLResponse, tags=["Interactive Classification"])
async def show_unclassified(request: Request):
    """Показывает ст��аницу с неклассифицированными терминами"""
    try:
        # Используем кастомный декоратор для кэширования
        @timed_lru_cache(seconds=300, maxsize=1)  # кэш на 5 инут
        def get_cached_options():
            batch_processor = get_batch_processor()
            return batch_processor.get_classification_options()

        batch_processor = get_batch_processor()
        terms = batch_processor.get_unclassified_terms()
        options = get_cached_options()
        
        return templates.TemplateResponse(
            "unclassified.html",
            {
                "request": request,
                "terms": terms,
                "options": options
            }
        )
    except Exception as e:
        logging.error(f"Error showing unclassified terms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_classification", tags=["Interactive Classification"])
async def add_classification(classification: ClassificationInput):
    """Добавляет новую классификацию"""
    try:
        # Проверяем обязательные поля
        if not all([
            classification.term,
            classification.normalized_name,
            classification.gbz_name,
            classification.group_name,
            classification.measurement_unit
        ]):
            raise HTTPException(
                status_code=400,
                detail="All fields except measurement_unit_alt are required"
            )

        batch_processor = get_batch_processor()
        success = batch_processor.add_classification(
            classification.term,
            {
                'normalized_name': classification.normalized_name,
                'gbz_name': classification.gbz_name,
                'group_name': classification.group_name,
                'measurement_unit': classification.measurement_unit,
                'measurement_unit_alt': classification.measurement_unit_alt
            }
        )
        
        if success:
            return {"status": "success", "message": "Classification added successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to add classification in batch processor"
            )
    except Exception as e:
        logging.error(f"Error adding classification: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/suggest_classification/{term}", tags=["Interactive Classification"])
async def suggest_classification(term: str):
    """Предлагает классификацию для темина"""
    try:
        batch_processor = get_batch_processor()
        suggestion = batch_processor.learner.suggest_classification(term)
        if suggestion:
            return suggestion
        return {"message": "No suggestion found"}
    except Exception as e:
        logging.error(f"Error suggesting classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove_unclassified/{term}", tags=["Interactive Classification"])
async def remove_unclassified(term: str):
    """Удаляет термин из списка неклассифицированных"""
    try:
        batch_processor = get_batch_processor()
        success = batch_processor.learner.db.remove_unclassified_term(term)
        if success:
            return {"status": "success", "message": "Term removed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to remove term")
    except Exception as e:
        logging.error(f"Error removing unclassified term: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dictionary", response_class=HTMLResponse, tags=["Dictionary Management"])
async def show_dictionary(request: Request):
    """Показывает страницу редактирования справочника"""
    try:
        db_manager = DatabaseManager()
        entries = db_manager.get_dictionary_entries()
        options = db_manager.get_classification_options()
        
        return templates.TemplateResponse(
            "dictionary.html",
            {
                "request": request,
                "entries": entries,
                "options": options
            }
        )
    except Exception as e:
        logging.error(f"Error showing dictionary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_dictionary_entry", tags=["Dictionary Management"])
async def update_dictionary_entry(entry: ClassificationInput):
    """Обновляет запись в справочнике"""
    try:
        db_manager = DatabaseManager()
        success = db_manager.update_dictionary_entry(entry.dict())
        if success:
            return {"status": "success", "message": "Entry updated successfully"}
        raise HTTPException(status_code=400, detail="Failed to update entry")
    except Exception as e:
        logging.error(f"Error updating dictionary entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_dictionary_entry/{unique_key}", tags=["Dictionary Management"])
async def delete_dictionary_entry(unique_key: str):
    """Удаляет запись из справочника"""
    try:
        db_manager = DatabaseManager()
        success = db_manager.delete_dictionary_entry(unique_key)
        if success:
            return {"status": "success", "message": "Entry deleted successfully"}
        raise HTTPException(status_code=400, detail="Failed to delete entry")
    except Exception as e:
        logging.error(f"Error deleting dictionary entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export_dictionary", tags=["Dictionary Management"])
async def export_dictionary():
    """Экспортирует справочник в Excel"""
    try:
        db_manager = DatabaseManager()
        filename = "dictionary_export.xlsx"
        filepath = db_manager.export_dictionary(filename)
        return FileResponse(
            filepath,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename
        )
    except Exception as e:
        logging.error(f"Error exporting dictionary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Глобальная переменная для хранения прогресса
upload_progress = {}

@app.post("/upload", tags=["File Processing"])
async def upload_file(file: UploadFile = File(...)):
    """Загрузка и обработка Excel файла"""
    process_id = str(uuid.uuid4())
    try:
        # Создаем временную директорию для файла
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            
            # Сохраняем загруженный файл
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Читаем файл для подсчета записей
            df = pd.read_excel(str(temp_path))
            total_records = len(df)
            
            # Создаем имя для выходного файла
            output_filename = f"classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            output_path = results_dir / output_filename
            
            # Инициализируем прогресс
            processing_progress[process_id] = {
                'current': 0,
                'total': total_records,
                'status': 'Начало обработки...'
            }
            
            # Обрабатываем файл
            batch_processor = get_batch_processor()
            batch_processor.process_id = process_id  # Передаем ID процесса
            
            result_path = batch_processor.process_excel(str(temp_path), str(output_path))
            
            return {
                "status": "success",
                "message": "File processed successfully",
                "result_file": output_filename,
                "process_id": process_id
            }
            
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/upload/progress/{process_id}")
async def get_progress(process_id: str):
    """Получение прогресса обработки"""
    if process_id in processing_progress:
        progress = processing_progress[process_id]
        percent = int((progress['current'] / progress['total']) * 100) if progress['total'] > 0 else 0
        return {
            "progress": percent,
            "status": f"Обработано {progress['current']} из {progress['total']} записей"
        }
    raise HTTPException(status_code=404, detail="Process not found")

@app.get('/upload/progress')
async def progress(request: Request):
    """Отправка обновлений о прогрессе через SSE"""
    async def event_generator():
        while True:
            try:
                # Получаем последний процесс
                if processing_progress:
                    latest_process = list(processing_progress.values())[-1]
                    current = latest_process['current']
                    total = latest_process['total']
                    
                    # Вычисляем процент
                    progress = int((current / total) * 100) if total > 0 else 0
                    
                    yield {
                        "data": {
                            "progress": progress,
                            "status": f"Обработано {current} из {total} з��писей"
                        }
                    }
                    
                    # Если обработка завершена, завершаем генератор
                    if current >= total:
                        break
                
                # Пауза между обновлениями
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in event generator: {e}")
                yield {
                    "data": {
                        "progress": 0,
                        "status": f"Ошибка: {str(e)}"
                    }
                }
                break

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    try:
        # Проверяем наличие всех необходимых файлов
        required_files = [
            Path("templates/base.html"),
            Path("templates/home.html"),
            Path("templates/upload.html"),
            Path("Справочник_для_редактирования_09.10.2024.xlsx")
        ]

        for file in required_files:
            if not file.exists():
                logging.error(f"Required file not found: {file}")
                raise FileNotFoundError(f"Required file not found: {file}")

        # Изменяем хост на 0.0.0.0 для доступа из контейнера
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, timeout_keep_alive=30)
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise