from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import os
from mineral_classifier import MineralClassifier, BatchProcessor
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import uuid
from functools import lru_cache

logging.basicConfig(level=logging.DEBUG)

# Создаем директорию для статических файлов, если её нет
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Создаем директорию для результатов
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Создаем приложение FastAPI
app = FastAPI(
    title="Mineral Classifier API",
    description="API для клссификации полезных ископаемых",
    version="1.0.0"
)


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
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.status = 'processing'
        self.output_file = None


class MineralInput(BaseModel):
    name: str


class MineralBatchInput(BaseModel):
    minerals: List[str]


class MineralRequest(BaseModel):
    mineral_name: str


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

        # Запускаем с обработкой ошибок
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


@app.post("/classify", tags=["Classification"])
async def classify_mineral(request: MineralRequest):
    try:
        classifier = get_classifier()
        logging.debug(f"Received request for mineral: {request.mineral_name}")
        result = classifier.classify_mineral(request.mineral_name)
        result = {k: str(v) for k, v in result.items()}
        logging.debug(f"Classification result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify_batch", tags=["Classification"])
async def classify_batch(minerals: MineralBatchInput):
    """
    Классифицирует список минералов
    
    - **minerals**: Список названий минералов для классификации
    """
    logging.debug(f"Received batch request for minerals: {minerals.minerals}")
    results = [classifier.classify_mineral(mineral) for mineral in minerals.minerals]
    return results


@app.post("/classify_form", response_class=HTMLResponse, tags=["Classification"])
async def classify_form(request: Request, mineral_name: str = Form(...)):
    """
    Классифицирует минерал через веб-форму
    """
    try:
        result = classifier.classify_mineral(mineral_name)
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": result,
                "mineral_name": mineral_name
            }
        )
    except Exception as e:
        logging.error(f"Error classifying mineral: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.on_event("startup")
async def startup_event():
    """Выполняется при запуске приложения"""
    cleanup_old_results(results_dir)
    asyncio.create_task(cleanup_old_tasks())

    # Инициализируем классификатор в отдельной задаче
    def init_classifier():
        return get_classifier()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_classifier)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Проверка здоровья сервиса
    """
    try:
        # Проверяем подключение к Redis
        classifier = get_classifier()
        classifier.db.redis_client.ping()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "classifier": "initialized"
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


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
