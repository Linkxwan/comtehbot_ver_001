import os
import base64
import uuid
import asyncio
import json
import logging

import aiohttp
import uvicorn
import edge_tts
import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from data.data_ml import data_set
from data.data_gpt import texts
import g4f

import requests
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Настройка логгера
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация переменных и настроек
MAX_CACHE_SIZE = 1000  # Максимальный размер кэша
MAX_QUESTION_HISTORY_SIZE = 10  # Максимальный размер списка
prefix_gpt = "voice_data_gpt"  # Префикс для путей к файлам для GPT
prefix_ml = "voice_data"  # Префикс для путей к файлам для ML
my_threshold = 0.7  # Пороговое значение для сравнения сходства векторов

# Создаем пустой словарь для кэширования результатов
cache = {}
# Создаем пустой список для хранения значений переменной question
question_history = []

# Инициализация FastAPI приложения
app = FastAPI()


# Логируем успешную инициализацию переменных и настроек
logging.info("Успешно инициализированы переменные и настройки.")


# Создание объекта TfidfVectorizer для векторизации ключей из data_set
tfidf_vectorizer = TfidfVectorizer()
# Преобразование ключей из data_set в векторную форму
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(list(data_set.keys()))
    logging.info("Успешно выполнено векторизация ключей из data_set")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации ключей из data_set: {e}")

# Создание объекта TfidfVectorizer для векторизации текстов из texts
texts_vectorizer = TfidfVectorizer()
# Преобразование текстов из texts в векторную форму
try:
    texts_vectors = texts_vectorizer.fit_transform(texts)
    logging.info("Успешно выполнено векторизация текстов из texts")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации текстов из texts: {e}")


def read_json(file_path):
    """
    Считывает JSON-файл и возвращает его содержимое в виде словаря.

    Параметры:
    - file_path (str): Путь к JSON-файлу.

    Возвращает:
    dict: Содержимое JSON-файла в виде словаря.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        logging.info(f"JSON-файл {file_path} успешно прочитан.")
        return data
    except FileNotFoundError:
        logging.error(f"JSON-файл {file_path} не найден.")
        return None
    except Exception as e:
        logging.error(f"Ошибка при чтении JSON-файла {file_path}: {e}")
        return None

# Считываем базу данных
data_ml = read_json("data\data_ml.json")
data_gpt = read_json("data\data_gpt.json")


async def process_chunk(voice, text_chunk, output_file):
    communicate = edge_tts.Communicate(text_chunk, voice)
    await communicate.save(output_file)


async def synthesis(data, prefix=""):
    voice = 'ru-RU-SvetlanaNeural'  # Установите желаемый голос
    unique_id = uuid.uuid4()
    created_files = []  # Список для хранения созданных файлов

    # Создаем и запускаем асинхронные потоки для каждого фрагмента текста
    output_file = os.path.join(prefix, f"data_{unique_id}_0.mp3")  # Уникальный путь к файлу
    task = asyncio.create_task(process_chunk(voice, data, output_file))

    # Дождемся завершения всех задач
    await asyncio.gather(task)

    # Соберем имена созданных файлов
    output_file = os.path.join(prefix, f"data_{unique_id}_0.mp3")
    created_files.append(output_file)

    return created_files, unique_id


async def vectorize(question, tfidf_vectorizer, tfidf_matrix):
    """
    Векторизация вопроса и сравнение его с данными из базы.

    Параметры:
    - question (str): Вопрос, который необходимо векторизовать и сравнить.
    - tfidf_vectorizer (TfidfVectorizer): Объект TfidfVectorizer для векторизации текста.
    - tfidf_matrix (sparse matrix): Матрица TF-IDF, представляющая данные из базы.

    Возвращает:
    - most_similar_index (int): Индекс наиболее похожего вопроса из базы.
    - similarity (float): Процент сходства между вопросом и наиболее похожим вопросом из базы.
    """
    # Векторизация вопроса
    question_vector = tfidf_vectorizer.transform([question])

    # Вычисление косинусного сходства между вопросом и данными из базы
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Находим индекс наиболее похожего вопроса
    most_similar_index = cosine_similarities.argmax()
    
    # Возвращаем индекс и процент сходства
    return most_similar_index, cosine_similarities[most_similar_index]


# Функция для дополнения файла новыми значениями переменной question
async def append_question_history_to_file():
    async with aiofiles.open("question_history.txt", "a", encoding="utf-8") as file:
        for question in question_history:
            await file.write(question + "\n")


async def read_files(files, prefix=""):
    """
    Асинхронно читает содержимое файлов и кодирует его в формат base64.

    Параметры:
    - files (list): Список имен файлов, которые нужно прочитать.
    - prefix (str): Префикс для путей к файлам (по умолчанию пустая строка).

    Возвращает:
    - file_contents (dict): Словарь, содержащий содержимое файлов в формате base64.
    """
    # Словарь для хранения содержимого файлов в формате base64
    file_contents = {}
    for file in files:
        file_path = os.path.join(prefix, file)
        async with aiofiles.open(file_path, mode='rb') as f:
            content = await f.read()
            file_contents[file] = base64.b64encode(content).decode("utf-8")
    return file_contents


async def remove_files(files, prefix=""):
    """
    Асинхронно удаляет файлы из указанного каталога.

    Параметры:
    - files (list): Список имен файлов, которые нужно удалить.
    - prefix (str): Префикс для путей к файлам (по умолчанию пустая строка).

    Возвращает:
    Ничего.
    """
    try:
        # Удаление каждого файла
        for file in files:
            file_path = os.path.join(prefix, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Файл {file_path} удален успешно.")
            else:
                logging.warning(f"Файл {file_path} не существует.")
    except Exception as e:
        # Логирование ошибки, если что-то пошло не так
        logging.error(f"Ошибка при удалении файлов: {e}")


# Функция для логирования остановки сервера
@app.on_event("shutdown")
async def shutdown_event():
    await append_question_history_to_file()
    logging.info("Сервер остановлен.")


@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        # Здесь вы должны указать путь к файлу на сервере
        file_path = f"{prefix_ml}/{filename}"
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Логируем успешную загрузку файла
        logging.info(f"Файл {filename} был успешно загружен")

        return FileResponse(file_path)
    
    except Exception as e:
        # Обработка ошибок
        logging.error(f"Произошла ошибка при загрузке файла {filename}: {e}")
        return {"error": str(e)}


@app.get("/download")
async def download_file(response: Response):
    file_path = "question_history.txt"  # Укажите путь к файлу
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="question_history.txt", media_type='application/octet-stream')
    else:
        response.status_code = 404
        return {"error": "File not found"}
    


@app.get("/get_response")
async def get_response(question: str):
    try:
        start = time.time()
        # Добавляем значение question в историю
        question_history.append(question)

        # Если размер списка достиг максимального значения, записываем его в файл и очищаем
        if len(question_history) >= MAX_QUESTION_HISTORY_SIZE:
            await append_question_history_to_file()
            question_history.clear()

        print(len(question_history))

        if question in cache:
            most_similar_index, threshold = cache[question]
            # Логируем успешное получение ответа из кэша
            logging.info(f"Успешно получен ответ из кэша для вопроса: {question}")
        else:
            # выводим значение ключа с наибольшим сходством и сохраняем результат в кэше
            most_similar_index, threshold = await vectorize(question, tfidf_vectorizer, tfidf_matrix)
            cache[question] = (most_similar_index, threshold)

            # Удаляем старые записи, если кэш превышает максимальный размер
            if len(cache) > MAX_CACHE_SIZE:
                # Удаляем самую старую запись из кэша
                oldest_question = min(cache, key=cache.get)
                del cache[oldest_question]

        if threshold >= my_threshold:
            # Если порог сходства превышен, получаем ответ из базы данных
            most_similar_key = list(data_set.keys())[most_similar_index]
            response = data_set[most_similar_key]
            # Проверяем, существует ли файл для этого ответа в папке voice_data
            json_item = data_ml[most_similar_key]
            logging.info(f"Успешно получен ответ для вопроса: {question}")

            # Возвращаем ответ с данными и метаданными
            return {
                    "question": question,
                    "response": response,
                    "time": time.time() - start,
                    "file_path": json_item["file_path"], 
                    "file_content": json_item["file_content"],
                }
        
        else:
            logging.info(f"Не получено ответа для вопроса: {question}")
            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": data_gpt["text"],
                "time": time.time() - start,
                "files": data_gpt["file_path"],
                "files_content": data_gpt["file_content"]
            }
        
    except Exception as e:
        # Логируем ошибку
        logging.error(f"Произошла ошибка при обработке вопроса: {question}. Ошибка: {e}")
        return {"error": str(e)}
    

import httpx
from fastapi.staticfiles import StaticFiles
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Словарь для хранения истории чата каждого пользователя
chat_history_by_user = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Получаем идентификатор пользователя из сессии или другим способом
    user_id = request.client.host  # Пример, можно использовать IP-адрес как идентификатор пользователя
    # Получаем историю чата для данного пользователя или создаем новую, если история отсутствует
    chat_history = chat_history_by_user.get(user_id, [])
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})


@app.post("/api", response_class=HTMLResponse)
async def get_answer(request: Request, question: str = Form(...)):
    user_id = request.client.host
    if not question:  # Проверяем, что вопрос не пустой
        return RedirectResponse("/", status_code=303)

    chat_history = chat_history_by_user.get(user_id, [])
    
    async with httpx.AsyncClient() as client:
        response = await client.get("comtehbot:8000/get_response", params={"question": question}, timeout=20)

        # Проверяем успешность запроса
        if response.status_code == 200:
            # Получаем данные из ответа
            response_data = response.json()
            chat_history.append({"user": "you", "message": question})
            chat_history.append({"user": "bot", "message": response_data["response"]})
            chat_history_by_user[user_id] = chat_history
        else:
            # Если запрос неуспешен, обрабатываем ошибку
            print(f"Ошибка: {response.status_code}")
    return RedirectResponse("/", status_code=303)


@app.get("/clear_history")
async def clear_history(request: Request):
    user_id = request.client.host
    # Очистка истории чата для данного пользователя
    chat_history_by_user[user_id] = []
    return RedirectResponse("/", status_code=303)


def main():
    pass

if __name__ == "__main__":
    main()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)