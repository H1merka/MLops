import json
import logging
import os

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder

# Параметры подключения к вашему API
CARS_HOST = os.environ.get("CARS_HOST", "carsapi")
CARS_SCHEMA = os.environ.get("CARS_SCHEMA", "http")
CARS_PORT = os.environ.get("CARS_PORT", "8081")

CARS_USER = os.environ["CARS_USER"]
CARS_PASSWORD = os.environ["CARS_PASSWORD"]

# Настройка логгера
logger = logging.getLogger(__name__)


def _get_session():
    """Создаёт сессию для запросов к вашему Car API."""
    session = requests.Session()
    session.auth = (CARS_USER, CARS_PASSWORD)
    base_url = f"{CARS_SCHEMA}://{CARS_HOST}:{CARS_PORT}"
    return session, base_url


def _get_all_cars(batch_size=100):
    """Получает все записи из /cars с пагинацией."""
    session, base_url = _get_session()
    url = f"{base_url}/cars"

    offset = 0
    total = None
    all_cars = []

    while total is None or offset < total:
        params = {"offset": offset, "limit": batch_size}
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        all_cars.extend(data["result"])
        offset += batch_size
        total = data["total"]

        if len(data["result"]) == 0:
            break

    return all_cars


def fetch_cars(**context):
    """Загружает все автомобили и сохраняет в JSON."""
    logger.info("Fetching all cars from the API...")

    cars = _get_all_cars(batch_size=100)
    logger.info(f"Fetched {len(cars)} car records.")

    # Путь для сохранения
    output_path = "/data/cars/cars_full.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved cars to {output_path}")


def clean_cars_data(**context):
    """
    Загружает сырые данные, удаляет дубликаты/пропуски,
    кодирует категориальные поля и сохраняет очищенный JSON.
    """
    input_path = "/data/cars/cars_full.json"
    output_path = "/data/cleaned/cars_cleaned.json"

    logger.info(f"Reading raw cars from {input_path}")
    df = pd.read_json(input_path)

    if df.empty:
        logger.warning("Raw dataset is empty. Nothing to clean.")
        return

    # Удаляем дубликаты и строки с пропусками.
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    categorical_cols = [
        col
        for col in ["Make", "Model", "Style", "Fuel_type", "Transmission"]
        if col in df.columns
    ]

    if categorical_cols:
        encoder = OrdinalEncoder()
        encoded = encoder.fit_transform(df[categorical_cols])
        df[categorical_cols] = encoded

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records")
    logger.info(f"Saved cleaned data to {output_path}")


def analyze_cars(**context):
    """Анализирует данные: например, средняя цена по году."""
    input_path = "/data/cleaned/cars_cleaned.json"
    output_path = "/data/cars/price_by_year.csv"

    logger.info(f"Reading cars from {input_path}")
    df = pd.read_json(input_path)

    if df.empty:
        logger.warning("No car data to analyze.")
        return

    # Пример анализа: средняя цена по году
    summary = df.groupby("Year")["Price_euro"].agg(
        mean_price="mean",
        count="count",
        min_price="min",
        max_price="max"
    ).round(2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path)
    logger.info(f"Analysis saved to {output_path}")


# Определяем DAG
with DAG(
    dag_id="01_cars",
    description="Fetches car data from the custom API and analyzes it.",
    start_date=datetime(2026, 2, 3),  # сегодняшняя дата (ваш контекст)
    schedule="@daily",               # можно оставить daily или сделать @once
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=fetch_cars,
    )

    clean_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=clean_cars_data,
    )

    analyze_task = PythonOperator(
        task_id="analyze_cars",
        python_callable=analyze_cars,
    )

    fetch_task >> clean_task >> analyze_task
