# MLOps Monorepo

Комплексный учебный проект, демонстрирующий end-to-end практики MLOps: от сбора данных до production-ready inference сервисов.

## Обзор

Монорепозиторий содержит 5 взаимосвязанных проектов, охватывающих полный цикл разработки ML-систем:
- REST API для работы с данными
- Оркестрация ML-пайплайнов (Apache Airflow)
- Эксперименты и трекинг моделей (MLflow)
- Production inference сервисы (FastAPI)
- Контейнеризация и оркестрация (Docker Compose)

**Технологический стек**: Python, Flask, FastAPI, Apache Airflow, MLflow, Docker, Redis, scikit-learn, pandas

---

## Проекты

### 1. 🚗 cars/
**Flask API + Apache Airflow Data Pipeline**

REST API для работы с датасетом автомобилей с интеграцией в Airflow.

**Ключевые возможности:**
- HTTP Basic Authentication (airflow:airflow)
- Фильтрация по параметрам: год, цена, марка, модель, тип топлива, трансмиссия
- Пагинация (offset/limit)
- Airflow DAGs:
  - `01_cars` - ежедневная загрузка и агрегация данных по годам
  - `02_hook` - использование кастомного `CarsHook` для интеграции
- Сохранение результатов в JSON/CSV

**Структура:**
```
cars/
├── cars-api/          # Flask приложение
│   ├── app.py         # REST API endpoints
│   ├── Dockerfile     # Контейнеризация
│   └── requirements.txt
└── dags/              # Airflow DAGs
    ├── 01_python.py   # Базовый Python Operator
    ├── 02_hook.py     # Кастомный Hook
    └── hooks.py       # CarsHook реализация
```

**Endpoints:**
- `GET /` - health check
- `GET /cars` - получение автомобилей с фильтрацией

**Пример запроса:**
```bash
curl -u airflow:airflow "http://localhost:8081/cars?min_year=2015&max_price=20000&offset=0&limit=50"
```

---

### 2. 🏗️ cars compose/
**Микросервисная архитектура с Redis кэшированием**

Production-ready вариант cars API с кэшированием и docker-compose оркестрацией.

**Ключевые возможности:**
- Redis кэширование запросов (TTL: 60 сек)
- Флаг `from_cache` в ответах
- Healthcheck endpoints
- Resource limits (CPU: 0.5, Memory: 512M)
- Auto-restart политика

**Структура:**
```
cars compose/
├── app/
│   └── main.py        # Flask + Redis integration
├── docker-compose.yaml # Оркестрация сервисов
├── Dockerfile
└── readme.md          # Инструкции по запуску
```

**Архитектура:**
```
[Client] → [car-api:8081] ← → [Redis:6379]
```

**Запуск:**
```bash
cd "MLOPS/cars compose"
docker compose up -d --build
docker compose logs -f
```

---

### 3. 🔄 lab airflow/
**Автоматизированный ML Pipeline с Apache Airflow**

Полный пайплайн машинного обучения: загрузка → очистка → обучение модели.

**Ключевые возможности:**
- Scheduled execution (каждые 5 минут)
- Data cleaning pipeline:
  - Удаление аномалий (Distance > 1e6, Price < 101, Year < 1971)
  - Фильтрация некорректных значений
  - Ordinal encoding категориальных признаков
- SGDRegressor для предсказания цен
- Concurrency control (max_active_runs=1)

**Структура:**
```
lab airflow/
├── airflow_pipe.py    # DAG определение
└── train_model.py     # Обучение модели
```

**DAG Flow:**
```
download_cars → clear_cars → train_cars
```

**Расписание:** Каждые 5 минут
**Catchup:** Отключен

---

### 4. 🔬 lab3/
**ML Эксперимент с MLflow Tracking**

Standalone эксперимент для подбора гиперпараметров и трекинга с MLflow.

**Ключевые возможности:**
- GridSearchCV по 5 параметрам SGDRegressor:
  - alpha: [0.0001, 0.001, 0.01, 0.05, 0.1]
  - l1_ratio: [0.001, 0.05, 0.01, 0.2]
  - penalty: [l1, l2, elasticnet]
  - loss: [squared_error, huber, epsilon_insensitive]
  - fit_intercept: [True, False]
- MLflow tracking:
  - Логирование параметров (alpha, l1_ratio, penalty, eta0)
  - Метрики: RMSE, MAE, R²
  - Артефакты: sklearn модель + signature
- PowerTransformer для нормализации цены
- Автоматический выбор лучшей модели по R²

**Структура:**
```
lab3/
├── download.py        # Загрузка и очистка данных
├── train_model.py     # GridSearch + MLflow
└── requirements.txt
```

**Запуск:**
```bash
cd MLOPS/lab3
python download.py
python train_model.py
mlflow ui  # Просмотр экспериментов
```

**Метрики:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

### 5. ⚡ lab_fastapi/
**Production ML Inference сервис (FastAPI)**

REST API для real-time предсказания цен автомобилей на основе обученной модели.

**Ключевые возможности:**
- Feature engineering:
  - `distance_by_year` - пробег на год эксплуатации
  - `age` - возраст автомобиля
  - `eng_cap_diff` - отклонение объема двигателя от среднего по стилю
  - `eng_cap_diff_max` - отклонение от максимума по стилю
- Загрузка pre-trained модели (cars.joblib + power.joblib)
- Ordinal encoding категориальных переменных
- Inverse transform для возврата цены в евро
- Логирование (Python logging)
- Dockerized deployment

**Структура:**
```
lab_fastapi/
├── main.py            # FastAPI приложение
├── Dockerfile
├── requirements.txt
└── readme.txt         # Docker инструкции
```

**API Endpoint:**
```
POST /predict
```

**Пример запроса:**
```json
{
  "make": "BMW",
  "model": "X5",
  "year": 2020,
  "style": "SUV",
  "distance": 50000,
  "engine_capacity": 3000,
  "fuel_type": "Diesel",
  "transmission": "Automatic"
}
```

**Пример ответа:**
```json
{
  "predicted_price": 42350.87
}
```

**Запуск:**
```bash
cd MLOPS/lab_fastapi
docker build -t fast_api:latest .
docker run -d -p 9005:8005 fast_api
# Сервис доступен по http://localhost:9005/docs
```

---

## Быстрый старт

### Требования
- Python 3.9+
- Docker & Docker Compose
- Apache Airflow (для проектов cars, lab airflow)
- MLflow (для проекта lab3)

### Установка зависимостей

Для каждого проекта:
```bash
cd MLOPS/<project_name>
pip install -r requirements.txt
```

### Запуск проектов

**Flask API (cars):**
```bash
cd MLOPS/cars/cars-api
python app.py
# API доступен по http://localhost:8081
```

**Docker Compose (cars compose):**
```bash
cd "MLOPS/cars compose"
docker compose up -d --build
```

**Airflow Pipeline (lab airflow):**
```bash
# Настроить Airflow (вне скоупа)
# Скопировать airflow_pipe.py в $AIRFLOW_HOME/dags/
airflow dags trigger train_pipe
```

**MLflow Эксперимент (lab3):**
```bash
cd MLOPS/lab3
python download.py && python train_model.py
mlflow ui --port 5000
```

**FastAPI Inference (lab_fastapi):**
```bash
cd MLOPS/lab_fastapi
docker build -t fast_api:latest .
docker run -d -p 9005:8005 fast_api
# Swagger UI: http://localhost:9005/docs
```

---

## Общие практики

### Обработка данных
- Ordinal encoding для категориальных переменных
- Power transformation для нормализации цены
- Фильтрация аномалий на основе:
  - Здравого смысла (цена < 100€, объем двигателя < 200 см³)
  - Статистического анализа (Distance > 1e6, Price > 1e5)

### Модели
- **Алгоритм**: SGDRegressor (Stochastic Gradient Descent)
- **Функция потерь**: squared_error, huber, epsilon_insensitive
- **Регуляризация**: L1, L2, ElasticNet
- **Feature engineering**: возраст, пробег на год, отклонения объема двигателя

### DevOps
- Контейнеризация всех сервисов (Docker)
- Оркестрация микросервисов (Docker Compose)
- Healthchecks и resource limits
- Environment variables для конфигурации
- HTTP Basic Auth для безопасности API
