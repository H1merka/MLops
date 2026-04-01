# MLOps Monorepo

Комплексный учебный проект, демонстрирующий end-to-end практики MLOps

## Обзор

Монорепозиторий содержит 3 взаимосвязанных проекта:
- REST API для работы с данными
- Оркестрация ML-пайплайнов (Apache Airflow)
- Эксперименты и трекинг моделей (MLflow)
- Production inference сервисы (FastAPI)
- Контейнеризация и оркестрация (Docker Compose)

**Технологический стек**: Python, Flask, FastAPI, Apache Airflow, MLflow, Docker, Redis, scikit-learn, pandas

---

## Проекты

### 1. api-airflow-project/
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
api-airflow-project/
├── cars-api/          # Flask приложение
│   ├── app.py         # REST API endpoints
│   ├── cars.csv       # Исходный датасет
│   ├── Dockerfile     # Контейнеризация
│   └── requirements_api.txt
├── config/            # Конфигурация Airflow
│   └── airflow.cfg
└── dags/              # Airflow DAGs
    ├── 01_python.py   # Базовый Python Operator
    ├── 02_hook.py     # Кастомный Hook
    └── hooks.py       # CarsHook реализация
```

**Endpoints:**
- `GET /` - health check
- `GET /cars` - получение автомобилей с фильтрацией

---

### 2. airflow-project/
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

### 3. mlflow-project/
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
mlflow-project/
├── data/              # Датасеты
│   └── student_mental_health_burnout.csv
├── mlruns/            # Эксперименты MLflow
├── compare_runs.py    # Сравнение экспериментов
├── train_mlflow.py    # Обучение модели
└── requirements.txt
```

**Запуск:**
```bash
cd MLOPS/mlflow-project
python compare_runs.py
python train_mlflow.py
mlflow ui  # Просмотр экспериментов
```

**Метрики:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

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
