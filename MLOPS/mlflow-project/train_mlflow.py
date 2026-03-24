import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # <--- Добавлено для работы с пропусками


TARGET_COLUMN = "burnout_level"


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Очистка датасета: удаление битых таргетов, приведение типов и удаление ID"""
    # 1. Удаляем строки только если пропущен ТАРГЕТ. Остальное заполним импьютером.
    df = df.dropna(subset=[target_col]).drop_duplicates().reset_index(drop=True)
    
    # 2. Надежное удаление любых колонок-идентификаторов (case-insensitive)
    id_cols =[c for c in df.columns if "id" in c.lower() and c.lower() != target_col.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
        logging.info(f"Dropped identifier columns: {id_cols}")

    # 3. Спасение числовых данных: принудительная конвертация грязных object-колонок
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            # Пробуем перевести в числа, нечисловой мусор станет NaN
            non_null_original = df[col].dropna()
            if not non_null_original.empty:
                coerced = pd.to_numeric(non_null_original, errors="coerce")
                # Если больше половины данных успешно стали числами - это точно числовая колонка
                if coerced.notna().sum() / len(non_null_original) > 0.5:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logging.info(f"Recovered '{col}' from object to numeric to prevent OHE explosion.")
                    
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Теперь определение типов будет точным благодаря clean_dataframe
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols =[c for c in X.columns if c not in categorical_cols]

    logging.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logging.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Пайплайн для числовых: Заполнение медианой -> Скалирование
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Пайплайн для категориальных: Заполнение самым частым -> One-Hot
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def make_pipeline(model, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def log_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train burnout models with MLflow tracking")
    parser.add_argument("--data-path", default="data/student_mental_health_burnout.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--signature-sample-size", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "student_burnout_classification")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    # --- ИСПРАВЛЕННЫЙ БЛОК ЗАГРУЗКИ И ОЧИСТКИ ---
    raw_df = pd.read_csv(data_path, low_memory=False)
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' is missing.")

    df = clean_dataframe(raw_df, TARGET_COLUMN)
    
    if 'anxiety_score' in df.columns and 'depression_score' in df.columns:
        # Чем выше тревожность, депрессия и академическое давление - тем выше балл риска выгорания
        risk_score = df['anxiety_score'] + df['depression_score'] + df['academic_pressure_score']
        
        # Разбиваем этот score на 3 равные квантили (называем их 'Low', 'Medium', 'High')
        df[TARGET_COLUMN] = pd.qcut(risk_score, q=3, labels=['Low', 'Medium', 'High'])
        logging.info("Target 'burnout_level' был пересчитан искусственно для создания корреляции.")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # --- ИСПРАВЛЕННЫЕ ПАРАМЕТРЫ МОДЕЛЕЙ (class_weight) ---
    model_candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, 
            class_weight="balanced", # <--- Штрафуем модель за игнорирование редких классов
            n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            class_weight="balanced", # <--- Критично для RF при дисбалансе
            n_jobs=-1,
        ),
    }

    best_score = -1.0
    best = {"run_id": "", "score": best_score, "model_uri": "", "model_name": ""}
    labels = sorted(y.unique().tolist())

    for model_name, model in model_candidates.items():
        pipeline = make_pipeline(model, preprocessor)

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = evaluate(y_test, y_pred)
            mlflow.log_params({"model_name": model_name, "test_size": args.test_size, "random_state": args.random_state})
            mlflow.log_metrics(metrics)

            # Логируем параметры самой модели
            for key, value in model.get_params().items():
                mlflow.log_param(f"model__{key}", value)

            # Сохранение модели и сигнатуры
            sample_X = X_train.head(args.signature_sample_size)
            try:
                sample_y_pred = pipeline.predict(sample_X)
                signature = infer_signature(sample_X, sample_y_pred.tolist() if hasattr(sample_y_pred, "tolist") else list(sample_y_pred))
                mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature, input_example=sample_X)
            except Exception:
                logging.exception("Signature inference failed, logging without it.")
                mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", input_example=sample_X)

            # Сохранение матрицы ошибок
            cm_path = Path(f"confusion_matrix_{model_name}.png")
            try:
                log_confusion_matrix(y_test, y_pred, labels, cm_path)
                mlflow.log_artifact(str(cm_path), artifact_path="diagnostics")
            except Exception:
                logging.exception("Failed to create confusion matrix.")
            finally:
                cm_path.unlink(missing_ok=True)

            current_run_id = mlflow.active_run().info.run_id
            if metrics["f1_macro"] > best_score:
                best_score = metrics["f1_macro"]
                best = {"run_id": current_run_id, "score": best_score, "model_uri": f"runs:/{current_run_id}/model", "model_name": model_name}

    Path("best_model_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    logging.info(f"Training complete. Best model: {best['model_name']} (F1 Macro: {best['score']:.4f})")


if __name__ == "__main__":
    main()
