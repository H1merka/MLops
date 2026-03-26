import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from mlflow.models import infer_signature
import mlflow
import joblib
import warnings

# (опционально) убираем спам warning'ов
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(
            random_state=42,
            max_iter=5000,              # увеличили
            early_stopping=True,        # 🔥 ключевое изменение
            n_iter_no_change=5,
            validation_fraction=0.1
        ))
    ])

    return pipeline


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(df_path: str = './df_clear.csv',
          out_bundle: str = 'model_bundle.pkl',
          out_pickle: str = 'lr_pipeline.pkl'):

    df = pd.read_csv(df_path)

    if 'total_revenue' not in df.columns:
        raise ValueError('df_clear.csv must contain total_revenue target')

    categorical_features = ['product_category', 'customer_region', 'payment_method']
    numeric_features = [c for c in df.columns if c not in categorical_features + ['total_revenue']]

    X = df.drop(columns=['total_revenue'])
    y = df['total_revenue'].values

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # 🔥 УМЕНЬШЕННЫЙ GRID (намного быстрее)
    params = {
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'regressor__l1_ratio': [0.01, 0.1],
        'regressor__penalty': ['l2', 'elasticnet'],
        'regressor__loss': ['squared_error', 'huber']
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    mlflow.set_experiment('sales_total_revenue')

    with mlflow.start_run():
        clf = GridSearchCV(
            pipeline,
            params,
            cv=3,
            n_jobs=-1,          # 🔥 использовать все ядра
            verbose=1
        )

        clf.fit(X_train, y_train)

        best = clf.best_estimator_

        y_pred = best.predict(X_val)
        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        mlflow.log_param('best_params', str(clf.best_params_))
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mae', mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)

        mlflow.sklearn.log_model(best, 'model', signature=signature)

        # сохраняем локально
        joblib.dump(best, out_bundle)
        joblib.dump(best, out_pickle)

        model_uri = mlflow.get_artifact_uri('model')
        model_path = model_uri.replace('file://', '')

        with open('best_model.txt', 'w') as bf:
            bf.write(model_path)

        print(f"Model saved locally to {out_bundle}")
        print(f"MLflow artifact path: {model_path}")


if __name__ == '__main__':
    train()
