import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

MODEL_PATH = "model.joblib"

def train_model(data_path: str) -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found.")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=["salary", "job_title"])
    X = pd.get_dummies(X)
    y = df["salary"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    
    return {"rmse": rmse}

def predict(input_data: dict) -> float:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet.")
    
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    columns = data["columns"]
    
    # Process input data
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    
    # Ensure all columns match what the model expects
    for col in columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[columns]
    
    prediction = model.predict(df)[0]
    return float(prediction)
