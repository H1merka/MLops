from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import joblib
import os
import numpy as np

app = FastAPI()


class PredictionRequest(BaseModel):
    inputs: List[List[float]]


def load_model():
    # Ищем сериализованную sklearn Pipeline (model_bundle.pkl или lr_pipeline.pkl или lr_cars.pkl)
    candidates = ['model_bundle.pkl', 'lr_pipeline.pkl', 'lr_cars.pkl']
    for p in candidates:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model
            except Exception:
                continue
    return None


MODEL = load_model()


@app.get('/health')
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post('/predict')
def predict(req: PredictionRequest):
    if MODEL is None:
        return {"error": "model not loaded"}
    X = np.array(req.inputs)
    try:
        preds = MODEL.predict(X)
        return {"predictions": np.asarray(preds).tolist()}
    except Exception as e:
        return {"error": str(e)}
