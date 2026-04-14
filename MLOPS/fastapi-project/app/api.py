from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from app.ml_model import predict, train_model
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    experience_years: float
    skills_count: int
    certifications: int
    education_level: str
    industry: str
    company_size: str
    location: str
    remote_work: str

class PredictResponse(BaseModel):
    predicted_salary: float

@router.post("/predict", response_model=PredictResponse)
def handle_predict(request: PredictRequest):
    try:
        prediction = predict(request.dict())
        return PredictResponse(predicted_salary=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
def handle_train():
    try:
        metrics = train_model("data/job_salary_prediction_dataset.csv")
        return {"status": "trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
