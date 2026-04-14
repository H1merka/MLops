from fastapi import FastAPI
from app.api import router

app = FastAPI(title="MLOps FastAPI Service", description="API for ML Model Inference & Training")

app.include_router(router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "MLOps API is running"}
