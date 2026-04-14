import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "MLOps API is running"}
    
def test_predict_without_model():
    response = client.post("/predict", json={
        "experience_years": 5,
        "skills_count": 5,
        "certifications": 2,
        "education_level": "Bachelor",
        "industry": "IT",
        "company_size": "Medium",
        "location": "NY",
        "remote_work": "Yes"
    })
    
    assert response.status_code == 500
    assert "detail" in response.json()
