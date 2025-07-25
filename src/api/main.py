from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import os

app = FastAPI()

# ---- Load model & preprocessor from MLflow ----
MODEL_URI = "models:/best_model/Production"

# Load trained model
model = mlflow.sklearn.load_model(MODEL_URI)

# Load preprocessor (we logged it as an artifact)
preprocessor_path = os.path.join("checkpoints", "preprocessor.pkl")
if not os.path.exists(preprocessor_path):
    # Download from MLflow artifacts if needed
    import requests
    raise FileNotFoundError("Preprocessor file missing. Did you run train.py first?")
preprocessor = joblib.load(preprocessor_path)

# ---- Define API schema ----
class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    CountryCode: str
    txn_hour: int = 12
    txn_day: int = 15
    txn_month: int = 6
    txn_year: int = 2023

class PredictionResponse(BaseModel):
    risk_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    # Convert input to dataframe
    data = pd.DataFrame([features.dict()])

    # Preprocess using same pipeline
    processed_data = preprocessor.transform(data)

    # Predict probability
    prob = model.predict_proba(processed_data)[0][1]
    return PredictionResponse(risk_probability=round(prob, 4))
