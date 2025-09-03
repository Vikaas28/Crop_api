# ===========================================
# Crop Recommendation API (FastAPI)
# ===========================================
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load artifacts
artifacts = joblib.load("crop_recommendation_xgb_v3.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
crop_encoder = artifacts["crop_encoder"]

# FastAPI app
app = FastAPI(
    title="🌾 Crop Recommendation API",
    description="Send soil & weather data → Get Top-3 crop recommendations",
    version="1.0"
)

# ----------------------------
# Request schema
# ----------------------------
class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# ----------------------------
# Helper: Recommend Crops
# ----------------------------
def recommend_crops(features, model, scaler, encoder, top_k=3):
    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    probs = model.predict_proba(arr_scaled)[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    top_crops = encoder.inverse_transform(top_idx)
    top_probs = probs[top_idx]

    return [{"crop": crop, "probability": float(round(prob * 100, 2))}
            for crop, prob in zip(top_crops, top_probs)]

# ----------------------------
# API Endpoints
# ----------------------------

@app.get("/")
def home():
    return {"message": "🌾 Crop Recommendation API is running!"}

@app.post("/predict")
def predict(request: CropRequest):
    features = [
        request.N, request.P, request.K,
        request.temperature, request.humidity,
        request.ph, request.rainfall
    ]
    recommendations = recommend_crops(features, model, scaler, crop_encoder)
    return {"recommendations": recommendations}