"""
backend/main.py
FastAPI service for Weather-Aware Grid Reliability Prediction.
Endpoints:
  POST /predict          → LOLP, EENS, risk_level
  GET  /feature-importance → feature importances for LOLP & EENS
  GET  /health           → service health
"""

import json
import logging
import os
import pickle
import sys
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── paths (resolved relative to this file) ───────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "..", "models")

# ── load models ───────────────────────────────────────────────
def _load(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    rf_lolp  = _load("rf_lolp.pkl")
    rf_eens  = _load("rf_eens.pkl")
    scaler   = _load("scaler.pkl")
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        MODEL_META = json.load(f)
    log.info("Models loaded successfully.")
except FileNotFoundError as e:
    log.error(str(e))
    log.error("Run train_model.py first to generate model artifacts.")
    sys.exit(1)

FEATURES = MODEL_META["features"]

# ── feature engineering (must mirror feature_engineering.py) ──
from sklearn.preprocessing import MinMaxScaler as _SKScaler

# Reference ranges from training data (approximate) for normalising inputs
_REF_MIN = np.array([-5.0,  0.0, 20.0,  0.0])   # temp, rain, hum, wind
_REF_MAX = np.array([50.0, 300.0, 100.0, 50.0])

def _compute_wsi_cwvi(temp, rain, hum, wind, lolp_proxy=None):
    """Mirror of feature_engineering.add_features for single-row inference."""
    arr = np.array([[temp, rain, hum, wind]], dtype=float)
    normed = (arr - _REF_MIN) / (_REF_MAX - _REF_MIN + 1e-9)
    normed = normed.clip(0, 1)
    t_n, r_n, h_n, w_n = normed[0]
    wsi  = 0.35*t_n + 0.25*r_n + 0.25*w_n + 0.15*h_n

    # Without real demand/capacity at inference time we use a proxy load ratio
    # based on temperature (higher temp → higher demand relative to capacity)
    demand_proxy = 1.0 + max(0, temp - 22) * 0.008 + max(0, 15 - temp) * 0.005
    cap_proxy    = demand_proxy * 1.05          # mirrors CAPACITY_MARGIN
    lr_proxy     = demand_proxy / cap_proxy
    lr_normed    = (lr_proxy - 0.9) / 0.15     # approx normalisation
    lr_normed    = float(np.clip(lr_normed, 0, 1))

    cwvi = wsi + lr_normed * 0.5
    return round(wsi, 6), round(cwvi, 6)


# ── risk classification ───────────────────────────────────────
def _risk_level(lolp: float, eens: float) -> str:
    if lolp < 0.01 and eens < 0.05:
        return "LOW"
    elif lolp < 0.05 or eens < 0.5:
        return "MODERATE"
    elif lolp < 0.15 or eens < 2.0:
        return "HIGH"
    else:
        return "CRITICAL"


# ── Pydantic models ───────────────────────────────────────────
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-10, le=55, description="°C")
    rainfall:    float = Field(..., ge=0,   le=350, description="mm/day")
    humidity:    float = Field(..., ge=0,   le=100, description="%")
    wind_speed:  float = Field(..., ge=0,   le=60,  description="m/s")

class PredictionOutput(BaseModel):
    temperature: float
    rainfall:    float
    humidity:    float
    wind_speed:  float
    WSI:         float
    CWVI:        float
    LOLP:        float
    EENS:        float
    risk_level:  Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    interpretation: str

class FeatureImportanceOutput(BaseModel):
    LOLP: dict
    EENS: dict


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Weather-Aware Grid Reliability API",
    description="Predicts Loss of Load Probability (LOLP) and Expected Energy Not Served (EENS) from weather inputs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_features": FEATURES}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: WeatherInput):
    """Predict LOLP and EENS for given weather conditions."""
    wsi, cwvi = _compute_wsi_cwvi(
        data.temperature, data.rainfall, data.humidity, data.wind_speed
    )

    X_raw = np.array([[
        data.temperature, data.rainfall, data.humidity, data.wind_speed,
        wsi, cwvi
    ]])
    X = scaler.transform(X_raw)

    lolp      = float(np.clip(rf_lolp.predict(X)[0], 0, 1))
    eens_log  = float(rf_eens.predict(X)[0])
    eens      = float(np.expm1(max(eens_log, 0)))   # inverse log1p

    risk = _risk_level(lolp, eens)

    interpretations = {
        "LOW":      "Grid is stable. No significant reliability concerns under current weather.",
        "MODERATE": "Minor stress on grid. Monitor conditions; demand response may be warranted.",
        "HIGH":     "Elevated risk of supply shortfall. Activate reserves and alert operators.",
        "CRITICAL": "Severe grid stress. Immediate load-shedding or emergency response required.",
    }

    log.info(f"Predict → LOLP={lolp:.5f}  EENS={eens:.4f}  Risk={risk}")

    return PredictionOutput(
        temperature=data.temperature,
        rainfall=data.rainfall,
        humidity=data.humidity,
        wind_speed=data.wind_speed,
        WSI=wsi,
        CWVI=cwvi,
        LOLP=round(lolp, 6),
        EENS=round(eens, 4),
        risk_level=risk,
        interpretation=interpretations[risk],
    )


@app.get("/feature-importance", response_model=FeatureImportanceOutput)
def feature_importance():
    """Return feature importances for LOLP and EENS models."""
    return FeatureImportanceOutput(
        LOLP=MODEL_META["feature_importance"]["LOLP"],
        EENS=MODEL_META["feature_importance"]["EENS"],
    )


@app.get("/dataset-stats")
def dataset_stats():
    """Return summary statistics from the training dataset."""
    import pandas as pd
    csv = os.path.join(BASE, "..", "data", "grid_dataset.csv")
    if not os.path.exists(csv):
        raise HTTPException(404, "Dataset not found. Run train_model.py first.")
    df = pd.read_csv(csv)
    cols = ["temperature", "rainfall", "humidity", "wind_speed", "LOLP", "EENS", "WSI", "CWVI"]
    stats = df[cols].describe().round(4).to_dict()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
