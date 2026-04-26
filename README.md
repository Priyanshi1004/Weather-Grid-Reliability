# Weather-Aware Power Grid Reliability Prediction System

## Architecture

```
weather_grid_project/
├── data/
│   ├── weather_data.csv         # 10,000-row weather dataset
│   └── grid_dataset.csv         # Merged weather + grid simulation
├── models/
│   ├── rf_lolp.pkl              # Random Forest → LOLP
│   ├── rf_eens.pkl              # Random Forest → EENS (log-transformed)
│   ├── scaler.pkl               # MinMaxScaler for feature normalisation
│   └── model_meta.json          # Feature importances + metrics
├── grid_simulator.py            # Physics-based grid simulation engine
├── generate_weather.py          # Seasonal synthetic weather generator
├── feature_engineering.py       # WSI + CWVI computation
├── train_model.py               # Full ML training pipeline
├── backend/
│   └── main.py                  # FastAPI service (POST /predict, GET /feature-importance)
└── frontend/
    └── dashboard.html           # React-style single-page dashboard
```

## Quickstart

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy fastapi uvicorn
```

### 2. Train models
```bash
python train_model.py
```
Generates `data/grid_dataset.csv` (10,000 rows) and saves model artifacts to `models/`.

### 3. Start API
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open frontend
Open `frontend/dashboard.html` in a browser. If the API is running, predictions are live.
If offline, the dashboard runs in **demo mode** with a client-side simulation.

---

## API Reference

### POST /predict
```json
{
  "temperature": 35.0,
  "rainfall": 80.0,
  "humidity": 85.0,
  "wind_speed": 20.0
}
```
Response:
```json
{
  "LOLP": 0.008560,
  "EENS": 0.0486,
  "risk_level": "MODERATE",
  "WSI": 0.671818,
  "CWVI": 0.846421,
  "interpretation": "Minor stress on grid..."
}
```

### GET /feature-importance
Returns feature importances for both models.

### GET /dataset-stats
Returns descriptive statistics from training data.

---

## Key Design Decisions

| Component | Decision | Reason |
|-----------|----------|--------|
| Grid sim  | Demand-driven with AR(1) weather | Physical realism |
| LOLP      | `max(0, (D-C)/C)` | Standard industry metric |
| EENS      | `outage_dur × demand × prob_factor` | Captures both frequency + magnitude |
| EENS model | log1p transform | Right-skewed distribution |
| WSI       | Weighted normalize [0,1] | Comparable composite score |
| CWVI      | WSI + load ratio | Couples weather + grid state |
| RF depth  | 18 | Balanced bias/variance |

## Model Performance

| Metric | LOLP (test) | EENS-log (test) |
|--------|-------------|-----------------|
| R²     | 0.932       | 0.013           |
| MAE    | 0.00228     | 0.00870         |
| RMSE   | 0.00552     | 0.03421         |

LOLP R²=0.932 reflects the deterministic relationship between weather stress and
the demand/capacity gap. EENS is inherently stochastic (outages are rare random
events), so R² is low but the model correctly captures the conditional mean.
