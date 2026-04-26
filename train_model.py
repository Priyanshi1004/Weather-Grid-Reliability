"""
train_model.py
Trains Random Forest models for LOLP and EENS prediction.
Produces: models/rf_lolp.pkl, models/rf_eens.pkl, models/model_meta.json
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
from generate_weather import generate_weather
from grid_simulator import simulate_grid
from feature_engineering import add_features

FEATURES = ["temperature", "rainfall", "humidity", "wind_speed", "WSI", "CWVI"]
TARGETS  = ["LOLP", "EENS"]
MODEL_DIR = "models"
DATA_DIR  = "data"


# ──────────────────────────────────────────────────────────────
# 1. Build / load dataset
# ──────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "grid_dataset.csv")
    if os.path.exists(csv_path):
        print(f"[dataset] Loading cached dataset from {csv_path}")
        return pd.read_csv(csv_path)

    print("[dataset] Generating weather data …")
    weather = generate_weather(10_000)

    print("[dataset] Running grid simulator …")
    grid_df = simulate_grid(weather)

    print("[dataset] Computing features …")
    full_df = add_features(grid_df)

    full_df.to_csv(csv_path, index=False)
    print(f"[dataset] Saved → {csv_path}  ({len(full_df)} rows)")
    return full_df


# ──────────────────────────────────────────────────────────────
# 2. Split
# ──────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    X = df[FEATURES].values
    # Log-transform EENS (right-skewed) for better regression
    y_lolp = df["LOLP"].values
    y_eens = np.log1p(df["EENS"].values)

    X_temp, X_test, yl_temp, yl_test, ye_temp, ye_test = train_test_split(
        X, y_lolp, y_eens, test_size=0.15, random_state=42
    )
    X_train, X_val, yl_train, yl_val, ye_train, ye_val = train_test_split(
        X_temp, yl_temp, ye_temp, test_size=0.15 / 0.85, random_state=42
    )
    print(f"[split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return (X_train, X_val, X_test,
            yl_train, yl_val, yl_test,
            ye_train, ye_val, ye_test)


# ──────────────────────────────────────────────────────────────
# 3. Train
# ──────────────────────────────────────────────────────────────
def train_rf(X_train, y_train, label: str) -> RandomForestRegressor:
    print(f"[train] Training RandomForest for {label} …")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"  OOB R²: {rf.oob_score_:.4f}")
    return rf


# ──────────────────────────────────────────────────────────────
# 4. Evaluate
# ──────────────────────────────────────────────────────────────
def evaluate(model, X, y_true, split: str, label: str) -> dict:
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"  [{split}] {label}  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.4f}")
    return {"split": split, "target": label, "MAE": mae, "RMSE": rmse, "R2": r2}


# ──────────────────────────────────────────────────────────────
# 5. Save
# ──────────────────────────────────────────────────────────────
def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = build_dataset()

    (X_train, X_val, X_test,
     yl_train, yl_val, yl_test,
     ye_train, ye_val, ye_test) = split_data(df)

    # ── Feature scaler (saved for inference) ──────────────────
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Train ──────────────────────────────────────────────────
    rf_lolp = train_rf(X_train_s, yl_train, "LOLP")
    rf_eens = train_rf(X_train_s, ye_train, "EENS(log)")

    # ── Evaluate ───────────────────────────────────────────────
    metrics = []
    for split, Xs, yl, ye in [
        ("val",  X_val_s,  yl_val,  ye_val),
        ("test", X_test_s, yl_test, ye_test),
    ]:
        metrics.append(evaluate(rf_lolp, Xs, yl, split, "LOLP"))
        metrics.append(evaluate(rf_eens, Xs, ye, split, "EENS(log)"))

    # ── Feature importances ────────────────────────────────────
    fi_lolp = dict(zip(FEATURES, rf_lolp.feature_importances_.tolist()))
    fi_eens = dict(zip(FEATURES, rf_eens.feature_importances_.tolist()))
    print("\n[importance] LOLP:", {k: f"{v:.4f}" for k, v in fi_lolp.items()})
    print("[importance] EENS:", {k: f"{v:.4f}" for k, v in fi_eens.items()})

    # ── Save artifacts ─────────────────────────────────────────
    save_model(rf_lolp,  f"{MODEL_DIR}/rf_lolp.pkl")
    save_model(rf_eens,  f"{MODEL_DIR}/rf_eens.pkl")
    save_model(scaler,   f"{MODEL_DIR}/scaler.pkl")

    meta = {
        "features":           FEATURES,
        "feature_importance": {"LOLP": fi_lolp, "EENS": fi_eens},
        "metrics":            metrics,
        "eens_log_transform": True,
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] Model meta saved → {MODEL_DIR}/model_meta.json")


if __name__ == "__main__":
    main()
