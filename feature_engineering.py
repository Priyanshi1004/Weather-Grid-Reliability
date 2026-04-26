"""
feature_engineering.py
Computes WSI (Weather Stress Index) and CWVI (Combined Weather-Vulnerability Index).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds WSI and CWVI to a DataFrame that contains:
        temperature, rainfall, humidity, wind_speed,
        load_demand, generation_capacity
    """
    df = df.copy()

    # ── Normalize weather variables to [0, 1] ─────────────────
    weather_cols = ["temperature", "rainfall", "humidity", "wind_speed"]
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(df[weather_cols])
    t_n, r_n, h_n, w_n = normed[:, 0], normed[:, 1], normed[:, 2], normed[:, 3]

    # ── WSI: Weather Stress Index ──────────────────────────────
    # Weights reflect empirical sensitivity to each variable
    #   temperature: 0.35  (heat/cold stress dominates)
    #   rainfall:    0.25  (flooding, equipment damage)
    #   wind_speed:  0.25  (turbine trips, line damage)
    #   humidity:    0.15  (insulation degradation)
    df["WSI"] = np.round(
        0.35 * t_n + 0.25 * r_n + 0.25 * w_n + 0.15 * h_n, 6
    )

    # ── CWVI: Combined Weather-Vulnerability Index ─────────────
    # Incorporates grid stress (demand vs capacity ratio)
    load_ratio = df["load_demand"] / df["generation_capacity"]
    lr_normed  = (load_ratio - load_ratio.min()) / (load_ratio.max() - load_ratio.min() + 1e-9)
    df["CWVI"] = np.round(df["WSI"] + lr_normed * 0.5, 6)   # 50% weight for grid margin

    return df
