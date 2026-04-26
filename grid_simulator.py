"""
grid_simulator.py
Weather-Aware Power Grid Reliability Simulator
Generates realistic synthetic electricity grid data based on weather conditions.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
BASE_LOAD_MW       = 5000   # MW baseline demand
CAPACITY_MARGIN    = 1.05   # capacity headroom over peak demand (tighter → realistic LOLP)
NOISE_SIGMA_LOAD   = 0.03   # ±3% Gaussian noise on load
NOISE_SIGMA_CAP    = 0.06   # ±6% Gaussian noise on capacity
SPIKE_PROB         = 0.005  # 0.5% chance of demand spike
SPIKE_MAGNITUDE    = 0.20   # +20% on spike events
RNG_SEED           = 42


# ──────────────────────────────────────────────────────────────
# Helper: season index  (0=winter, 1=spring, 2=summer, 3=autumn)
# ──────────────────────────────────────────────────────────────
def _season(month: int) -> int:
    return (month % 12) // 3


# ──────────────────────────────────────────────────────────────
# 1. Load Demand  (MW)
# ──────────────────────────────────────────────────────────────
def compute_load_demand(
    temperature: np.ndarray,
    month: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Load = BASE × (1 + seasonal_factor) × temp_effect × noise × [spike]
    - Summer peak: +15%  | Winter heating: +10%
    - Every 1°C above 22°C adds 0.8%  (cooling load)
    - Every 1°C below 15°C adds 0.5%  (heating load)
    """
    n = len(temperature)

    # Seasonal multiplier
    season = _season(month)
    seasonal = np.where(season == 2, 0.15,       # summer
               np.where(season == 0, 0.10,       # winter
               np.where(season == 1, -0.05,      # spring: mild
               -0.02)))                           # autumn: mild

    # Temperature sensitivity
    cooling = np.maximum(0, temperature - 22) * 0.008
    heating = np.maximum(0, 15 - temperature) * 0.005
    temp_effect = 1.0 + cooling + heating

    # Gaussian noise
    noise = rng.normal(1.0, NOISE_SIGMA_LOAD, n)

    # Rare demand spikes (storm prep, industrial surge, etc.)
    spikes = 1.0 + (rng.random(n) < SPIKE_PROB) * SPIKE_MAGNITUDE

    demand = BASE_LOAD_MW * (1 + seasonal) * temp_effect * noise * spikes
    return demand.clip(BASE_LOAD_MW * 0.4, BASE_LOAD_MW * 2.0)


# ──────────────────────────────────────────────────────────────
# 2. Generation Capacity  (MW)
# ──────────────────────────────────────────────────────────────
def compute_capacity(
    demand: np.ndarray,
    wind_speed: np.ndarray,
    rainfall: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Capacity = demand × CAPACITY_MARGIN × derating × noise
    - High wind (>25 m/s): turbines trip, −8%
    - Heavy rain (>80 mm): hydro surge +5%, thermal −3%
    """
    n = len(demand)

    # Wind derating
    wind_derating = np.where(wind_speed > 30, 0.88,
                    np.where(wind_speed > 25, 0.94, 1.00))

    # Rainfall effect: slight hydro boost partially offset by flooding risk
    rain_effect = np.where(rainfall > 80, 1.02,
                  np.where(rainfall > 50, 1.01, 1.00))

    noise = rng.normal(1.0, NOISE_SIGMA_CAP, n)

    capacity = demand * CAPACITY_MARGIN * wind_derating * rain_effect * noise
    return capacity.clip(demand * 0.70, demand * 1.60)


# ──────────────────────────────────────────────────────────────
# 3. Outage Simulation
# ──────────────────────────────────────────────────────────────
def compute_outages(
    temperature: np.ndarray,
    rainfall: np.ndarray,
    wind_speed: np.ndarray,
    humidity: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (outage_count, outage_duration_hours)
    Base outage rate is amplified by adverse weather conditions.
    """
    n = len(temperature)

    # Base probability of at least one outage on a given day
    base_p = 0.04   # 4% on a calm day

    # Weather stress multipliers (additive contributions)
    heat_stress   = np.maximum(0, temperature - 40) * 0.015
    rain_stress   = np.maximum(0, rainfall    - 80) * 0.003
    wind_stress   = np.maximum(0, wind_speed  - 25) * 0.008
    humid_stress  = np.maximum(0, humidity    - 85) * 0.004

    p_outage = (base_p + heat_stress + rain_stress + wind_stress + humid_stress).clip(0, 0.85)

    # Sample outage occurrence
    outage_occurs = rng.random(n) < p_outage

    # Outage count: 1–5 per event, more events in severe conditions
    max_count = (1 + (heat_stress + wind_stress + rain_stress) * 2).clip(1, 8).astype(int)
    outage_count = np.where(outage_occurs, rng.integers(1, max_count + 1, n), 0)

    # Duration: 0.5–12 hours, correlated with severity
    severity = (heat_stress + rain_stress * 1.5 + wind_stress * 2).clip(0, 1)
    mean_dur = 1.0 + severity * 11.0   # 1 h baseline → up to 12 h in extreme events
    raw_dur  = rng.exponential(mean_dur, n)
    outage_duration = np.where(outage_occurs, raw_dur.clip(0.5, 24), 0.0)

    return outage_count.astype(int), outage_duration


# ──────────────────────────────────────────────────────────────
# 4. LOLP – Loss of Load Probability
# ──────────────────────────────────────────────────────────────
def compute_lolp(demand: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """LOLP = max(0, (demand - capacity) / capacity)"""
    raw = (demand - capacity) / capacity
    return np.maximum(0, raw)


# ──────────────────────────────────────────────────────────────
# 5. EENS – Expected Energy Not Served  (MWh)
# ──────────────────────────────────────────────────────────────
def compute_eens(
    outage_duration: np.ndarray,
    demand: np.ndarray,
    lolp: np.ndarray,
) -> np.ndarray:
    """
    EENS = outage_duration × demand × probability_factor
    probability_factor blends LOLP with a minimum floor for non-zero outages.
    """
    prob_factor = np.where(outage_duration > 0,
                           np.maximum(lolp, 0.01),   # at least 1% when outage exists
                           lolp)
    eens = outage_duration * (demand / 1000) * prob_factor  # convert MW→GW for scale
    return eens.clip(0)


# ──────────────────────────────────────────────────────────────
# Main simulation entry point
# ──────────────────────────────────────────────────────────────
def simulate_grid(weather_df: pd.DataFrame, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Given a weather DataFrame with columns:
        date, temperature, rainfall, humidity, wind_speed
    Returns the same DataFrame augmented with grid reliability columns.
    """
    rng = np.random.default_rng(seed)

    temperature = weather_df["temperature"].values.astype(float)
    rainfall    = weather_df["rainfall"].values.astype(float)
    humidity    = weather_df["humidity"].values.astype(float)
    wind_speed  = weather_df["wind_speed"].values.astype(float)
    month       = pd.to_datetime(weather_df["date"]).dt.month.values

    # Run simulation chain
    demand   = compute_load_demand(temperature, month, rng)
    capacity = compute_capacity(demand, wind_speed, rainfall, rng)
    outage_count, outage_duration = compute_outages(
        temperature, rainfall, wind_speed, humidity, rng
    )
    lolp = compute_lolp(demand, capacity)
    eens = compute_eens(outage_duration, demand, lolp)

    out = weather_df.copy()
    out["load_demand"]       = np.round(demand, 2)
    out["generation_capacity"] = np.round(capacity, 2)
    out["outage_count"]      = outage_count
    out["outage_duration"]   = np.round(outage_duration, 3)
    out["LOLP"]              = np.round(lolp, 6)
    out["EENS"]              = np.round(eens, 4)

    return out


if __name__ == "__main__":
    # Quick smoke test
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    sample = pd.DataFrame({
        "date":        dates,
        "temperature": [25, 38, 42, 15, 10, 30, 35, 45, 20, 28],
        "rainfall":    [5, 20, 90, 2, 0, 60, 30, 10, 0, 45],
        "humidity":    [60, 75, 90, 50, 45, 80, 70, 65, 55, 78],
        "wind_speed":  [10, 15, 30, 8, 5, 20, 12, 28, 7, 18],
    })
    result = simulate_grid(sample)
    print(result[["date","load_demand","generation_capacity","outage_count","outage_duration","LOLP","EENS"]].to_string())
