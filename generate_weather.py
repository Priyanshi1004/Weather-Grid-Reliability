"""
generate_weather.py
Generates 10,000 rows of realistic synthetic weather data.
Seasonal patterns with autocorrelation and physical plausibility constraints.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS   = 10_000

def generate_weather(n: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    month = dates.month.values
    day_of_year = dates.dayofyear.values

    # ── Temperature (°C) ───────────────────────────────────────
    # Annual sinusoid: peak ~summer (day 200), trough ~winter (day 20)
    t_seasonal = 25 + 18 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # AR(1) autocorrelation for day-to-day persistence
    t_noise = np.zeros(n)
    t_noise[0] = rng.normal(0, 3)
    for i in range(1, n):
        t_noise[i] = 0.80 * t_noise[i-1] + rng.normal(0, 2)
    temperature = (t_seasonal + t_noise).clip(-5, 50)

    # ── Rainfall (mm/day) ──────────────────────────────────────
    # Monsoon: months 6–9 are much wetter
    monsoon = np.isin(month, [6, 7, 8, 9])
    base_rain_p = np.where(monsoon, 0.45, 0.15)
    rain_occurs  = rng.random(n) < base_rain_p
    rain_amount  = np.where(
        monsoon,
        rng.exponential(18, n),    # heavier monsoon rain
        rng.exponential(6, n),     # lighter off-season rain
    )
    # Occasional extreme events
    extreme = rng.random(n) < 0.005
    rain_amount = np.where(extreme, rain_amount * 6, rain_amount)
    rainfall = np.where(rain_occurs, rain_amount, 0.0).clip(0, 300)

    # ── Humidity (%) ───────────────────────────────────────────
    # Correlated with rainfall + seasonal baseline
    h_base = 55 + 20 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    h_rain_bonus = np.minimum(rainfall * 0.25, 30)
    h_noise = rng.normal(0, 5, n)
    humidity = (h_base + h_rain_bonus + h_noise).clip(20, 100)

    # ── Wind Speed (m/s) ───────────────────────────────────────
    # Higher in winter/spring; calmer in summer
    w_seasonal = 8 + 4 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
    w_noise = rng.gamma(shape=1.5, scale=1.5, size=n)  # right-skewed
    # Storms correlated with heavy rain
    w_storm = np.where(rainfall > 60, rng.uniform(5, 20, n), 0)
    wind_speed = (w_seasonal + w_noise + w_storm).clip(0, 50)

    df = pd.DataFrame({
        "date":        dates,
        "temperature": np.round(temperature, 1),
        "rainfall":    np.round(rainfall,    1),
        "humidity":    np.round(humidity,    1),
        "wind_speed":  np.round(wind_speed,  1),
    })
    return df


if __name__ == "__main__":
    df = generate_weather()
    df.to_csv("data/weather_data.csv", index=False)
    print(f"Generated {len(df)} rows → data/weather_data.csv")
    print(df.describe())
