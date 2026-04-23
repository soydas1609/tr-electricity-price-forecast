"""
data_loader.py
==============
Download and cache Turkish electricity market data.

Sources:
  - ENTSO-E Transparency Platform: Day-Ahead prices for Turkey (10YTR-TEIAS----W)
  - Open-Meteo: Hourly temperature for Istanbul, Ankara, Izmir

Usage:
    python src/data_loader.py
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TURKEY_ZONE = "10YTR-TEIAS----W"
START = "20200101"
END = "20251231"

CITIES = {
    "istanbul": (41.0082, 28.9784),
    "ankara": (39.9334, 32.8597),
    "izmir": (38.4192, 27.1287),
}

# USD/TRY approximate annual averages (rough conversion for portfolio context)
USD_TRY = {
    2020: 7.5, 2021: 8.8, 2022: 16.5, 2023: 26.0, 2024: 32.0, 2025: 36.0,
}
# EUR/USD rough annual averages
EUR_USD = {
    2020: 1.14, 2021: 1.18, 2022: 1.05, 2023: 1.08, 2024: 1.09, 2025: 1.10,
}


def _eur_to_try(price_eur: float, year: int) -> float:
    eur_usd = EUR_USD.get(year, 1.10)
    usd_try = USD_TRY.get(year, 32.0)
    return price_eur * eur_usd * usd_try


def fetch_entsoe_prices() -> pd.DataFrame:
    """
    Download hourly Day-Ahead prices for Turkey from ENTSO-E.

    Returns a DataFrame with columns: ts (UTC), dam_price_eur_mwh, dam_price_try_mwh.
    Falls back to a realistic synthetic dataset if API key is not configured.
    """
    api_key = os.getenv("ENTSOE_API_KEY", "")
    cache_path = RAW_DIR / "entsoe_dam_prices.csv"

    if cache_path.exists():
        log.info("Loading cached ENTSO-E prices from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["ts"])

    if not api_key or api_key == "your_api_key_here":
        log.warning("ENTSOE_API_KEY not set — generating synthetic DAM prices.")
        return _generate_synthetic_prices()

    try:
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(START, tz="UTC")
        end = pd.Timestamp(END + "T235959", tz="UTC")
        log.info("Fetching ENTSO-E day-ahead prices %s → %s", start, end)
        series = client.query_day_ahead_prices(TURKEY_ZONE, start=start, end=end)
        df = series.reset_index()
        df.columns = ["ts", "dam_price_eur_mwh"]
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        df["year"] = df["ts"].dt.year
        df["dam_price_try_mwh"] = df.apply(
            lambda r: _eur_to_try(r["dam_price_eur_mwh"], r["year"]), axis=1
        )
        df.drop(columns=["year"], inplace=True)
        df.to_csv(cache_path, index=False)
        log.info("Saved %d rows to %s", len(df), cache_path)
        return df

    except Exception as exc:
        log.warning("ENTSO-E fetch failed (%s) — falling back to synthetic.", exc)
        return _generate_synthetic_prices()


def _generate_synthetic_prices() -> pd.DataFrame:
    """
    Generate 6 years of realistic synthetic Turkish DAM prices.

    Uses a mean-reverting stochastic process with:
    - Hourly seasonality (peak 18:00-22:00)
    - Weekly seasonality (weekends ~10% lower)
    - Annual upward trend (TL inflation)
    - Random shocks
    """
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", "2025-12-31 23:00", freq="h")
    n = len(idx)

    hour = idx.hour
    dow = idx.dayofweek
    year_frac = (idx.year - 2020) + idx.dayofyear / 366

    # Base TRY price growing with inflation
    base = 300 + 250 * year_frac

    # Hourly seasonality: two peaks (morning + evening)
    hourly = (
        50 * np.sin(np.pi * (hour - 6) / 16) +
        30 * np.sin(2 * np.pi * (hour - 18) / 24)
    )
    hourly = np.where((hour >= 8) & (hour <= 20), hourly + 40, hourly - 20)

    # Weekly: weekends ~10% discount
    weekly = np.where(dow >= 5, -0.10, 0.0)

    # Random walk component
    shocks = np.random.normal(0, 1, n)
    rw = np.cumsum(shocks) * 0.5
    # Mean revert the random walk
    rw = rw - np.convolve(rw, np.ones(168) / 168, mode="same")

    prices = base * (1 + weekly) + hourly + rw
    prices = np.clip(prices, 50, None)  # no negative prices

    df = pd.DataFrame({"ts": idx, "dam_price_eur_mwh": np.nan, "dam_price_try_mwh": prices})
    cache_path = RAW_DIR / "entsoe_dam_prices.csv"
    df.to_csv(cache_path, index=False)
    log.info("Synthetic prices generated: %d rows", len(df))
    return df


def fetch_weather() -> pd.DataFrame:
    """
    Download hourly temperature data for 3 Turkish cities via Open-Meteo.

    Returns a DataFrame with columns: ts, city, temp_c.
    No API key required.
    """
    cache_path = RAW_DIR / "weather.csv"
    if cache_path.exists():
        log.info("Loading cached weather from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["ts"])

    frames = []
    start_str = "2020-01-01"
    end_str = "2025-12-31"

    for city, (lat, lon) in CITIES.items():
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_str}&end_date={end_str}"
            "&hourly=temperature_2m&timezone=UTC"
        )
        log.info("Fetching weather for %s ...", city)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            df_city = pd.DataFrame({
                "ts": pd.to_datetime(data["hourly"]["time"]),
                "city": city,
                "temp_c": data["hourly"]["temperature_2m"],
            })
            frames.append(df_city)
        except Exception as exc:
            log.warning("Weather fetch failed for %s (%s) — using synthetic.", city, exc)
            frames.append(_generate_synthetic_weather(city))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache_path, index=False)
    log.info("Weather data saved: %d rows", len(df))
    return df


def _generate_synthetic_weather(city: str) -> pd.DataFrame:
    """Generate synthetic hourly temperature for a city."""
    np.random.seed({"istanbul": 1, "ankara": 2, "izmir": 3}.get(city, 0))
    idx = pd.date_range("2020-01-01", "2025-12-31 23:00", freq="h")
    n = len(idx)
    doy = idx.dayofyear
    hour = idx.hour

    base_temps = {"istanbul": 14, "ankara": 11, "izmir": 17}
    base = base_temps.get(city, 13)

    seasonal = -15 * np.cos(2 * np.pi * doy / 365)
    diurnal = 6 * np.sin(np.pi * (hour - 6) / 12)
    noise = np.random.normal(0, 1.5, n)

    temp = base + seasonal + diurnal + noise
    return pd.DataFrame({"ts": idx, "city": city, "temp_c": temp})


def build_processed_dataset(prices: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price and weather data into a single processed hourly DataFrame.

    Returns a wide-format DataFrame with columns:
        ts, dam_price_try_mwh, temp_istanbul, temp_ankara, temp_izmir
    """
    prices = prices[["ts", "dam_price_try_mwh"]].copy()
    prices["ts"] = pd.to_datetime(prices["ts"])
    prices = prices.sort_values("ts").reset_index(drop=True)

    weather["ts"] = pd.to_datetime(weather["ts"])
    weather_wide = weather.pivot_table(index="ts", columns="city", values="temp_c")
    weather_wide.columns = [f"temp_{c}" for c in weather_wide.columns]
    weather_wide = weather_wide.reset_index()

    df = prices.merge(weather_wide, on="ts", how="left")
    df = df.dropna(subset=["dam_price_try_mwh"])

    out_path = PROCESSED_DIR / "hourly_market.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Processed dataset saved: %d rows → %s", len(df), out_path)
    return df


if __name__ == "__main__":
    log.info("=== Data Loader ===")
    prices = fetch_entsoe_prices()
    log.info("Prices: %d rows, columns: %s", len(prices), list(prices.columns))

    weather = fetch_weather()
    log.info("Weather: %d rows", len(weather))

    df = build_processed_dataset(prices, weather)
    log.info("Final dataset: %d rows, %d columns", len(df), df.shape[1])
    log.info("Date range: %s → %s", df["ts"].min(), df["ts"].max())
    log.info("Price range: %.1f – %.1f TL/MWh", df["dam_price_try_mwh"].min(), df["dam_price_try_mwh"].max())
    print("\nHead:\n", df.head())
    print("\nDone.")
