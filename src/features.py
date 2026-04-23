"""
features.py
===========
Feature engineering for Turkish electricity price forecasting.

All features use only information available at prediction time (t-1 or earlier).
No data leakage: current price is NEVER used as an input feature.

Key feature groups:
  - Calendar: hour, day_of_week, month, season, is_weekend, is_holiday
  - Cyclical encodings: sin/cos transforms for hour and month
  - Lag features: price at t-24, t-48, t-168 (1 week)
  - Rolling statistics: 24h and 168h mean/std of price
  - Weather: current temperature + lag-24 for 3 cities
"""

import numpy as np
import pandas as pd

# Turkish public holidays (approximate — not exhaustive)
TURKISH_HOLIDAYS = {
    (1, 1),   # New Year
    (4, 23),  # National Sovereignty Day
    (5, 1),   # Labor Day
    (5, 19),  # Atatürk Commemoration
    (7, 15),  # Democracy Day
    (8, 30),  # Victory Day
    (10, 29), # Republic Day
}


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """
    Add calendar-based features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a datetime column named `ts_col`.
    ts_col : str
        Name of the timestamp column.

    Returns
    -------
    pd.DataFrame with new columns added in-place copy.
    """
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["quarter"] = ts.dt.quarter
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = ts.apply(
        lambda t: int((t.month, t.day) in TURKISH_HOLIDAYS)
    )
    df["season"] = ts.dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos cyclical encodings for hour and month.

    These encode periodicity without imposing ordinal distance assumptions.
    """
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, price_col: str = "dam_price_try_mwh") -> pd.DataFrame:
    """
    Add lagged price features.

    Lags used:
      - t-24  : same hour yesterday
      - t-48  : same hour two days ago
      - t-168 : same hour last week (strongest seasonality signal)

    Note: rows with insufficient history will have NaN lags and should be dropped
    before training.
    """
    df = df.copy()
    df["price_lag_24"] = df[price_col].shift(24)
    df["price_lag_48"] = df[price_col].shift(48)
    df["price_lag_168"] = df[price_col].shift(168)
    return df


def add_rolling_features(df: pd.DataFrame, price_col: str = "dam_price_try_mwh") -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.

    Windows:
      - 24h  : captures short-term price regime
      - 168h : captures weekly baseline and volatility
    """
    df = df.copy()
    df["price_roll_mean_24"] = df[price_col].shift(1).rolling(24).mean()
    df["price_roll_std_24"] = df[price_col].shift(1).rolling(24).std()
    df["price_roll_mean_168"] = df[price_col].shift(1).rolling(168).mean()
    df["price_roll_std_168"] = df[price_col].shift(1).rolling(168).std()
    return df


def add_weather_features(
    df: pd.DataFrame,
    cities: list = None,
) -> pd.DataFrame:
    """
    Add weather features for available cities.

    For each city:
      - temp_{city}       : current temperature
      - temp_{city}_lag24 : temperature 24 hours ago
      - temp_{city}_dev   : deviation from 30-day rolling mean (anomaly signal)
    """
    if cities is None:
        cities = ["istanbul", "ankara", "izmir"]

    df = df.copy()
    for city in cities:
        col = f"temp_{city}"
        if col not in df.columns:
            continue
        df[f"{col}_lag24"] = df[col].shift(24)
        rolling_mean = df[col].shift(1).rolling(30 * 24).mean()
        df[f"{col}_dev"] = df[col] - rolling_mean

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence and return the full feature matrix.

    Also drops rows with NaN values introduced by lags/rolling windows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw processed dataset with columns: ts, dam_price_try_mwh, temp_*

    Returns
    -------
    pd.DataFrame with all features, NaN rows dropped.
    """
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour", "day_of_week", "month", "year", "quarter", "is_weekend", "is_holiday", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "price_lag_24", "price_lag_48", "price_lag_168",
    "price_roll_mean_24", "price_roll_std_24", "price_roll_mean_168", "price_roll_std_168",
    "temp_istanbul", "temp_istanbul_lag24", "temp_istanbul_dev",
    "temp_ankara", "temp_ankara_lag24", "temp_ankara_dev",
    "temp_izmir", "temp_izmir_lag24", "temp_izmir_dev",
]

TARGET_COL = "dam_price_try_mwh"


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return list of feature columns that are present in df."""
    return [c for c in FEATURE_COLS if c in df.columns]
