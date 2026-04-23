"""
test_features.py
================
Unit tests for feature engineering.

Critical tests:
  1. No data leakage: lag features must shift data, not expose current values.
  2. Cyclical features are bounded in [-1, 1].
  3. Feature matrix has no NaN in expected columns after dropna.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    FEATURE_COLS,
    TARGET_COL,
)


@pytest.fixture
def sample_df():
    """Create a minimal DataFrame with 300 hourly rows for testing."""
    np.random.seed(0)
    idx = pd.date_range("2023-01-01", periods=300, freq="h")
    return pd.DataFrame({
        "ts": idx,
        "dam_price_try_mwh": 1000 + np.random.randn(300) * 100,
        "temp_istanbul": 10 + np.random.randn(300) * 5,
        "temp_ankara": 8 + np.random.randn(300) * 5,
        "temp_izmir": 14 + np.random.randn(300) * 5,
    })


def test_lag24_no_leakage(sample_df):
    """
    price_lag_24 at row i must equal the price at row i-24.
    If it equals the current row's price, we have leakage.
    """
    df = add_lag_features(sample_df.copy())
    # Check a specific row that definitely has a valid lag
    i = 50
    expected = sample_df.loc[i - 24, "dam_price_try_mwh"]
    actual = df.loc[i, "price_lag_24"]
    assert abs(actual - expected) < 1e-6, f"Lag-24 at row {i}: expected {expected}, got {actual}"


def test_lag168_no_leakage(sample_df):
    """price_lag_168 must never equal the current price."""
    df = add_lag_features(sample_df.copy())
    valid = df.dropna(subset=["price_lag_168"])
    # For lag-168, shifted price should not match current price (statistically impossible)
    leaks = (valid["price_lag_168"] == valid["dam_price_try_mwh"]).sum()
    assert leaks == 0, f"price_lag_168 equals current price in {leaks} rows — possible leakage!"


def test_cyclical_bounds(sample_df):
    """sin/cos features must be in [-1, 1]."""
    df = add_calendar_features(sample_df.copy())
    df = add_cyclical_features(df)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert df[col].between(-1, 1).all(), f"{col} out of [-1, 1] range"


def test_rolling_features_use_shift(sample_df):
    """
    Rolling mean at row i should use data up to row i-1 (exclusive of current).
    This ensures we don't leak current-hour price into rolling stats.
    """
    df = add_lag_features(sample_df.copy())
    df = add_rolling_features(df)
    # Row 30: rolling mean should equal mean of rows 7..29 (shifted by 1, window 24)
    i = 30
    expected = sample_df.loc[6:29, "dam_price_try_mwh"].mean()
    actual = df.loc[i, "price_roll_mean_24"]
    assert abs(actual - expected) < 1e-6, f"Rolling mean mismatch: expected {expected:.2f}, got {actual:.2f}"


def test_build_feature_matrix_no_nans(sample_df):
    """After build_feature_matrix, there should be no NaN values in feature columns."""
    df = build_feature_matrix(sample_df.copy())
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    nan_counts = df[feature_cols_present].isna().sum()
    assert nan_counts.sum() == 0, f"NaN values found:\n{nan_counts[nan_counts > 0]}"


def test_calendar_features_range(sample_df):
    """Calendar features must be within expected ranges."""
    df = add_calendar_features(sample_df.copy())
    assert df["hour"].between(0, 23).all()
    assert df["month"].between(1, 12).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["is_weekend"].isin([0, 1]).all()
    assert df["is_holiday"].isin([0, 1]).all()
    assert df["season"].isin([0, 1, 2, 3]).all()
