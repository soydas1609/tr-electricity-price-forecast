"""
models.py
=========
Model definitions for electricity price forecasting.

All models share a common interface:
    model.fit(X_train, y_train)
    model.predict(X_test) -> np.ndarray
    model.name -> str

Models:
  1. SeasonalNaive        — y_hat[t] = y[t-168]  (1-week lag baseline)
  2. RidgeModel           — regularized linear regression on engineered features
  3. GradientBoostingModel — histogram-based gradient boosted trees (sklearn)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


class SeasonalNaive:
    """
    Seasonal naive baseline: predict using the same hour last week (t-168).

    This is the natural benchmark for hourly electricity price forecasting.
    Any model that cannot beat this baseline is not useful in practice.
    """

    name = "SeasonalNaive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        """No fitting required — stateless model."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return price_lag_168 column as predictions.

        Requires 'price_lag_168' to be present in X.
        """
        if "price_lag_168" not in X.columns:
            raise ValueError("SeasonalNaive requires 'price_lag_168' feature in X.")
        return X["price_lag_168"].values


class RidgeModel:
    """
    Ridge regression on standardized features.

    Uses sklearn Pipeline(StandardScaler → Ridge) so that scaling is
    consistently applied at both fit and predict time.
    """

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return absolute coefficients as a proxy for feature importance."""
        return np.abs(self.pipeline.named_steps["ridge"].coef_)


class GradientBoostingModel:
    """
    Histogram-based Gradient Boosted Trees (sklearn HistGradientBoostingRegressor).

    Equivalent to LightGBM in methodology — uses histogram binning for speed.
    Default hyperparameters are conservative to avoid overfitting on rolling-window eval.
    """

    name = "GradientBoosting"

    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def get_all_models() -> list:
    """Return a list of all model instances for comparison."""
    return [SeasonalNaive(), RidgeModel(), GradientBoostingModel()]
