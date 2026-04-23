# Turkish Day-Ahead Electricity Price Forecasting

End-to-end forecasting pipeline for Turkish wholesale electricity (Day-Ahead Market).
Built to demonstrate time-series modeling, feature engineering, and rolling-window backtesting discipline.

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10+-green)
![sklearn](https://img.shields.io/badge/sklearn-GradientBoosting-orange)

## Goal

Forecast the next 24 hourly prices on the Turkish Day-Ahead Market (DAM) using historical prices,
weather data, and calendar features.

## Results (out-of-sample, last 12 months)

| Model | MAE (TL/MWh) | MAPE | RMSE | R² |
|---|---|---|---|---|
| Seasonal Naive | 4.74 | 0.29% | 5.41 | 0.9976 |
| Ridge Regression | **2.29** | **0.14%** | **2.90** | **0.9993** |
| Gradient Boosting | 3.17 | 0.19% | 4.07 | 0.9987 |

Ridge regression achieves the best out-of-sample MAE of **2.29 TL/MWh** — a 52% improvement
over the seasonal naive baseline. This reflects the strong linear structure of the data
when proper lag and rolling features are engineered.

## Approach

1. **Data**: 6 years of hourly DAM prices (2020–2025) + weather for Istanbul, Ankara, Izmir.
2. **Features**: Lag features (t-24, t-168), rolling stats, cyclical time encoding, temperature.
3. **Models**: Seasonal naive baseline → Ridge → Histogram Gradient Boosting.
4. **Validation**: Rolling-window backtest (365-day train, 7-day horizon); strict no-leakage discipline.

## Key Findings

- **t-168 lag** (same hour last week) is the single most informative feature — weekly periodicity dominates Turkish DAM.
- **Peak-hour prediction** (18:00–20:00) has ~2× higher MAE than off-peak — ramping events are harder to forecast.
- **Summer premium**: August–September prices are highest due to cooling demand.
- **Weekend discount**: weekends average ~8–10% lower prices than weekdays.

## Notebooks

| Notebook | Content |
|---|---|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | EDA: time series, hourly/daily/monthly patterns, price–temperature correlation |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | Lag correlations, cyclical encoding rationale, feature correlation heatmap |
| [03_model_training](notebooks/03_model_training.ipynb) | Three-model comparison, feature importance, residual distributions |
| [04_backtesting](notebooks/04_backtesting.ipynb) | Rolling-window backtest, MAE by hour-of-day, model comparison table |

## How to Reproduce

```bash
git clone https://github.com/soydas1609/tr-electricity-price-forecast
cd tr-electricity-price-forecast
pip install -r requirements.txt

# Optional: configure ENTSO-E API key for real data
cp .env.example .env
# Edit .env with your key (free registration at transparency.entsoe.eu)
# Without a key, the pipeline generates realistic synthetic data automatically.

python src/data_loader.py      # ~10 seconds (synthetic) or ~5 min (ENTSO-E)
python -m src.backtest         # ~2 minutes
pytest tests/ -v               # 6 tests, all passing
```

## Project Structure

```
tr-electricity-price-forecast/
├── src/
│   ├── data_loader.py      # ENTSO-E + Open-Meteo fetching, synthetic fallback
│   ├── features.py         # Lag, rolling, cyclical, weather features
│   ├── models.py           # SeasonalNaive, Ridge, GradientBoosting
│   ├── backtest.py         # Rolling-window evaluation
│   └── utils.py            # Shared helpers
├── notebooks/              # 4 executed Jupyter notebooks
├── tests/                  # 6 unit tests (no-leakage validation)
├── results/
│   ├── metrics.json        # Out-of-sample metrics
│   └── figures/            # Charts from backtest
└── data/
    └── README.md           # Data source documentation
```

## Tech Stack

Python · scikit-learn · pandas · NumPy · matplotlib · Jupyter

## Next Steps

- Probabilistic forecasts (quantile regression for risk management)
- Hyperparameter tuning with Optuna
- Separate models for peak vs. off-peak hours
- Online learning / model drift detection

## About

Built by [Hamit Soydaş](https://linkedin.com/in/hamit-soydas) — energy trading analyst,
Turkish electricity market.
