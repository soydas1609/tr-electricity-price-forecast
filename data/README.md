# Data

Raw and processed data files are **not committed** to this repository (see `.gitignore`).

## Sources

### Day-Ahead Market Prices
- **Source**: ENTSO-E Transparency Platform
- **URL**: https://transparency.entsoe.eu/
- **Bidding zone**: Turkey (10YTR-TEIAS----W)
- **Resolution**: Hourly
- **Coverage**: 2020-01-01 to 2025-12-31
- **Unit**: EUR/MWh (converted to TL/MWh using historical exchange rates)

### Weather Data
- **Source**: Open-Meteo Historical Weather API
- **URL**: https://open-meteo.com/
- **Cities**: Istanbul (41.01°N, 28.98°E), Ankara (39.93°N, 32.86°E), Izmir (38.42°N, 27.14°E)
- **Resolution**: Hourly
- **Variable**: temperature_2m (°C)
- **No API key required**

## How to Download

```bash
cp .env.example .env
# Edit .env with your ENTSO-E API key

python src/data_loader.py
```

Estimated time: 5-10 minutes.
