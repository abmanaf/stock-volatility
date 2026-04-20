# Stock Volatility & Returns Prediction Backend

FastAPI-based backend for training GARCH volatility models and ARIMA returns models, with forecasting capabilities.

## Overview

This backend provides:
- **GARCH model training** - For predicting stock return volatility
- **ARIMA model training** - For predicting expected stock returns  
- **Volatility forecasting** - Multi-day ahead volatility predictions
- **Returns forecasting** - Multi-day ahead return predictions

## Tech Stack

- **FastAPI** - REST API framework
- **SQLite** - Local database for stock price data
- **arch** - GARCH model implementation
- **statsmodels** - ARIMA model implementation
- **AlphaVantage API** - Stock price data source
- **joblib** - Model serialization

## Project Structure

```
stock-backend/
├── main.py           # FastAPI application & endpoints
├── model.py         # GarchModel & ARIMA model classes
├── data.py         # Data fetching & SQL repository
├── config.py       # Settings & environment config
├── requirements.txt
├── models/         # Saved model files (*.pkl)
├── stocks.sqlite   # SQLite database
└── .env          # Environment variables
```

## API Endpoints

### POST /fit
Train GARCH and ARIMA models for a stock.

```json
{
  "ticker": "AAPL",
  "use_new_data": true,
  "n_observations": 1000,
  "p": 1,
  "q": 1,
  "arima_order": [1, 0, 1]
}
```

**Parameters:**
- `ticker` - Stock symbol (e.g., AAPL, MSFT)
- `use_new_data` - Fetch fresh data from AlphaVantage
- `n_observations` - Number of days to use (50-5000)
- `p` - GARCH ARCH lag order (1-10)
- `q` - GARCH lag order (1-10)
- `arima_order` - ARIMA (p, d, q) order (default: [1, 0, 1])

**Response:**
```json
{
  "success": true,
  "message": "Trained and saved 'models/2026-04-20_AAPL.pkl'. Metrics: AIC -1234.56, BIC -1225.00."
}
```

### POST /predict
Generate volatility or returns forecast.

```json
{
  "ticker": "AAPL",
  "n_days": 10,
  "predict_type": "volatility"
}
```

**Parameters:**
- `ticker` - Stock symbol
- `n_days` - Forecast horizon (1-30 days)
- `predict_type` - "volatility" or "returns"

**Response:**
```json
{
  "success": true,
  "forecast": {
    "2026-04-21": 1.2345,
    "2026-04-22": 1.2567,
    ...
  },
  "message": ""
}
```

### GET /api/system/status
Get system and model status.

**Response:**
```json
{
  "api": {"status": "online", "latency": 12.5},
  "model": {"status": "ready", "modelType": "GARCH for AAPL"},
  "engine": {"status": "running", "version": "1.0.0"}
}
```

### GET /health
Health check endpoint.

## Configuration

Create a `.env` file:

```env
alpha_api_key=YOUR_ALPHAVANTAGE_API_KEY
db_name=stocks.sqlite
model_directory=models
```

## Model Details

### GARCH (p, q)
- **p** - ARCH lag order (autoregressive on squared returns)
- **q** - GARCH lag order (lagged variance)
- Captures volatility clustering in returns

### ARIMA (p, d, q)
- **p** - Autoregressive lags
- **d** - Differencing order
- **q** - Moving average lags

**Returns Calculation:**
```python
df['return'] = df['close'].pct_change() * 100
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
uvicorn main:app --reload --port 8000
```

## Running on Render

1. Set environment variables in Render dashboard:
   - `alpha_api_key`
   - `db_name`
   - `model_directory`

2. Start command:
   ```
   gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker
   ```

## Model Files

Models are saved in `models/` directory:
- `*_garch.pkl` - GARCH model
- `*_arima.pkl` - ARIMA model

## Interpreting Results

### Volatility Forecast
- Values represent **expected standard deviation** of daily returns (%)
- Higher values = more volatile/unpredictable stock

### Returns Forecast  
- Values represent **expected daily return** (%)
- Positive = expected gain
- Negative = expected loss
- These are point estimates, not guarantees

## License

MIT