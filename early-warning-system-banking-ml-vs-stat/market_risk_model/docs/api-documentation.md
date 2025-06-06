# API Documentation

## Base URL

```
origins = [
    "http://localhost:4200",  # Angular dev server
    "http://127.0.0.1:4200",
]
```

## Endpoints

### 1. Predict Market Risk (Individual)

- `POST /predict/individual`
- Request JSON:
```json
{
  "ticker": "AAPL",
  "features": [...]
}
```
- Response:
```json
{
 {
  "message": "Prediction successful",
  "historical_var_30d": "1.87%",
  "historical_var_60d": "2.64%",
  "historical_var_90d": "3.24%",
  "monte_carlo_var_30d": "6.18%",
  "monte_carlo_var_60d": "1.08%",
  "monte_carlo_var_90d": "1.44%"
}
}
```

### 2. Predict Portfolio Risk

- `POST /predict/portfolio`
- Request JSON:
```json
{
  "portfolio": ["AAPL", "GOOG", "MSFT","AVGO","BKL","NVDA","TSLA","VTV","VUG"],
  "weights": [0.02, 0.3, 0.2, 0.05, 0.14, 0.16, 0.11, 0.02]
}
```
- Response:
```json
{
 
  "message": "Prediction successful",
  "historicalVar30d": "2.64%",
  "historicalEs30d": "2.64%",
  "monteCarloVar30d": "4.39%",
  "monteCarloEs30d": "5.44%",
  "historicalVar60d": "3.73%",
  "historicalEs60d": "3.73%",
  "monteCarloVar60d": "6.06%",
  "monteCarloEs60d": "7.75%",
  "historicalVar90d": "4.57%",
  "historicalEs90d": "4.57%",
  "monteCarloVar90d": "7.31%",
  "monteCarloEs90d": "9.21%"

}
```