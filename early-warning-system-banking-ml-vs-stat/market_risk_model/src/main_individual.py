from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

origins = [
    "http://localhost:4200",  # Angular dev server
    "http://127.0.0.1:4200",
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
with open("stacked.pkl", "rb") as f:
    model = pickle.load(f)

# Enhanced Preprocessing function (from saved preprocessing code)
def preprocess_ohlcv(df, dataset_type='stock', weights=None, epsilon=1e-6):
    if dataset_type == 'stock':
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError("Single stock DataFrame must contain: Date, Open, High, Low, Close, Volume")
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[expected_cols].copy()
    else:
        if weights is None:
            raise ValueError("Weights must be provided for portfolio dataset")
        n_stocks = len(weights)
        if abs(sum(weights) - 1) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1")
        expected_cols = ['Date'] + [f'{col}_{i}' for i in range(n_stocks) for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Portfolio DataFrame must contain: Date, Open_0, High_0, ..., Volume_{n_stocks-1}")
        ohlcv_cols = [col for col in df.columns if col != 'Date']
        df = df[expected_cols].copy()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.interpolate(method='time', limit_direction='both')
    else:
        df = df.interpolate(method='linear', limit_direction='both')

    def remove_iqr_outliers(data, cols):
        Q1 = data[cols].quantile(0.25)
        Q3 = data[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = ((data[cols] >= lower_bound) & (data[cols] <= upper_bound)).all(axis=1)
        return data[mask]

    if dataset_type == 'stock':
        df = remove_iqr_outliers(df, ohlcv_cols)
    else:
        mask = None
        for i in range(len(weights)):
            stock_cols = [f'{col}_{i}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
            stock_mask = remove_iqr_outliers(df, stock_cols).index
            mask = stock_mask if mask is None else mask.intersection(stock_mask)
        df = df.loc[mask]

    scaler = MinMaxScaler()
    df[ohlcv_cols] = scaler.fit_transform(df[ohlcv_cols])

    processed_df = df.copy()
    epsilon = 1e-8

    if dataset_type == 'stock':
        processed_df['log_return'] = np.log((processed_df['Close'] + epsilon) / (processed_df['Close'].shift(1) + epsilon))
        processed_df['volatility_20d'] = processed_df['log_return'].rolling(window=20).std()
        processed_df['sma_50'] = processed_df['Close'].rolling(window=50).mean()
    else:
        for i in range(len(weights)):
            close_col = f'Close_{i}'
            log_return_col = f'{i}_log_return'
            volatility_col = f'{i}_volatility_20d'
            sma_col = f'{i}_sma_50'

            processed_df[log_return_col] = np.log((processed_df[close_col] + epsilon) / (processed_df[close_col].shift(1) + epsilon))
            processed_df[volatility_col] = processed_df[log_return_col].rolling(window=20).std()
            processed_df[sma_col] = processed_df[close_col].rolling(window=50).mean()

        processed_df['portfolio_return'] = sum(
            weights[i] * processed_df[f'{i}_log_return'] for i in range(len(weights))
        )
        processed_df['portfolio_volatility_30d'] = processed_df['portfolio_return'].rolling(window=30).std()
        processed_df['portfolio_sma_50'] = sum(
            weights[i] * processed_df[f'{i}_sma_50'] for i in range(len(weights))
        )

    processed_df = processed_df.dropna()
    return processed_df

# API endpoint
@app.post("/predict-individual-stock/")
async def predict_individual_stock(
    file: UploadFile = File(...),
     portfolio_value: float = Form(1_000_000)  # default to ₹1,000,000 if not provided
):
    try:
        contents = await file.read()
        s = contents.decode('utf-8')
        df = pd.read_csv(StringIO(s))

        # Preprocess
        df_processed = preprocess_ohlcv(df)

        # Predict
        predictions = model.predict(df_processed[['log_return', 'volatility_20d', 'sma_50']])
        predicted_log_returns = np.array(predictions) / 100

        var_days = [30, 60, 90]
        confidence_level = 5
        historical_vars = {}
        historical_ess = {}
        mc_vars = {}
        mc_ess = {}
        mu = np.mean(predicted_log_returns)
        sigma = np.std(predicted_log_returns)
        simulations = 10000

        for days in var_days:
            # Historical VaR and ES
            scaled_returns = predicted_log_returns * np.sqrt(days)
            var_hist = np.percentile(scaled_returns, confidence_level)
            es_hist = scaled_returns[scaled_returns <= var_hist].mean()
            historical_vars[f"historicalVar{days}d"] = f"{abs(var_hist)*100:.2f}%"
            historical_ess[f"historicalEs{days}d"] = f"{abs(es_hist)*100:.2f}%"

            # Monte Carlo VaR and ES
            simulated_returns = np.random.normal(loc=mu, scale=sigma * np.sqrt(days), size=simulations)
            var_mc = np.percentile(simulated_returns, confidence_level)
            es_mc = simulated_returns[simulated_returns <= var_mc].mean()
            mc_vars[f"monteCarloVar{days}d"] = f"{abs(var_mc)*100:.2f}%"
            mc_ess[f"monteCarloEs{days}d"] = f"{abs(es_mc)*100:.2f}%"

        return {
            "message": "Prediction successful",
            **historical_vars,
            **historical_ess,
            **mc_vars,
            **mc_ess
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
