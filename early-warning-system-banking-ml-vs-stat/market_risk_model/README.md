# Market Risk Analysis Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Angular](https://img.shields.io/badge/Angular-Frontend-red.svg)](https://angular.io/)

A hybrid deep learning-based system to predict **Value at Risk (VaR)** and **Expected Shortfall (ES)** for individual stocks and investment portfolios over 30, 60, and 90-day horizons.

---

##  Overview

This project provides a user-friendly Angular-based UI with a FastAPI backend and Python-based machine learning models for market risk analysis. It supports individual stock risk assessment using ensemble models and portfolio risk analysis using Graph Attention Networks (GAT). Users can upload CSV data for real-time predictions.

---

##  Features

-  Predict VaR and ES for individual stocks or portfolios.
-  Ensemble models (LSTM-Attention, Transformer, 1D CNN-LSTM) for stocks; GAT for portfolios.
-  Real-time CSV upload and inference via Angular UI.
-  Performance metrics: RMSE, MAE, R².

---

##  Tech Stack

- **Frontend**: Angular
- **Backend**: FastAPI
- **ML/Processing**: Python, PyTorch, Scikit-learn, Keras, NumPy, Pandas
- **Data**: yFinance (historical data), user-uploaded CSVs

---

##  Folder Structure

```
market_risk_model/
├── assets/
├── data/
├── docs/
├── src/
│   └── frontend/
│       ├── public/
│       ├── src/
├── ml models/
│   ├── individual_ensemble.py
│   ├── individual_preprocessing.py
│   ├── portfolio_GAT.py
│   ├── portfolio_preprocessing.py
│   ├── individual_GAT.pkl
│   ├── portfolio_GAT_model.pkl
│   ├── main_individual.py
│   ├── main_portfolio.py
├── README.md
└── requirements.txt
```

> For a detailed file breakdown, refer to `docs/project-structure.md`.

---

##  Installation

### Prerequisites
- Python 3.9+ (`python --version` to check)
- Node.js 18+ (`node --version` to check)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/market_risk_model.git
cd market_risk_model
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
Install Python dependencies:
```bash
cd "ml models"
pip install -r requirements.txt
cd ..
```

Install Angular frontend dependencies:
```bash
cd src/frontend
npm install
cd ../..
```

---

##  Usage

### 1. Run the Backend (FastAPI)
Start the FastAPI server for API requests and ML inference:
```bash
cd "ml models"
python main_individual.py  # For individual stock analysis
```
Or:
```bash
python main_portfolio.py  # For portfolio analysis
```
- The server runs on `http://0.0.0.0:8000`.
- See `docs/api-documentation.md` for API endpoints.

### 2. Launch the Frontend (Angular)
Start the Angular development server:
```bash
cd src/frontend
ng serve
```
- Open `http://localhost:4200/` in your browser to access the UI.

### 3. Using the Application
- Upload a CSV file with stock or portfolio data via the Angular UI.
- Select **Individual** or **Portfolio** analysis.
- For portfolios, input stock weightages if prompted.
- Click **Generate** to view VaR and ES predictions for 30, 60, and 90 days.

---

##  How It Works

- **Data Processing**: [`import_real_data.py`](ml%20models/import_real_data.py) fetches or processes user-uploaded CSV data.
- **Model Training**: Preprocessing ([`individual_preprocessing.py`](ml%20models/individual_preprocessing.py), [`portfolio_preprocessing.py`](ml%20models/portfolio_preprocessing.py)) and training ([`individual_ensemble.py`](ml%20models/individual_ensemble.py), [`portfolio_GAT.py`](ml%20models/portfolio_GAT.py)) generate pickle files (`individual_GAT.pkl`, `portfolio_GAT_model.pkl`).
- **Inference**: [`main_individual.py`](ml%20models/main_individual.py) and [`main_portfolio.py`](ml%20models/main_portfolio.py) handle real-time predictions via FastAPI.
- **UI Integration**: The Angular frontend (`src/frontend`) communicates with the FastAPI API to display results.

---

##  Model Details

### Individual Stock Risk
- **Model**: Stacked Ensemble (LSTM-Attention, Transformer, 1D CNN-LSTM)
- **Meta Learner**: Regression model
- **Metrics**: RMSE, MAE, R²

### Portfolio Risk
- **Model**: Graph Attention Network (GAT)
- **Metrics**: RMSE, MAE, R²

---

##  Data
- Simulated data for training robustness.
- Real-time stock data via yFinance or user-uploaded CSVs.
- Inference data processed through the UI.

---

##  Troubleshooting

- **CORS Errors**: If the frontend can’t connect to the backend, ensure CORS is enabled in FastAPI (check `docs/api-documentation.md`).
- **Angular Build Fails**: Run `npm cache clean --force && npm install` in `src/frontend`.
- **Dependency Issues**: Verify Python 3.9+ and Node.js 18+ are installed, and re-run `pip install -r requirements.txt`.

---

##  Notes
- Refer to `docs/installation.md` for detailed setup steps.
- See `docs/user-guide.md` for UI usage instructions.
- Ensure all dependencies are installed to avoid runtime errors.
```
