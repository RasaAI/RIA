# Market Risk Analysis System

This repository hosts a professional, dual-level risk forecasting system for individual stocks and investment portfolios using both statistical and machine learning techniques. The objective is to provide early and accurate predictions of Value at Risk (VaR) and Expected Shortfall (ES) using advanced models and a clean deployment stack.

## 📌 Project Objectives

- Predict market risk at individual stock and portfolio levels
- Use statistical models (ARIMA-GARCH, DCC-GARCH) and ML (ensemble models, GAT)
- Simulated stock data (OHLCV) for controlled experimentation and stress testing
- Integration with Angular (frontend) and FastAPI (backend)

## 🧱 System Architecture

- **Frontend:** Angular (Client UI)
- **Backend:** FastAPI (Model Serving API)
- **Models:** Python-based forecasting models (deployed via API)
- **Notebooks:** Initial prototype modeling (converted into scripts)
- **Deployment:** Ready for cloud/containers via FastAPI endpoints

## 🧠 Core Technologies

- Python, Pandas, NumPy, Scikit-Learn, Statsmodels, PyTorch
- ARIMA-GARCH, DCC-GARCH, LSTM, CNN-LSTM, GAT
- FastAPI, Angular
- Jupyter, VS Code

## 🧾 Directory Structure

```
├── README.md
├── docs/
├── src/
│   ├── frontend/
│   ├── backend/
│   ├── ml_models/
│   │   ├── preprocessing.py
│   │   ├── individual.py
│   │   ├── portfolio.py
│   │   └── utils.py
├── notebooks/
├── assets/
├── deployment/
├── tests/
├── requirements.txt
└── .gitignore
```

## 📈 Modeling Highlights

- **Statistical Modeling**: ARIMA-GARCH (individual), DCC-GARCH (portfolio)
- **Machine Learning**: Ensemble models (stacking), GAT for graph-based portfolio modeling
- **Metrics**: RMSE, MAPE, R²; Kupiec & Christoffersen tests for VaR and Acerbic-Szekely and Ridge tests for ES risk validation
- **Cross-Validation**: Embargo method to prevent leakage

## 📦 Installation

```bash
git clone https://github.com/your-username/market-risk-analysis.git
cd market-risk-analysis
pip install -r requirements.txt
```

## 🚀 Running the System

1. Launch backend (FastAPI)
2. Start frontend (Angular)
3. Access UI and test predictions

## 🧪 Testing

All model and API tests are inside the `/tests/` folder. Run using:

```bash
pytest tests/
```

## 📚 Documentation

- Installation steps: `/docs/installation.md`
- API endpoints: `/docs/api-documentation.md`
- How to use system: `/docs/user-guide.md`
