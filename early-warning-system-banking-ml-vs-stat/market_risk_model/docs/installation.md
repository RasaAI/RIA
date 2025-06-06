# Installation Guide

Follow these steps to set up the Market Risk Analysis System locally.

## Requirements

- Python 3.8+
- Node.js + Angular CLI (for frontend)
- pip (Python package manager)

## Backend Setup

```bash
git clone https://github.com/RasaAI/RIA/tree/main/early-warning-system-banking-ml-vs-stat/market_risk_model
cd market-risk-analysis
pip install -r requirements.txt
```

## Frontend Setup

```bash
cd src/frontend
npm install
ng serve
```

## Run the Backend Server

```bash
cd src/backend
uvicorn main:app --reload
```