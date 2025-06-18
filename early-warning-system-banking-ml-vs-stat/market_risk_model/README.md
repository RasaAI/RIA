# Market Risk Analysis Model

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Angular](https://img.shields.io/badge/Angular-Frontend-red.svg)](https://angular.io/)
[![Hackathon Project](https://img.shields.io/badge/Status-Hackathon-blueviolet.svg)]()

A hybrid deep learning-based Market Risk Analysis system to predict **Value at Risk (VaR)** and **Expected Shortfall (ES)** for **individual stocks** and **investment portfolios** over 30, 60, and 90-day horizons.

---

## 🚀 Overview

This project was developed during a hackathon to provide internal risk analysis capabilities for financial firms. It allows users to:

- Upload their own stock/portfolio data in CSV format.
- Choose between **individual** or **portfolio** risk analysis.
- Receive predicted VaR and ES values based on trained models.

---

## ✨ Features

- 📈 Predict market risk for individual stocks or portfolios.
- 🧠 Ensemble models for individuals; GAT (Graph Attention Network) for portfolios.
- 🔁 Real-time CSV upload and inference through a simple Angular UI.
- 📊 Performance metrics: RMSE, MAE, R².

---

## 🧰 Tech Stack

**Frontend:** Angular  
**Backend:** FastAPI  
**ML/Processing:** Python, PyTorch, Scikit-learn, Keras, NumPy, Pandas  
**Data:** yFinance (for historical simulation), user-uploaded CSVs

---

## 📁 Folder Structure

```

market\_risk\_model/
├── requirements.txt
├── README.md
├── assets/
│   └── images/                 # Graphs of model performance (Actual vs Predicted)
├── data/
│   ├── training/               # Simulated training data
│   │   ├── individual\_simulated.csv
│   │   └── portfolio\_simulated.csv
│   └── import\_real\_data.py     # Fetch real-time data
├── docs/
│   ├── API documentation/
│   ├── installation.md
│   └── userguide.md
├── src/
│   ├── frontend/               # Angular frontend code
│   └── ml models/             # Python ML scripts and trained model pickle files
│       ├── individual\_preprocessing.py
│       ├── individual\_ensemble.py
│       ├── portfolio\_preprocessing.py
│       ├── portfolio\_GAT.py
│       ├── individual\_ensemble.pkl
│       └── portfolio\_GAT.pkl

````

---

## ⚙️ Getting Started

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/market_risk_model.git
   cd market_risk_model
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run FastAPI backend:

   ```bash
   uvicorn src.main:app --reload
   ```

---

### Frontend Setup (Angular)

Navigate to the frontend directory:

```bash
cd src/frontend
```

Run the Angular development server:

```bash
ng serve
```

Once the server is running, open your browser and go to:

```
http://localhost:4200/
```

---

## 🧪 How to Use

1. Choose **Individual** or **Portfolio** analysis tab.
2. Upload a valid `.csv` file with your stock or portfolio data.
3. For portfolios, also input stock **weightages**.
4. Click **Generate** — the model returns predictions for VaR and ES for 30, 60, and 90 days.

---

## 🧠 Model Details

### Individual Stock Risk:

* Model: **Stacked Ensemble**
* Components: LSTM-Attention, Transformer, 1D CNN-LSTM
* Meta Learner: Regression model
* Evaluation: RMSE, MAE, R²

### Portfolio Risk:

* Model: **Graph Attention Network (GAT)**
* Evaluation: RMSE, MAE, R²

---

## 📊 Data

* Simulated data generated from real stock data to improve training robustness.
* Real-time stock data pulled using `yfinance` (if needed).
* Inference is done on user-uploaded CSVs via frontend UI.

---

## 📸 Screenshots (Optional)

You can view the result graphs and UI previews in the [`assets/images`](./assets/images) folder.

---

## 👨‍👩‍👧‍👦 Credits

Built by:

* **Adza Rajasekhar E A**
* **Infant Anto D**
* **Lourdu Virgin Maria P**

**Mentor:** Santhosh P

**References:**
📚 *Advances in Financial Machine Learning* by Marcos López de Prado

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgements

Thanks to the organizing team Rasa Institute of Analytics and Rasa.Ai Labs of the hackathon for the opportunity, and the contributors of open-source tools and research that made this project possible.

---
