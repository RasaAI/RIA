# Credit Health Model

# Credit Risk Analysis

This project provides a comprehensive credit risk analysis solution using machine learning and AI. It exposes a FastAPI backend that takes loan application data, predicts the probability of default, calculates expected loss, and provides a human-readable interpretation of the results.

### Key Features

* **Advanced ML Modeling**: Utilizes three separate models to predict Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).
* **Expected Loss Calculation**: Combines the outputs of the ML models to calculate the overall Expected Loss (EL) for each application.
* **AI-Powered Interpretation**: Generates a natural language summary of the risk analysis for easier understanding.
* **High-Performance API**: Built with FastAPI for modern, fast, and asynchronous request handling.
* **Persistent Logging**: All prediction requests and results are logged to a MySQL database for auditing and further analysis.
* **Interactive Frontend**: Comes with an Angular frontend for easy interaction and file uploads.

### Tech Stack

* **Backend**: FastAPI, Uvicorn
* **Database**: MySQL, SQLAlchemy (ORM)
* **Data Science**: Pandas, NumPy, Scikit-learn
* **AI Integration**: OpenAI
* **Frontend**: Angular

---

### Getting Started

Follow these steps to set up and run the project locally.

#### **1. Prerequisites**

* Python 3.8+
* MySQL Server

#### **2. Installation & Setup**

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Database & User**:
    * **IMPORTANT**: Edit `grant_permissions.py` to match your MySQL root password.
    * Run the script to create the database and user. This requires root access to MySQL and only needs to be run once.
        ```bash
        python grant_permissions.py
        ```

5.  **Configure Environment**:
    * Create a `.env` file in the project root.
    * Add your database password and your OpenAI API key. This file is essential for the application to connect to its services.
        ```env
        # .env
        MYSQL_PASSWORD="mukunth"
        LLM_API_KEY="sk-your-secret-openai-key"
        ```

6.  **Create Database Tables**:
    * Run the `db_connection.py` script (formerly `test_db_connection.py`) to create the necessary tables in the database.
        ```bash
        python db_connection.py
        ```

---

### Usage

#### **1. Run the Backend Server**

Start the FastAPI application with `uvicorn`:

```bash
uvicorn app.main:app --reload
```

The API will now be available at `http://localhost:8000`.

#### **2. Use the API**

You can send requests to the prediction endpoint using any API client or the provided Angular frontend.

* **Endpoint**: `POST /api/v1/predict/batch_json`
* **Request Body**: A JSON object with an `applications` key containing a list of loan applications.

**Example with `curl`**:

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch_json" \
-H "Content-Type: application/json" \
-d '{
      "applications": [
        {
          "loan_amnt": 5000,
          "funded_amnt": 5000,
          "funded_amnt_inv": 4975,
          "term": "36 months",
          "int_rate": 10.65,
          "installment": 162.87,
          "grade": "B",
          "emp_length": 10,
          "home_ownership": "RENT",
          "annual_inc": 24000,
          "verification_status": "Verified",
          "dti": 27.65,
          "delinq_2yrs": 0,
          "inq_last_6mths": 1,
          "mths_since_last_delinq": null,
          "open_acc": 3,
          "pub_rec": 0,
          "revol_bal": 13648,
          "revol_util": 83.7,
          "total_acc": 9,
          "initial_list_status": "f",
          "out_prncp": 0,
          "total_pymnt": 0,
          "total_rec_prncp": 0,
          "total_rec_int": 0,
          "total_rec_late_fee": 0,
          "recoveries": 0,
          "collection_recovery_fee": 0,
          "last_pymnt_amnt": 0,
          "tot_coll_amt": 0,
          "tot_cur_bal": 13648,
          "total_rev_hi_lim": 16300,
          "mths_since_earliest_cr_line": 60,
          "purpose": "debt_consolidation"
        }
      ]
    }'
```

### Further Documentation

For more detailed information, please refer to the `docs/` directory:
* `docs/installation.md`: A step-by-step guide to setting up the project.
* `docs/api-documentation.md`: Detailed information about the API endpoints.
* `docs/user-guide.md`: A guide on how to use the application and understand its output.
