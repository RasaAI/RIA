import pandas as pd
from sqlalchemy import create_engine

# CSV file path
csv_path = "CJ.csv"

# PostgreSQL connection string
DATABASE_URL = "postgresql+psycopg2://postgres:surya78792@localhost:5433/Intern"

# Load CSV into DataFrame
df = pd.read_csv(csv_path)

# (Optional) Show first few rows
print("📄 Preview CSV data:\n", df.head())

# Replace NaN with None for SQL compatibility
df = df.where(pd.notnull(df), None)

# Create DB engine
engine = create_engine(DATABASE_URL, echo=True)  # echo=True to show SQL statements

# Insert data into table (append mode)
try:
    df.to_sql('liquidity', con=engine, index=False, if_exists='append')
    print("✅ Data inserted successfully into 'liquidity' table.")
except Exception as e:
    print("❌ Error inserting data:", e)
