from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd

# --- 1. Database Connection ---
DATABASE_URL = "postgresql+psycopg2://postgres:surya78792@localhost:5433/Intern"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# --- 2. Define SQLAlchemy Model ---
class User(Base):
    __tablename__ = 'liquidity'

    id = Column(Integer, primary_key=True, autoincrement=True)
    MLA = Column(Float)
    EWAQ_Capital = Column(Float)
    EWAQ_GrossLoans = Column(Float)
    CURR_ACC = Column("01_CURR_ACC", Float)
    XX_MLA = Column(Float)
    XX_TOTAL_LIQUID_ASSET = Column(Float)
    SAVINGS = Column("03_SAVINGS", Float)
    TIME_DEPOSIT = Column("02_TIME_DEPOSIT", Float)
    F077_ASSETS_TOTAL = Column(Float)
    EWAQ_NPL = Column(Float)
    XX_BAL_IN_OTHER_BANKS = Column(Float)
    FOREIGN_DEPOSITS_AND_BORROWINGS = Column("10_FOREIGN_DEPOSITS_AND_BORROWINGS", Float)
    EWAQ_NPLsNetOfProvisions = Column(Float)
    LR = Column(Float)
    F125_LIAB_TOTAL = Column(Float)
    BANKS_ABROAD = Column("19_BANKS_ABROAD", Float)
    INF = Column(Float)
    XX_BOT_BALANCE = Column(Float)
    EWAQ_NPLsNetOfProvisions2CoreCapital = Column(Float)
    DR = Column(Float)

# --- 3. Drop and Recreate Tables (Only for Debugging) ---
# ⚠️ WARNING: This deletes data each time you run the script.
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# --- 4. Insert Function with Debug Statements ---
def insert_df_to_db(df: pd.DataFrame) -> int:
    print("🔍 Original DataFrame:")
    print(df.head())

    # Skip the first row
    df = df.iloc[1:]
    print("📉 After skipping first row:")
    print(df.head())

    # Replace NaN with None
    df = df.where(pd.notnull(df), None)

    # Create DB session
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Convert DataFrame to SQLAlchemy objects
        records = [User(**row.to_dict()) for _, row in df.iterrows()]
        print(f"📦 Prepared {len(records)} records for insertion.")

        # Insert into DB
        session.bulk_save_objects(records)
        session.commit()
        print("✅ Commit successful.")

        # Confirm insertion
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM liquidity")
            count = result.scalar()
            print(f"🧾 Rows in 'liquidity' table: {count}")

        return len(records)

    except Exception as e:
        print("❌ Error during insertion:", e)
        session.rollback()
        return 0
    finally:
        session.close()

# --- 5. Test Insert (Optional) ---
if __name__ == "__main__":
    try:
        df = pd.read_csv("your_file.csv")  # Change to your file path
        inserted = insert_df_to_db(df)
        print(f"✅ Total rows inserted: {inserted}")
    except Exception as e:
        print("❌ Failed to load CSV or insert:", e)
