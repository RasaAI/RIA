import os
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Load environment variables from .env
load_dotenv()

# Load DB and OpenAI credentials from env variables
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Setup SQLAlchemy engine and LangChain
engine = create_engine(DATABASE_URL)
db = SQLDatabase(engine)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

app = FastAPI()

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "API is running."}

@app.post("/chatbot")
def chatbot_response(req: ChatRequest):
    try:
        response = db_chain.run(req.question)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/face_logins")
def get_face_logins():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM face_logins ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        conn.close()
        return {"face_logins": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/detections")
def get_detections():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        conn.close()
        return {"detections": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/uploads")
def get_uploads():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploads ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        conn.close()
        return {"uploads": results}
    except Exception as e:
        return {"error": str(e)}
