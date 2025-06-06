import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine
import psycopg2

# Load environment variables
load_dotenv()

# ✅ PostgreSQL DB config from env
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ✅ OpenAI via LangChain
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# ✅ LangChain SQL
engine = create_engine(DATABASE_URL)
db = SQLDatabase(engine)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# ✅ FastAPI app
app = FastAPI()

# ✅ CORS (for Angular frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class ChatRequest(BaseModel):
    question: str

# ✅ Root route
@app.get("/")
def root():
    return {"message": "✅ Chatbot is running"}

# ✅ Chatbot Q&A route
@app.post("/chatbot")
def chatbot_response(req: ChatRequest):
    try:
        answer = db_chain.run(req.question)
        return {"response": answer}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}

# ✅ Recent Face Login Logs
@app.get("/face_logins")
def get_face_logins():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM face_logins ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        return {"logins": results}
    except Exception as e:
        return {"error": str(e)}

# ✅ Recent Animal Detections
@app.get("/detections")
def get_detections():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        return {"detections": results}
    except Exception as e:
        return {"error": str(e)}

# ✅ Recent File Uploads
@app.get("/uploads")
def get_uploads():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploads ORDER BY timestamp DESC LIMIT 10")
        results = cursor.fetchall()
        return {"uploads": results}
    except Exception as e:
        return {"error": str(e)}
