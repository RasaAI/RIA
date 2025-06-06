from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib  # or pickle
import json
import io
from fastapi.middleware.cors import CORSMiddleware
from Database import insert_df_to_db

app = FastAPI()
origins = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ["*"] to allow all (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],    # GET, POST, PUT, etc.
    allow_headers=["*"],    # Authorization, Content-Type, etc.
)

# Load your trained model once when the app startsa
model = joblib.load("model.pkl")  # replace with your model file path

# Define your mapping function
def map_risk(rating: int) -> str:
    if rating == 0:
        data = {
            "predict" : "No Risk",
            "reasons":  [
            "Very high liquid assets",
            "Low or zero non-performing loans (NPLs)",
            "High stable deposits",
            "Low short-term liabilities"]
        }
        json_string = json.dumps(data)
        return data
    elif rating == 1:
        data = {
            "predict": "Low Risk",
            "reasons": [
            "High liquid assets",
            "Low NPLs",
            "Good deposit base",
            "Manageable liabilities"],
        }
        json_string = json.dumps(data)
        return data
    elif rating == 2:
        data =  {
            "predict": "Medium Risk",
            "reasons": [
            "Adequate but not strong liquidity",
            "Some NPLs",
            "Moderate deposit base",
            "Moderate liabilities"]}
        json_string = json.dumps(data)
        return data
    elif rating == 3:
        data =  {
             "predict": "Moderately High Risk",
            "reasons": [
            "Lower liquidity ratios",
            "Growing NPLs",
            "Less stable deposits",
            "Higher short-term liabilities"
        ]}
        json_string = json.dumps(data)
        return data
    elif rating == 4:
        data = {
            "predict": "High Risk",
            "reasons": [
            "Low liquid assets",
            "High NPLs",
            "Unstable or declining deposits",
            "High liabilities"]}
        json_string = json.dumps(data)
        return data
    elif rating == 5:
        data =  {
            "predict": "Very High Risk",
            "reasons": [
            "Critically low liquidity",
            "Very high NPLs",
            "Major deposit outflows",
            "Excessive liabilities"]}
        json_string = json.dumps(data)
        print("json_string", json_string)
        return data 
    return "Unknown"


# #class FeaturesInput(BaseModel):
#     01_CURR_ACC: float
#     02_TIME_DEPOSIT: float
    

@app.post("/upload-portfolio")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        # Read CSV file into DataFrame
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        insert_df_to_db(df)
        predictions = model.predict(df)

        # Map predictions to risk labels
        risk_labels = map_risk(predictions)

        # Combine into output
        # response = [
        #     {"index": i, "liquidity_rating": int(pred), "risk_level": risk}
        #     for i, (pred, risk) in enumerate(zip(predictions, risk_labels))
        # ]
        print("risk_labels", risk_labels)

        return JSONResponse(content=risk_labels)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)