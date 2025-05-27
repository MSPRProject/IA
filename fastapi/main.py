from fastapi import FastAPI
from fastapi.responses import Response
import requests
from dotenv import dotenv_values, load_dotenv
from fastapi import Query
from fastapi import Body
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "random_forest_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

load_dotenv()

app = FastAPI()

url = "http://localhost:8080"

BEARER_TOKEN = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

if BEARER_TOKEN is None or BEARER_TOKEN == "":
    raise ValueError("Bearer token is not set in the .env file.")

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(new_cases: int, new_deaths: int, year: int):
    X = pd.DataFrame([[new_cases, new_deaths, year]], columns=["new_cases", "new_deaths", "year"])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return {"prediction": prediction[0]}
