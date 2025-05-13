from fastapi import FastAPI
from fastapi.responses import Response
import requests
from dotenv import dotenv_values, load_dotenv
from tree.random_forest_service import RandomForestService
import uvicorn
import sys

load_dotenv()

app = FastAPI()

url = "http://localhost:8080/"

BEARER_TOKEN = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

if BEARER_TOKEN is None or BEARER_TOKEN == "":
    raise ValueError("Bearer token is not set in the .env file.")



if len(sys.argv) > 1:
    if sys.argv[1] == "train":
        try:
            model = RandomForestService()
            accuracy = model.train_model_once()
            print(f"Model accuracy: {accuracy}")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("Invalid argument. Use 'train' to train the model.")
else:
    model = RandomForestService().load()

@app.get("/prompt")
def get_data(body: dict):
    return Response(model.predict(body), status_code=200, media_type="application/json")
