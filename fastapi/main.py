from fastapi import FastAPI
from fastapi.responses import Response
import requests
from dotenv import dotenv_values, load_dotenv

load_dotenv()

app = FastAPI()

url = "http://127.0.0.1:8080"

BEARER_TOKEN = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

if BEARER_TOKEN is None or BEARER_TOKEN == "":
    raise ValueError("Bearer token is not set in the .env file.")

@app.get("/prompt")
def get_data(body: dict):
    print(body)
