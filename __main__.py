from fastapi import FastAPI
from fastapi.responses import Response
import requests
from dotenv import dotenv_values, load_dotenv
from tree.random_forest_service import RandomForestService
import uvicorn
import sys
from services.api_service import ApiService

load_dotenv()
config = dotenv_values(".env")

def print_usage():
    print(f"Usage: {sys.argv[0]} train/serve")
    exit(1)

def train_model():
    API_BASE_URL = config.get("API_BASE_URL")
    if not API_BASE_URL:
        raise ValueError("API_BASE_URL is not set or invalid in the .env file.")

    model = RandomForestService()
    api_service = ApiService(API_BASE_URL)
    accuracy = model.train_model_once(api_service)
    print(f"Précision du modèle : {accuracy}")
    exit(0)

def serve():
    try:
        model = RandomForestService().load()
        print("Modèle chargé avec succès.")  
    except ValueError as e:
        print(f"Erreur lors du chargement du modèle : {e}") 
        exit(1)

    app = FastAPI()

    BEARER_TOKEN = config.get("BEARER_TOKEN")
    if not BEARER_TOKEN:
        raise ValueError("Bearer token is not set or invalid in the .env file.")

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

    @app.get("/")
    def health_check():
        return Response("Ok", status_code=200)

    @app.get("/predict")
    def predict(body: dict):
        try:
            prediction = model.predict(body)
            print(f"Prédiction générée : {prediction}") 
            return Response(prediction, status_code=200, media_type="application/json")
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}") 
            return Response({"error": str(e)}, status_code=500, media_type="application/json")

if len(sys.argv) != 2:
    print_usage()

command = sys.argv[1].lower()
if command == "train":
    train_model()
elif command == "serve":
    serve()
else:
    print_usage()

