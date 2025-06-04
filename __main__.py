from typing import Annotated
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from dotenv import dotenv_values, load_dotenv
from services.random_forest_service import RandomForestService
import uvicorn
import sys
from services.api_service import ApiService
from services.random_forest_service import PredictData
from services.random_forest_service import Report

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
    model.train_model(api_service, "model")
    exit(0)

def serve():
    try:
        model = RandomForestService().load("model")
        print("Modèle chargé avec succès.")
    except ValueError as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit(1)

    app = FastAPI()

    BEARER_TOKEN = config.get("BEARER_TOKEN")
    if not BEARER_TOKEN:
        raise ValueError("Bearer token is not set or invalid in the .env file.")

    @app.get("/")
    def health_check(authorization: Annotated[str, Header()]):
        """
        Health check endpoint to verify if the service is running.
        Returns a 200 OK response if the service is healthy.
        """
        if authorization != f"Bearer {BEARER_TOKEN}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return JSONResponse({"status":"ok"}, status_code=200)

    @app.post("/predict", response_model=Report)
    def predict(body: PredictData, authorization: Annotated[str, Header()]):
        """
        Endpoint to make predictions using the trained model.
        Expects a JSON body with the input data for prediction.
        Returns the prediction result as a JSON response.
        """
        if authorization != f"Bearer {BEARER_TOKEN}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        try:
            prediction = model.predict(body)
            print(f"Prédiction générée : {prediction}")
            return JSONResponse(prediction, status_code=200, media_type="application/json")
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return JSONResponse({"error": str(e)}, status_code=500, media_type="application/json")
    uvicorn.run(app, port=5000, log_level="info")

if len(sys.argv) != 2:
    print_usage()

command = sys.argv[1].lower()
if command == "train":
    train_model()
elif command == "serve":
    serve()
else:
    print_usage()
