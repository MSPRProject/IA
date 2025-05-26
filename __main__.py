from fastapi import FastAPI
from fastapi.responses import Response
import requests
from dotenv import dotenv_values, load_dotenv
from tree.random_forest_service import RandomForestService
from tree.random_forest_service import get_training_data
import uvicorn
import sys

load_dotenv()

print("Chargement de l'application FastAPI...") 
app = FastAPI()

url = "http://localhost:8080/"
print(f"URL de base de l'API : {url}") 

config = dotenv_values(".env")
BEARER_TOKEN = config.get("BEARER_TOKEN")
if not BEARER_TOKEN:
    raise ValueError("Bearer token is not set or invalid in the .env file.")
print(f"Bearer Token chargé : {BEARER_TOKEN}") 

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}
print(f"En-têtes HTTP configurés : {headers}") 

if len(sys.argv) > 1:
    print(f"Argument reçu : {sys.argv[1]}") 
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "train":
            model = RandomForestService()
            accuracy = model.train_model_once()
            print(f"Précision du modèle : {accuracy}")
        else:
            print("Argument invalide. Utilisez 'train' pour entraîner le modèle.")
    else:
        print("Aucun argument fourni. Utilisez 'train' pour entraîner le modèle.")
else:
    print("Chargement du modèle RandomForestService...") 
    try:
        model = RandomForestService().load()
        print("Modèle chargé avec succès.")  
    except ValueError as e:
        print(f"Erreur lors du chargement du modèle : {e}") 

@app.get("/prompt")
def get_data(body: dict):
    print(f"Requête reçue avec le corps : {body}") 
    try:
        prediction = model.predict(body)
        print(f"Prédiction générée : {prediction}") 
        return Response(prediction, status_code=200, media_type="application/json")
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}") 
        return Response({"error": str(e)}, status_code=500, media_type="application/json")
