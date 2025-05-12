import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
BEARER_TOKEN = config.get("BEARER_TOKEN")
BASE_URL = "http://localhost:8081" 

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

def get_all_data():
    try:
        response = requests.get(f"{BASE_URL}/prompt", headers=headers)
        response.raise_for_status()
        data = response.json()
        # print("Données complètes retournées par get_all_data() :", data)
        return data
    except requests.HTTPError as e:
        print(f"Erreur HTTP lors de l'appel à FastAPI : {e}")
        return None
    except requests.RequestException as e:
        print(f"Erreur lors de l'appel à FastAPI : {e}")
        return None
