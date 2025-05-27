import requests
from tqdm import tqdm
import json

class ApiService:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_training_data(self):
        url = f"{self.base_url}/api/ai/trainingData"
        body = {"page": 0, "size": 1000, "sort": "date"}
        all_data = []
        
        response = requests.get(url, json=body)
        if response.status_code != 200:
            raise ValueError(f"Erreur HTTP {response.status_code} lors de l'appel à {url}")
        
        data = response.json()
        content = data.get("content", [])
        
        if not content:
            raise EOFError("Aucune donnée retournée par l'API")
        
        all_data.extend(content)
        page_count = data.get("totalPages", 0)

        for i in tqdm(range(1, page_count)):
            body["page"] = i
            response = requests.get(url, json=body)

            if response.status_code != 200:
                raise ValueError(f"Erreur HTTP {response.status_code} lors de l'appel à {url}")

            data = response.json()
            content = data.get("content", [])
            if not content:
                break
            
            all_data.extend(content)

        print("Données brutes récupérées (extrait) :")
        for i, item in enumerate(all_data[:5]):
            print(f"Item {i} : {json.dumps(item, indent=2, ensure_ascii=False)}")

        print(f"Nombre total d'éléments récupérés : {len(all_data)}")
        return all_data
