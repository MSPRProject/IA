import requests
from tqdm import tqdm

class ApiService:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_training_data(self):
        url = f"{self.base_url}/api/ai/trainingData"
        body = {"page": 0, "size": 1000, "sort": "date"}
        all_data = []

        response = requests.get(url, params=body)
        if response.status_code != 200:
            raise ValueError(f"[ApiService] Unexpected HTTP status while fetching {url}: {response.status_code}")

        data = response.json()
        content = data.get("content", [])

        if not content:
            raise ValueError("[ApiService] No content found in the response from the API.")

        all_data.extend(content)
        page_count = data.get("totalPages", 0)

        for i in tqdm(range(1, page_count)):
            body["page"] = i
            response = requests.get(url, params=body)

            if response.status_code != 200:
                raise ValueError(f"[ApiService] Unexpected HTTP status while fetching {url}: {response.status_code}")

            data = response.json()
            content = data.get("content", [])
            if not content:
                break

            all_data.extend(content)

        print(f"[ApiService] Fetched {len(all_data)} training data entries from the API.")
        return all_data
