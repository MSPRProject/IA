import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tree.data_sorting import DataSorting 
import joblib
import requests
import json
from tqdm import tqdm
from tree.get_data import get_all_countries, get_all_pandemics, get_all_infections, get_all_reports

def get_all_entities(url, embedded_key):
    params = {"page": 0, "size": 20}
    response = requests.get(url, params=params)
    data = response.json()
    embedded = data.get("_embedded", {})
    entities = embedded.get(embedded_key, [])
    page_info = data.get("page", {})
    total_pages = page_info.get("totalPages", 1)

    for page in tqdm(range(1, total_pages), desc=f"Récupération de {embedded_key}"):
        params["page"] = page
        response = requests.get(url, params=params)
        data = response.json()
        embedded = data.get("_embedded", {})
        page_entities = embedded.get(embedded_key, [])
        entities.extend(page_entities)

    print(f"Nombre total de {embedded_key} récupérés : {len(entities)}")
    return entities

def get_training_data():
    url = "http://localhost:8080/api/ai/trainingData"
    body = {"page": 0, "size": 1000, "sort": "date"}
    all_data = []
    
    response = requests.get(url, json=body)
    if response.status_code != 200:
        print(f"Erreur HTTP {response.status_code} lors de l'appel à {url}")
        return None
    
    data = response.json()
    content = data.get("content", [])
    
    if not content:
        print("Aucune donnée retournée.")
        return None
    
    all_data.extend(content)
    page_count = data.get("totalPages", 0)
    if page_count == 0:
        print("Aucune page de données disponible.")
        return None

    for i in tqdm(range(1, page_count)):
        body["page"] = i
        response = requests.get(url, json=body)

        if response.status_code != 200:
            print(f"Erreur HTTP {response.status_code} lors de l'appel à {url} pour la page {i}")
            break

        data = response.json()
        content = data.get("content", [])
        if not content:
            print(f"Aucune donnée retournée pour la page {i}. Arrêt.")
            break
        
        all_data.extend(content)

    if not all_data:
        print("Aucune donnée n'a été récupérée.")
        return None
    
    print("Données brutes récupérées (extrait) :")
    for i, item in enumerate(all_data[:5]):
        print(f"Item {i} : {json.dumps(item, indent=2, ensure_ascii=False)}")

    print(f"Nombre total d'éléments récupérés : {len(all_data)}")
    return {"content": all_data}

class RandomForestService:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.scaler = None
        self.is_trained = False

    @staticmethod
    def load():
        try:
            model = joblib.load("random_forest_model.pkl")
            scaler = joblib.load("scaler.pkl")
            return model, scaler
        except FileNotFoundError:
            raise ValueError("Le modèle ou le scaler n'a pas été trouvé. Veuillez entraîner le modèle d'abord.")

    def train_model_once(self):
        if self.is_trained:
            return self.accuracy

        print("Récupération des données paginées...")
        countries = get_all_countries()
        pandemics = get_all_pandemics()
        infections = get_all_infections()
        reports = get_all_reports()

        print("Formatage des données...")
        df = DataSorting.format_and_sort_data(countries, pandemics, infections, reports)

        if df.empty:
            raise ValueError("Les données formatées sont vides. Impossible d'entraîner le modèle.")

        print("Données formatées et triées :")
        print(df.head())

        print(f"Nombre de lignes avant suppression des doublons : {len(df)}")
        df = df.drop_duplicates()
        print(f"Nombre de lignes après suppression des doublons : {len(df)}")

        df["year"] = df["report_date"].dt.year
        X = df[["new_cases", "new_deaths", "year"]]
        y = df["pandemic_name"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if len(df) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X_scaled, y
            X_test, y_test = X_scaled, y

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.model = model
        self.is_trained = True

        joblib.dump(self.model, "random_forest_model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")

        print(f"Modèle entraîné avec une précision de {self.accuracy:.2f}")
        return self.accuracy
    
    def _format_data(self, data: dict) -> pd.DataFrame:
        reports = data.get("reports", [])
        infections = {i.get("country_iso3"): i for i in data.get("infections", []) if i.get("country_iso3")}
        pandemics = {p.get("name"): p for p in data.get("pandemics", []) if p.get("name")}
        countries = {c.get("iso3"): c for c in data.get("countries", []) if c.get("iso3")}
    
        print("Nombre de reports :", len(reports))
        print("Nombre d'infections :", len(infections))
        print("Nombre de pandémies :", len(pandemics))
        print("Nombre de pays :", len(countries))
        
    
        rows = []
        for report in reports:
            country_iso3 = report.get("country_iso3")
            if not country_iso3:
                # print(f"Report ignoré : pas de 'country_iso3' dans {report}")
                continue
    
            infection = infections.get(country_iso3)
            if not infection:
                print(f"Infection manquante pour le pays {country_iso3}")
                continue
    
            country = countries.get(country_iso3)
            if not country:
                print(f"Pays manquant pour le code {country_iso3}")
                continue
    
            pandemic = pandemics.get(infection.get("pandemic_name"))
            if not pandemic:
                print(f"Pandémie manquante pour {infection.get('pandemic_name')}")
                continue
    
            rows.append({
                "report_date": report.get("date"),
                "total_cases": infection.get("new_cases", 0),
                "total_deaths": infection.get("new_deaths", 0),
                "population": country.get("population", 0),
                "pandemic_name": pandemic.get("name", "Unknown")
            })
    
        if not rows:
            print("Aucune ligne valide n'a été créée.")
            raise ValueError("Aucune ligne de données formatées valides.")

        return pd.DataFrame(rows)
    