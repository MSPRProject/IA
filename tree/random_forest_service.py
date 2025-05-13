import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
import joblib
import requests

def get_training_data():
    url = "http://localhost:8080/api/ai/trainingData"
    body = { "page": 0, "size": 1000, "sort": "date" }
    
    data = []
    for i in range(1, 100):
        response = requests.get(url, json=body)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            break

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

        data = get_training_data()
                
        if data is None:
            raise ValueError("Impossible de récupérer les données via l'API FastAPI.")

        if not isinstance(data, dict):
            try:
                print
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Les données reçues ne sont pas au format JSON valide.")

        required_keys = ["reports", "infections", "pandemics", "countries"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Les données sont manquantes pour les clés : {', '.join(missing_keys)}")

        df = self._format_data(data)

        df["report_date"] = pd.to_datetime(df["report_date"], errors='coerce')
        df["year"] = df["report_date"].dt.year

        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        df["total_cases"] = pd.to_numeric(df["total_cases"], errors="coerce")
        df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce")

        df.dropna(subset=["total_cases", "total_deaths", "population", "year", "pandemic_name"], inplace=True)

        if len(df) < 2:
            raise ValueError("Pas assez de données après nettoyage pour entraîner le modèle.")

        print("Valeurs manquantes par colonne :\n", df.isnull().sum())

        X = df[["total_cases", "total_deaths", "population", "year"]]
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

        return self.accuracy

    def predict(self, data: dict):
        if not self.is_trained:
            raise ValueError("Le modèle n'a pas encore été entraîné.")

        required_keys = ["total_cases", "total_deaths", "population", "year"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Données manquantes : {', '.join(missing_keys)}")

        df = pd.DataFrame([data])
        df = df[["total_cases", "total_deaths", "population", "year"]]

        df_scaled = self.scaler.transform(df)

        return self.model.predict(df_scaled)[0]

    def _format_data(self, data: dict) -> pd.DataFrame:
        reports = data.get("reports", [])
        infections = {i.get("id"): i for i in data.get("infections", [])}
        pandemics = {p.get("id"): p for p in data.get("pandemics", [])}
        countries = {c.get("id"): c for c in data.get("countries", [])}

        print("Nombre de reports :", len(reports))
        print("Nombre d'infections :", len(infections))
        print("Nombre de pandémies :", len(pandemics))
        print("Nombre de pays :", len(countries))

        rows = []
        for report in reports:
            infection = infections.get(report.get("infection_id"))
            if not infection:
                print(f"Infection manquante pour le report ID {report.get('id')}")
                continue
            country = countries.get(infection.get("country_id"))
            if not country:
                print(f"Pays manquant pour l'infection ID {infection.get('id')}")
                continue
            pandemic = pandemics.get(infection.get("pandemic_id"))
            if not pandemic:
                print(f"Pandémie manquante pour l'infection ID {infection.get('id')}")
                continue

            rows.append({
                "report_date": report.get("date"),
                "total_cases": infection.get("total_cases", 0),
                "total_deaths": infection.get("total_deaths", 0),
                "population": country.get("population", 0),
                "pandemic_name": pandemic.get("name", "Unknown")
            })

        if not rows:
            print("Aucune ligne valide n'a été créée.")
            raise ValueError("Aucune ligne de données formatées valides.")

        return pd.DataFrame(rows)