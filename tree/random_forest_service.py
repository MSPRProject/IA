import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import requests
import json
from tqdm import tqdm
from tree.get_data import get_all_countries, get_all_pandemics, get_all_infections, get_all_reports
from services.api_service import ApiService

INPUT_COLUMNS = (
    ["pandemic_name", "pandemic_pathogen", "country_iso3", "continent"]
    + list(f"report{i}_date" for i in range(100))
    + list(f"report{i}_new_cases" for i in range(100))
    + list(f"report{i}_new_deaths" for i in range(100))
)

OUTPUT_COLUMNS = ["target_date", "target_new_cases", "target_new_deaths"]

class RandomForestService:
    def __init__(self, model = None, scaler = None):
        self.model = model
        self.accuracy = None
        self.scaler = scaler
        self.is_trained = model != None and scaler != None


    @staticmethod
    def format_predict_data(data: dict):
        # data schema:
        # {
        #     reports: [{
        #         date: date,
        #         new_cases: int,
        #         new_deaths: int,
        #     }; 100],
        #     pandemic_name: str,
        #     pandemic_pathogen: str,
        #     pandemic_duration: int (in days),
        #     country_iso3: str,
        #     continent
        # }
        result = pd.DataFrame()

        row = pd.Series()
        row["pandemic_name"] = data["pandemic_name"]
        row["pandemic_pathogen"] = data["pandemic_pathogen"]
        row["country_iso3"] = data["country_iso3"]
        row["continent"] = data["continent"]

        for i in range(100):
            if len(data["reports"]) >= i:
                report = data["reports"][i]
            else:
                report = {"date": None, "new_cases": None, "new_deaths": None}

            row[f"report{i}_date"] = report["date"]
            row[f"report{i}_new_cases"] = report["new_cases"]
            row[f"report{i}_new_deaths"] = report["new_deaths"]

        return pd.concat([result, row])

    @staticmethod
    def format_training_data(data: [dict]):
        # data schema:
        # {
        #     reports: [{
        #         date: date,
        #         new_cases: int,
        #         new_deaths: int,
        #     }; 100],
        #     pandemic_name: str,
        #     pandemic_pathogen: str,
        #     pandemic_duration: int (in days),
        #     target: {
        #         date: date,
        #         new_cases: int,
        #         new_deaths: int,
        #     },
        #     location: str,
        # }
        result = pd.DataFrame()

        def format_one_data(d: dict):
            row = {}
            row["pandemic_name"] = d["pandemic_name"]
            row["pandemic_pathogen"] = d["pandemic_pathogen"]
            row["country_iso3"] = d["country_iso3"]
            row["continent"] = d["continent"]

            for i in range(100):
                if len(d["reports"]) > i:
                    report = d["reports"][i]
                else:
                    report = {"date": None, "new_cases": None, "new_deaths": None}

                row[f"report{i}_date"] = report["date"]
                row[f"report{i}_new_cases"] = report["new_cases"]
                row[f"report{i}_new_deaths"] = report["new_deaths"]

            row["target_date"] = d["target"]["date"]
            row["target_new_cases"] = d["target"]["new_cases"]
            row["target_new_deaths"] = d["target"]["new_deaths"]

            return row

        return pd.DataFrame(joblib.Parallel(n_jobs=16)(joblib.delayed(format_one_data)(d) for d in tqdm(data)))

    @staticmethod
    def load(base: str):
        try:
            model = joblib.load(base + "random_forest_model.pkl")
            scaler = joblib.load(base + "scaler.pkl")
            return RandomForestService(model, scaler)
        except FileNotFoundError:
            raise ValueError("Le modèle ou le scaler n'a pas été trouvé. Veuillez entraîner le modèle d'abord.")

    def train_model_once(self, api_service: ApiService):
        if self.is_trained:
            return self.accuracy

        training_data = api_service.get_training_data()

        print("Formatage des données...")
        df = RandomForestService.format_training_data(training_data)

        if df.empty:
            raise ValueError("Les données formatées sont vides. Impossible d'entraîner le modèle.")

        print("Données formatées:")
        print(df.head())

        print(f"Nombre de lignes avant suppression des doublons : {len(df)}")
        df = df.drop_duplicates()
        print(f"Nombre de lignes après suppression des doublons : {len(df)}")

        X = df[INPUT_COLUMNS]
        y = df[OUTPUT_COLUMNS]

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

    def predict(self, predict_data: dict):
        formatted = RandomForestService.format_predict_data(predict_data)
        print("Prediction data: {formatted}")

        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return prediction[0]

