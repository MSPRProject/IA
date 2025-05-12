import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from get_data import get_all_data  
import json

class RandomForestService:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.is_trained = False

    def train_model_once(self):
        if self.is_trained:
            return self.accuracy

        data = get_all_data()
        if data is None:
            raise ValueError("Impossible de récupérer les données via l'API FastAPI.")

        if not isinstance(data, dict):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Les données reçues ne sont pas au format JSON valide.")

        required_keys = ["reports", "infections", "pandemics", "countries"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Les données sont manquantes pour les clés : {', '.join(missing_keys)}")
        
        df = self._format_data(data)
        df.info()

        df["report_date"] = pd.to_datetime(df["report_date"])
        df["year"] = df["report_date"].dt.year

        df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)

        required_columns = ["pandemic_name", "total_cases", "total_deaths", "population", "year"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Les données formatées sont incomplètes. Colonnes manquantes : {', '.join(missing_columns)}")

        df.info()

        # print("Colonnes présentes :", df.columns.tolist())
        print("Valeurs manquantes par colonne :\n", df.isnull().sum())

        X = df[["total_cases", "total_deaths", "population", "year"]]
        y = df["pandemic_name"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # print("Données d'entraînement (X_train) :\n", X_train.head())
        # print("Labels d'entraînement (y_train) :\n", y_train.head())
        # print("Classes uniques dans y :", y.unique())
        # print("Statistiques descriptives de X :\n", X.describe())
        # print("Valeurs uniques dans y :", y.value_counts())
        # print("Données formatées (df) :\n", df.head())
        # print("Données brutes retournées par l'API :", data)
        # exit(0)

        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.model = model
        self.is_trained = True

        return self.accuracy


    def predict(self, data: dict):
        if not self.is_trained:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        required_keys = ["total_cases", "total_deaths", "population", "year"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Les données sont manquantes pour les clés : {', '.join(missing_keys)}")

        df = pd.DataFrame([data])
        return self.model.predict(df)[0]

    def _format_data(self, data: dict) -> pd.DataFrame:
        embedded = data.get("reports", {}).get("_embedded", {})
        reports = embedded.get("reports", [])

        if not isinstance(reports, list):
            raise ValueError(f"La clé 'reports' ne contient pas une liste valide : {reports}")

        if not all(isinstance(report, dict) for report in reports):
            raise ValueError("Les éléments de 'reports' ne sont pas tous des dictionnaires.")

        infections = {i.get("id"): i for i in data.get("infections", {}).get("_embedded", {}).get("infections", [])}
        pandemics = {p.get("id"): p for p in data.get("pandemics", {}).get("_embedded", {}).get("pandemics", [])}
        countries = {c.get("id"): c for c in data.get("countries", {}).get("_embedded", {}).get("countries", [])}

        rows = []
        for report in reports:
            infection = infections.get(report.get("id"))
            if not infection:
                continue
            country = countries.get(infection.get("id"))
            pandemic = pandemics.get(infection.get("id"))
            if not country or not pandemic:
                continue

            rows.append({
                "report_date": report.get("date"),
                "total_cases": infection.get("total_cases", 0),
                "total_deaths": infection.get("total_deaths", 0),
                "population": country.get("population", 0),
                "pandemic_name": pandemic.get("name", "Unknown")
            })

        if not rows:
            raise ValueError("Les données formatées sont incomplètes.")

        return pd.DataFrame(rows)
