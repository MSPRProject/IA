import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from database_service import DatabaseService

class RandomForestService:
    def __init__(self):
        self.model = None
        self.accuracy = None

    def train_model(self):
        db_service = DatabaseService(password="mspr_password") 
        df = db_service.get_full_dataset()
        db_service.close()

        df["report_date"] = pd.to_datetime(df["report_date"])
        df["year"] = df["report_date"].dt.year

        X = df[["total_cases", "total_deaths", "population", "year"]]
        y = df["pandemic_name"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.model = model
        self.accuracy = accuracy

        return accuracy

    def predict(self, data: dict):
        if not self.model:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        df = pd.DataFrame([data])
        return self.model.predict(df)[0]
