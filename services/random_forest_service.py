from services.api_service import ApiService
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import json

INPUT_COLUMNS = (
    ["pandemic_name", "pandemic_pathogen", "country_iso3", "continent", "target_date"]
    + list(f"report{i}_date" for i in range(100))
    + list(f"report{i}_new_cases" for i in range(100))
    + list(f"report{i}_new_deaths" for i in range(100))
)

OUTPUT_COLUMNS = ["target_new_cases", "target_new_deaths"]

class RandomForestService:
    def __init__(self, model = None, scaler = None):
        self.model = model
        self.accuracy = None
        self.scaler = scaler
        self.is_trained = model is not None and scaler is not None


    @staticmethod
    def format_predict_data(data: dict):
        row = {}
        row["pandemic_name"] = data["pandemic_name"]
        row["pandemic_pathogen"] = data["pandemic_pathogen"]
        row["country_iso3"] = data["country_iso3"]
        row["continent"] = data["continent"]

        for i in range(100):
            if len(data["reports"]) > i:
                report = data["reports"][i]
            else:
                report = {"date": None, "new_cases": None, "new_deaths": None}

            row[f"report{i}_date"] = report["date"]
            row[f"report{i}_new_cases"] = report["new_cases"]
            row[f"report{i}_new_deaths"] = report["new_deaths"]

        row["target_date"] = data["target"]["date"]

        return pd.DataFrame([row])

    @staticmethod
    def format_training_data(data: [dict]):
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

        formatted_rows = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(format_one_data)(d) for d in tqdm(data, desc="Formatting training data")
        )
        return pd.DataFrame(formatted_rows)

    @staticmethod
    def load(base: str):
        try:
            model_file = "random_forest_model.pkl"
            scaler_file = "scaler.pkl"
            if base:
                model_file = f"{base.rstrip('/')}/{model_file}"
                scaler_file = f"{base.rstrip('/')}/{scaler_file}"

            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            return RandomForestService(model, scaler)
        except FileNotFoundError:
            raise ValueError("Le modèle ou le scaler n'a pas été trouvé. Veuillez entraîner le modèle d'abord.")

    def train_model(self, api_service: ApiService, model_base_path: str):
        if self.is_trained:
            return self.accuracy

        training_data = api_service.get_training_data()
        df = RandomForestService.format_training_data(training_data)

        if df.empty:
            raise ValueError("Les données formatées sont vides. Impossible d'entraîner le modèle.")

        print(f"Données formatées initializs {len(df)} lignes.")

        df.dropna(subset=OUTPUT_COLUMNS, inplace=True)
        print(f"Nombre de lignes aprés nettoyage des cibles NaN : {len(df)}")
        if df.empty:
            raise ValueError("Les données formatées sont vides. Impossible d'entraîner le modèle.")

        df = df.drop_duplicates()
        print(f"Nombre de lignes après suppression des doublons : {len(df)}")

        categorical_features = ["pandemic_name", "pandemic_pathogen", "country_iso3", "continent"]
        date_features = [f"report{i}_date" for i in range(100)] + ["target_date"]
        numerical_features = [f"report{i}_new_cases" for i in range(100)] + \
                             [f"report{i}_new_deaths" for i in range(100)]

        for col in tqdm(date_features, desc="Converting report dates"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = (df[col].astype(np.int64) // 10**9).astype(float)

        report_new_cases_cols = [f"report{i}_new_cases" for i in range(100)]
        report_new_deaths_cols = [f"report{i}_new_deaths" for i in range(100)]
        all_report_value_cols = report_new_cases_cols + report_new_deaths_cols
        no_reports_mask = df[all_report_value_cols].isnull().all(axis=1)

        num_rows_before_report_drop = len(df)
        df = df[~no_reports_mask]
        num_rows_dropped = num_rows_before_report_drop - len(df)

        if num_rows_dropped > 0:
            print(f"{num_rows_dropped} lignes supprimées car elles ne contenaient aucune donnée de rapport (cas/décès).")
            if df.empty:
                raise ValueError("Les données formatées sont vides. Impossible d'entraîner le modèle.")

        X = df[list(INPUT_COLUMNS)]
        y = df[OUTPUT_COLUMNS]

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        numerical_cols_for_transformer = date_features + numerical_features
        self.scaler = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_cols_for_transformer),
                ("cat", categorical_pipeline, categorical_features)
            ],
            remainder="drop"
        )

        if len(df) <= 1:
            X_train, y_train = X, y
            X_test, y_test = X, y
            print("Warning: Dataset <=1 sample. Evaluating on training data.")
        elif len(df) < 10:
            X_train, y_train = X, y
            X_test, y_test = X, y
            print(f"Warning: Dataset small ({len(df)} samples). Evaluating on training data.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        X_train_processed = self.scaler.fit_transform(X_train)
        X_test_processed = self.scaler.transform(X_test)

        if X_train_processed.shape[0] == 0:
            raise ValueError("Training data empty after preprocessing.")

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_processed, y_train)

        score = -np.inf
        mse = np.inf
        if X_test_processed.shape[0] > 0:
            print("Prédiction sur le modèle de test...")
            y_pred = self.model.predict(X_test_processed)
            score = r2_score(y_test, y_pred)
            print(f"Modèle RandomForestRegressor antraîné avec un score R2 de {score:.4f}")

            y_test_cases = y_test['target_new_cases']
            y_pred_cases = y_pred[:, 0]
            mse_cases = mean_squared_error(y_test_cases, y_pred_cases)
            rmse_cases = np.sqrt(mse_cases)
            print(f"New Cases - MSE: {mse_cases:.4f}, RMSE: {rmse_cases:.4f} cases")

            y_test_deaths = y_test['target_new_deaths']
            y_pred_deaths = y_pred[:, 1]
            mse_deaths = mean_squared_error(y_test_deaths, y_pred_deaths)
            rmse_deaths = np.sqrt(mse_deaths)
            print(f"New Deaths - MSE: {mse_deaths:.4f}, RMSE: {rmse_deaths:.4f} deaths")
            print(f"MSE: {mse:.4f}")
        else:
            print("Test set was empty, no evaluation performed on test data.")
            if X_train.equals(X_test) and X_train_processed.shape[0] > 0:
                print("Evaluation score reflects performance on the training data.")

        self.accuracy = score
        self.is_trained = True

        model_file = "random_forest_model.pkl"
        scaler_file = "scaler.pkl"
        if model_base_path:
            model_file = f"{model_base_path.rstrip('/')}/{model_file}"
            scaler_file = f"{model_base_path.rstrip('/')}/{scaler_file}"

        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)

        print(f"Modèle entraîné avec une précision de {self.accuracy:.2f}")
        return self.accuracy

    def predict(self, predict_data: dict):
        formatted = RandomForestService.format_predict_data(predict_data)
        print(f"Prediction data: {formatted}")

        X_scaled = self.scaler.transform(formatted)
        prediction = self.model.predict(X_scaled)
        print(prediction)
        return json.dumps({
            "new_cases": prediction[0][0],
            "new_deaths": prediction[0][1]
        })

