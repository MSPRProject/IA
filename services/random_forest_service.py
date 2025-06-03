from services.api_service import ApiService
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import json
import random

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
            row[f"report{i}_new_cases"] = report["new_cases"] if report["new_cases"] is not None and report["new_cases"] > 0 else 0
            row[f"report{i}_new_deaths"] = report["new_deaths"] if report["new_deaths"] is not None and report["new_deaths"] > 0 else 0

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
                row[f"report{i}_new_cases"] = report["new_cases"] if report["new_cases"] is not None and report["new_cases"] > 0 else 0
                row[f"report{i}_new_deaths"] = report["new_deaths"] if report["new_deaths"] is not None and report["new_deaths"] > 0 else 0

            row["target_date"] = d["target"]["date"]
            row["target_new_cases"] = d["target"]["new_cases"] if d["target"]["new_cases"] is not None and d["target"]["new_cases"] > 0 else 0
            row["target_new_deaths"] = d["target"]["new_deaths"] if d["target"]["new_deaths"] is not None and d["target"]["new_deaths"] > 0 else 0

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
            raise ValueError("[RandomForestService] Model or scaler file not found. Please train the model first.")

    def _prepare_training_data(self, df: pd.DataFrame):
        df = df.copy()

        df.dropna(subset=OUTPUT_COLUMNS, inplace=True)
        print(f"[RandomForestService] Row count after null/NaN cleaning: {len(df)}")
        if df.empty:
            raise ValueError("[RandomForestService] Unable to train model: formatted data is empty.")

        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[RandomForestService] Row count after deleting duplicated: {len(df)}")

        categorical_features = ["pandemic_name", "pandemic_pathogen", "country_iso3", "continent"]
        date_features = [f"report{i}_date" for i in range(100)] + ["target_date"]
        numerical_features = [f"report{i}_new_cases" for i in range(100)] + \
                             [f"report{i}_new_deaths" for i in range(100)]

        for col in tqdm(date_features, desc="Converting report dates"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = (df[col].astype(np.int64) // 10**9).astype(float)

        num_rows_before_report_drop = len(df)

        report_new_cases_cols = [f"report{i}_new_cases" for i in range(100)]
        report_new_deaths_cols = [f"report{i}_new_deaths" for i in range(100)]
        report_value_cols = report_new_cases_cols + report_new_deaths_cols

        # Drop rows where all report values are either 0, negative, or NaN
        temp_df = df[report_value_cols].replace(0, np.nan)

        mask = temp_df.isnull().all(axis=1)
        df = df[~mask].reset_index(drop=True)

        num_rows_dropped = num_rows_before_report_drop - len(df)

        if num_rows_dropped > 0:
            print(f"[RandomForestService] Deleted {num_rows_dropped} rows with no/empty reports.")
            if df.empty:
                raise ValueError("[RandomForestService] Unable to train model: formatted data is empty.")

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
            print("[RandomForestService] Warning: Dataset has <= 1 sample. Evaluating on training data.")
        elif len(df) < 10:
            X_train, y_train = X, y
            X_test, y_test = X, y
            print(f"[RandomForestService] Warning: Small dataset ({len(df)} samples). Evaluating on training data.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"[RandomForestService] Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        X_train_processed = self.scaler.fit_transform(X_train)
        X_test_processed = self.scaler.transform(X_test)

        if X_train_processed.shape[0] == 0:
            raise ValueError("[RandomForestService] Training data empty after preprocessing.")

        return X_train, X_test, X_train_processed, y_train, X_test_processed, y_test

    def train_model(self, api_service: ApiService, model_base_path: str):
        if self.is_trained:
            return self.accuracy

        # Data fetching and processing
        training_data = api_service.get_training_data()
        df = RandomForestService.format_training_data(training_data)

        X_train, X_test, X_train_processed, y_train, X_test_processed, y_test = self._prepare_training_data(df)

        if df.empty:
            raise ValueError("[RandomForestService] Unable to train model: formatted data is empty.")

        print(f"[RandomForestService] Initial formatted data: {len(df)} rows.")

        # Train the Random Forest model
        random_state = random.randint(0, 1000)

        print(f"[RandomForestService] Random state for model training: {random_state}")
        parameters_range = {
            "n_estimators": randint(50, 100),
            "max_depth": randint(5, 15),
        }

        # HPO
        hpo_sample_fraction = 0.5
        min_samples_for_hpo = 100
        if (X_train_processed.shape[0] * hpo_sample_fraction) < min_samples_for_hpo and X_train_processed.shape[0] > min_samples_for_hpo:
            print(f"[RandomForestService] HPO sample fraction too small, using min_samples_for_hpo: {min_samples_for_hpo}")
            hpo_sample_fraction = min_samples_for_hpo / X_train_processed.shape[0]


        if X_train_processed.shape[0] <= min_samples_for_hpo:
            print("[RandomForestService] Dataset too small for HPO subsampling, using full training data for HPO.")
            X_hpo, y_hpo = X_train_processed, y_train
        else:
            X_hpo, _, y_hpo, _ = train_test_split(
                X_train_processed,
                y_train,
                train_size=hpo_sample_fraction,
                random_state=random_state,
            )
        print(f"[RandomForestService] Using {X_hpo.shape[0]} samples for HPO.")

        regressor_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        optimization_model = RandomizedSearchCV(
            regressor_model,
            parameters_range,
            n_iter=10,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=3,
            n_jobs=-1,
            random_state=random_state
        )

        print("[RandomForestService] Starting HPO on subset of training data...")
        optimization_model.fit(X_hpo, y_hpo)

        print("[RandomForestService] HPO complete. Training best model on full training data...")
        self.model = optimization_model.best_estimator_
        self.model.fit(X_train_processed, y_train)

        # Model evaluation
        score = -np.inf
        if X_test_processed.shape[0] > 0:
            print("[RandomForestService] Predicting test data...")
            y_pred = self.model.predict(X_test_processed)

            score = r2_score(y_test, y_pred)
            print(f"[RandomForestService] Model trained with a R2 of {score:.4f}")

            y_test_cases = y_test['target_new_cases']
            y_pred_cases = y_pred[:, 0]
            mse_cases = mean_squared_error(y_test_cases, y_pred_cases)
            rmse_cases = np.sqrt(mse_cases)
            print(f"[RandomForestService] | New Cases - MSE: {mse_cases:.4f}, RMSE: {rmse_cases:.4f} cases")

            y_test_deaths = y_test['target_new_deaths']
            y_pred_deaths = y_pred[:, 1]
            mse_deaths = mean_squared_error(y_test_deaths, y_pred_deaths)
            rmse_deaths = np.sqrt(mse_deaths)
            print(f"[RandomForestService] | New Deaths - MSE: {mse_deaths:.4f}, RMSE: {rmse_deaths:.4f} deaths")

            print(f"[RandomForestService] | Mean new cases: {y_test_cases.mean():.2f}")
            print(f"[RandomForestService] | Mean new deaths: {y_test_deaths.mean():.2f}")

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

        return self.accuracy

    def predict(self, predict_data: dict):
        formatted = RandomForestService.format_predict_data(predict_data)
        print(f"Prediction data: {formatted}")

        X_scaled = self.scaler.transform(formatted)
        prediction_transformed = self.model.predict(X_scaled)
        print(f"Log-transformed prediction: {prediction_transformed}")

        predicted_cases_original = np.maximum(0, np.expm1(prediction_transformed[0][0]))
        predicted_deaths_original = np.maximum(0, np.expm1(prediction_transformed[0][1]))

        return json.dumps({
            "new_cases": predicted_cases_original,
            "new_deaths": predicted_deaths_original
        })
