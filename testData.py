# from fastapi import FastAPI
# import joblib
# import numpy as np
# import pandas as pd

# app = FastAPI()

# model = joblib.load("random_forest_model.pkl")
# scaler = joblib.load("scaler.pkl")

# print("Classes du modèle :", model.classes_)

# try:
#     print("Features du scaler :", scaler.feature_names_in_)
# except AttributeError:
#     print("Le scaler n'a pas d'attribut 'feature_names_in_' (ancienne version sklearn ?)")

# print("Moyennes du scaler :", scaler.mean_)
# print("Variances du scaler :", scaler.var_)

# X_test = pd.DataFrame([
#     [1000, 50, 2022],      # test mpox
#     [100000, 3000, 2020],  # test covid-19
#     [0, 0, 2021],          # test neutre
# ], columns=["new_cases", "new_deaths", "year"])

# X_scaled = scaler.transform(X_test)
# predictions = model.predict(X_scaled)
# print("Prédictions pour différents cas :", predictions)

# import requests
# r = requests.post("http://127.0.0.1:8081/predict")
# print(r.json())









# import requests
# from tree.get_data import get_all_reports

# # Récupère tous les reports
# reports = get_all_reports()

# reports_hist = [r for r in reports if r.get("date", "").startswith(("2020", "2021", "2022"))]

# print(f"Nombre de reports historiques (2020-2022) : {len(reports_hist)}")

# for report in reports_hist[:5]:
#     new_cases = report.get("new_cases", 0)
#     new_deaths = report.get("new_deaths", 0)
#     year = 2023
#     date = report.get("date")

#     response = requests.post(
#         "http://127.0.0.1:8081/predict",
#         params={
#             "new_cases": new_cases,
#             "new_deaths": new_deaths,
#             "year": year
#         }
#     )
#     if response.status_code == 200:
#         prediction = response.json()["prediction"]
#         print(f"Report {report['id']} (original date {date}): cases={new_cases}, deaths={new_deaths}, year=2023 => Prediction: {prediction}")
#     else:
#         print(f"Erreur pour le report {report['id']} : {response.text}")
        



import joblib

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Features attendues par le scaler :", getattr(scaler, "feature_names_in_", "Non disponible"))
if hasattr(model, "classes_"):
    print("Classes du modèle :", model.classes_)
else:
    print("Le modèle n'est pas un classifieur, pas de classes à afficher.")

print("Moyennes du scaler :", getattr(scaler, "mean_", "Non disponible"))
print("Variances du scaler :", getattr(scaler, "var_", "Non disponible"))
