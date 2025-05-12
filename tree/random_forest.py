from random_forest_service import RandomForestService

if __name__ == "__main__":
    try:
        rfs = RandomForestService()
        accuracy = rfs.train_model_once()
        print(f"Précision du modèle : {accuracy}")
        prediction = rfs.predict({
            "total_cases": 100000,
            "total_deaths": 3000,
            "population": 50000000,
            "year": 2020
        })
        print("Prediction:", prediction)
    except ValueError as e:
        print(f"Erreur: {e}")
