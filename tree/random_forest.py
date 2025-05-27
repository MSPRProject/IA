from tree.random_forest_service import RandomForestService
from tree.get_data import get_all_countries, get_all_pandemics, get_all_infections, get_all_reports
from tree.data_sorting import DataSorting

if __name__ == "__main__":
    model = RandomForestService()
    accuracy = model.train_model_once()
    print(f"Précision du modèle : {accuracy}")

