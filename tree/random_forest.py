from tree.random_forest_service import RandomForestService
from tree.get_data import get_all_countries, get_all_pandemics, get_all_infections, get_all_reports
from tree.data_sorting import DataSorting

if __name__ == "__main__":
    
    countries = get_all_countries()
    pandemics = get_all_pandemics()
    infections = get_all_infections()
    reports = get_all_reports()

    df = DataSorting.format_and_sort_data(countries, pandemics, infections, reports)
    
    model = RandomForestService()
    accuracy = model.train_model_once()
    print(f"Précision du modèle : {accuracy}")

