import requests
from dotenv import dotenv_values
import json

config = dotenv_values(".env")
BEARER_TOKEN = config.get("BEARER_TOKEN")
BASE_URL = "http://localhost:8081/prompt" 

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

def get_all_data():
    try:
        countries = get_countries()
        pandemics = get_pandemics()
        infections = get_infections()

        reports = []
        for infection in infections:
            infection_id = infection["id"]
            reports.extend(get_reports(infection_id))

        return {
            "countries": countries,
            "pandemics": pandemics,
            "infections": infections,
            "reports": reports,
        }
    except requests.HTTPError as e:
        print(f"Erreur HTTP lors de l'appel à FastAPI : {e}")
        return None
    except requests.RequestException as e:
        print(f"Erreur lors de l'appel à FastAPI : {e}")
        return None

def get_countries():
    url = f"{BASE_URL}/countries"
    response = requests.get(url)
    data = response.json()
    countries = data.get("_embedded", {}).get("countries", [])
    return [
        {
            "id": country.get("id"),
            "name": country.get("name"),
            "continent": country.get("continent"),
            "iso3": country.get("iso3"),
            "population": country.get("population"),
        }
        for country in countries
    ]

def get_pandemics():
    url = f"{BASE_URL}/pandemics"
    response = requests.get(url)
    data = response.json()
    pandemics = data.get("_embedded", {}).get("pandemics", [])
    return [
        {
            "id": pandemic.get("id"),
            "name": pandemic.get("name"),
            "pathogen": pandemic.get("pathogen"),
            "start_date": pandemic.get("start_date"),
            "end_date": pandemic.get("end_date"),
        }
        for pandemic in pandemics
    ]

def get_infections():
    url = f"{BASE_URL}/infections"
    response = requests.get(url)
    data = response.json()
    infections = data.get("_embedded", {}).get("infections", [])
    return [
        {
            "id": infection.get("id"),
            "total_cases": infection.get("total_cases"),
            "total_deaths": infection.get("total_deaths"),
            "country_id": extract_id_from_url(infection["_links"]["country"]["href"]),
            "pandemic_id": extract_id_from_url(infection["_links"]["pandemic"]["href"]),
        }
        for infection in infections
    ]

def get_reports():
    url = f"{BASE_URL}/infections/{id}/reports"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    reports = data.get("_embedded", {}).get("reports", [])
    return [
        {
            "id": report.get("id"),
            "date": report.get("date"),
            "new_cases": report.get("new_cases"),
            "new_deaths": report.get("new_deaths"),
            "infection_id": extract_id_from_url(report["_links"]["infection"]["href"]),
        }
        for report in reports
    ]

def extract_id_from_url(url):
    return int(url.rstrip("/").split("/")[-1])
