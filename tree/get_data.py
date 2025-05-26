import requests
from tqdm import tqdm

def get_all_countries():
    url = "http://localhost:8080/countries"
    params = {"page": 0, "size": 20}
    countries = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        embedded = data.get("_embedded", {})
        page_countries = embedded.get("countries", [])
        countries.extend(page_countries)
        page_info = data.get("page", {})
        if page_info.get("number", 0) >= page_info.get("totalPages", 1) - 1:
            break
        params["page"] += 1
    return countries

def get_all_pandemics():
    url = "http://localhost:8080/pandemics"
    params = {"page": 0, "size": 20}
    pandemics = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        embedded = data.get("_embedded", {})
        page_pandemics = embedded.get("pandemics", [])
        pandemics.extend(page_pandemics)
        page_info = data.get("page", {})
        if page_info.get("number", 0) >= page_info.get("totalPages", 1) - 1:
            break
        params["page"] += 1
    return pandemics

def get_all_infections():
    url = "http://localhost:8080/infections"
    params = {"page": 0, "size": 20}
    infections = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        embedded = data.get("_embedded", {})
        page_infections = embedded.get("infections", [])
        for infection in page_infections:
            country_url = infection["_links"]["country"]["href"]
            country_response = requests.get(country_url)
            country_data = country_response.json()
            infection["country_id"] = country_data["id"]
            pandemic_url = infection["_links"]["pandemic"]["href"]
            pandemic_response = requests.get(pandemic_url)
            pandemic_data = pandemic_response.json()
            infection["pandemic_id"] = pandemic_data["id"]
        infections.extend(page_infections)
        page_info = data.get("page", {})
        if page_info.get("number", 0) >= page_info.get("totalPages", 1) - 1:
            break
        params["page"] += 1
    return infections

def get_all_reports():
    url = "http://localhost:8080/reports"
    params = {"page": 0, "size": 20}
    reports = []
    response = requests.get(url, params=params)
    data = response.json()
    embedded = data.get("_embedded", {})
    page_reports = embedded.get("reports", [])
    page_info = data.get("page", {})
    total_pages = page_info.get("totalPages", 1)

    for page in tqdm(range(0, total_pages), desc="Récupération des reports"):
        params["page"] = page
        response = requests.get(url, params=params)
        data = response.json()
        embedded = data.get("_embedded", {})
        page_reports = embedded.get("reports", [])
        for report in page_reports:
            infection_url = report["_links"]["infection"]["href"]
            infection_response = requests.get(infection_url)
            infection_data = infection_response.json()
            report["infection_id"] = infection_data["id"]
        reports.extend(page_reports)

    print(f"Nombre total de reports récupérés : {len(reports)}")
    return reports