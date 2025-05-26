import pandas as pd

class DataSorting:
    @staticmethod
    def format_and_sort_data(countries, pandemics, infections, reports):
        countries_dict = {c['id']: c for c in countries}
        pandemics_dict = {p['id']: p for p in pandemics}
        infections_dict = {i['id']: i for i in infections}        
        rows = []
        
        for report in reports:
            infection_id = report.get("infection_id")
            infection = infections_dict.get(infection_id)
            if not infection:
                print("Exemples de clés dans infections_url_dict :", list(infections_dict.keys())[:5])
                continue
            country = countries_dict.get(infection['country_id'])
            pandemic = pandemics_dict.get(infection['pandemic_id'])
            if not country or not pandemic:
                continue

            rows.append({
                "report_date": report.get("date"),
                "new_cases": report.get("new_cases", 0),
                "new_deaths": report.get("new_deaths", 0),
                "country_iso3": country.get("iso3"),
                "country_name": country.get("name"),
                "continent": country.get("continent"),
                "population": country.get("population"),
                "pandemic_name": pandemic.get("name"),
                "pandemic_pathogen": pandemic.get("pathogen"),
                "total_cases": infection.get("total_cases", 0),
                "total_deaths": infection.get("total_deaths", 0),
            })

        if not rows:
            print("Aucune ligne n'a été ajoutée. Vérifiez les données brutes.")
            raise ValueError("Les données brutes ne contiennent pas de rapports valides.")

        df = pd.DataFrame(rows)
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.sort_values(by=["report_date", "country_iso3", "pandemic_name"]).reset_index(drop=True)
        df = df.drop_duplicates()

        print("Aperçu des données formatées :")
        print(df.head())
        print("Pays uniques :", df["country_iso3"].unique())
        print("Pandémies uniques :", df["pandemic_name"].unique())
        print("Dates de rapport min/max :", df["report_date"].min(), df["report_date"].max())
        return df