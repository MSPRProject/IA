import pandas as pd
import torch
import os
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from chatbot.bdd import lire_pays

class ChatbotModel:
    def __init__(self, model_name='google/flan-t5-base', device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model.to(self.device)

        self.df = self.load_csv("chatbot/data_mspr.csv")
        self.previous_countries = []
        self.countries_data = self.load_countries_from_db()

    def load_countries_from_db(self):
        countries = lire_pays() 
        country_list = [country[1] for country in countries] 
        return country_list

    def extract_country_data(self, prompt):
        if not self.countries_data:
            return {}

        prompt_lower = prompt.lower()
        for country in self.countries_data:
            if country.lower() in prompt_lower:
                return {
                    "country": country
                }
        return {}

    def load_csv(self, path):
        if not os.path.exists(path):
            print(f"[ERREUR] Le fichier {path} est introuvable.")
            return None
        try:
            df = pd.read_csv(path, low_memory=False)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            return df
        except Exception as e:
            print(f"[ERREUR CSV] {e}")
            return None

    def extract_all_data(self, country):
        if self.df is None:
            return None
        if "Country/Region" not in self.df.columns:
            return None
        country_data = self.df[self.df["Country/Region"].str.lower() == country.lower()]
        if country_data.empty:
            return None
        return country_data.iloc[0].to_dict()

    def get_chart_parameters(self, prompt):
        if self.df is None:
            return {}

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["évolution", "temps", "courbe", "evolution", "time", "curve", "linear"]):
            chart_type = "line"
        elif any(word in prompt_lower for word in ["camembert", "pie", "camembert", "pie"]):
            chart_type = "pie"
        elif any(word in prompt_lower for word in ["radar"]):
            chart_type = "radar"
        elif any(word in prompt_lower for word in ["barre", "comparaison", "diagramme", "bar", "comparison", "diagram"]):
            chart_type = "bar"
        elif any(word in prompt_lower for word in ["horizontale", "horizontal"]):
            chart_type = "horizontal_bar"
        else:
            chart_type = "bar" 

        countries = []
        if "Country/Region" in self.df.columns:
            countries = [c for c in self.df["Country/Region"].dropna().unique() if c.lower() in prompt_lower]

        metrics = [col for col in self.df.columns if col.lower() in prompt_lower and col.lower() not in ["country/region", "date"]]
        if not metrics:
            metrics = ["Confirmed"] if "Confirmed" in self.df.columns else self.df.columns[2:3].tolist()

        return {
            "chart_type": chart_type,
            "countries": countries,
            "metrics": metrics,
            "time_series": chart_type == "line"
        }

    def generate_chart_data(self, chart_params):
        if self.df is None or "Country/Region" not in self.df.columns:
            return {}

        countries = chart_params.get("countries", [])
        metrics = chart_params.get("metrics", [])
        chart_type = chart_params.get("chart_type", "bar")

        chart_data = {"labels": [], "datasets": []}
        if not countries or not metrics:
            return {}

        if chart_type == "line" and "Date" in self.df.columns:
            dates = sorted(self.df["Date"].dropna().unique())
            chart_data["labels"] = [str(d.date()) for d in dates]
            for country in countries:
                country_df = self.df[self.df["Country/Region"].str.lower() == country.lower()]
                if country_df.empty:
                    continue
                country_df = country_df.sort_values("Date")
                for metric in metrics:
                    if metric not in country_df.columns:
                        continue
                    values = country_df[metric].fillna(0).tolist()
                    chart_data["datasets"].append({
                        "label": f"{metric} - {country}",
                        "data": values,
                        "fill": False,
                        "borderColor": "rgba(75,192,192,1)",
                        "tension": 0.1
                    })

        elif chart_type == "pie":
            for metric in metrics:
                values = []
                for country in countries:
                    row = self.df[self.df["Country/Region"].str.lower() == country.lower()]
                    if row.empty or metric not in row.columns:
                        values.append(0)
                    else:
                        try:
                            val = float(row.iloc[0][metric])
                            values.append(val)
                        except:
                            values.append(0)
                chart_data["datasets"].append({
                    "label": metric,
                    "data": values,
                    "backgroundColor": ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF5733'],
                    "hoverOffset": 4
                })
            chart_data["labels"] = countries

        elif chart_type == "radar":
            for metric in metrics:
                values = []
                for country in countries:
                    row = self.df[self.df["Country/Region"].str.lower() == country.lower()]
                    if row.empty or metric not in row.columns:
                        values.append(0)
                    else:
                        try:
                            val = float(row.iloc[0][metric])
                            values.append(val)
                        except:
                            values.append(0)
                chart_data["datasets"].append({
                    "label": metric,
                    "data": values,
                    "backgroundColor": 'rgba(255, 99, 132, 0.2)',
                    "borderColor": 'rgba(255, 99, 132, 1)',
                    "pointBackgroundColor": 'rgba(255, 99, 132, 1)',
                    "pointBorderColor": '#fff'
                })
            chart_data["labels"] = countries

        elif chart_type == "horizontal_bar":
            chart_data["labels"] = countries
            for metric in metrics:
                values = []
                for country in countries:
                    row = self.df[self.df["Country/Region"].str.lower() == country.lower()]
                    if row.empty or metric not in row.columns:
                        values.append(0)
                    else:
                        try:
                            val = float(row.iloc[0][metric])
                            values.append(val)
                        except:
                            values.append(0)
                chart_data["datasets"].append({
                    "label": metric,
                    "data": values
                })

        else:
            chart_data["labels"] = countries
            for metric in metrics:
                values = []
                for country in countries:
                    row = self.df[self.df["Country/Region"].str.lower() == country.lower()]
                    if row.empty or metric not in row.columns:
                        values.append(0)
                    else:
                        try:
                            val = float(row.iloc[0][metric])
                            values.append(val)
                        except:
                            values.append(0)
                chart_data["datasets"].append({
                    "label": metric,
                    "data": values
                })

        return {
            "action": "chart",
            "chart_type": chart_type,
            "data": chart_data
        }

    def extract_country_data(self, prompt):
        if self.df is None or "Country/Region" not in self.df.columns:
            return {}

        prompt_lower = prompt.lower()
        for country in self.df["Country/Region"].dropna().unique():
            if country.lower() in prompt_lower:
                row = self.df[self.df["Country/Region"].str.lower() == country.lower()].iloc[0]
                return {
                    "country": country,
                    "deaths": row.get("Deaths", 0),
                    "recovered": row.get("Recovered", 0),
                    "confirmed": row.get("Confirmed", 0),
                    "active": row.get("Active", 0),
                    "new_cases": row.get("New cases", 0),
                    "new_deaths": row.get("New deaths", 0),
                    "population": row.get("Population", 0)
                }
        return {}

    def math_verified(self, prompt):
        return bool(re.fullmatch(r'^[\d\s+\-*/().]+$', prompt))

    def should_generate_chart(self, prompt):
        keywords = ["graphique", "courbe", "diagramme", "afficher", "montrer", "évolution", "comparaison", 
                    "chart", "visualiser", "graph", "curve", "diagram", "display", "show", "evolution", "comparison", "visualize", "pie", "radar"]
        return any(word in prompt.lower() for word in keywords)

    def generate_response(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, num_return_sequences=1):
        if self.math_verified(prompt):
            try:
                return str(eval(prompt))
            except:
                return "Désolé, je n'ai pas pu résoudre cette opération."

        if self.should_generate_chart(prompt):
            chart_info = self.get_chart_parameters(prompt)
            return self.generate_chart_data(chart_info)

        context_info = self.extract_country_data(prompt)
        enriched_prompt = f"Context:\n{context_info}\n\nQuestion: {prompt}" if context_info else f"Question: {prompt}"

        inputs = self.tokenizer(enriched_prompt, return_tensors="pt", truncation=True).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    chatbot = ChatbotModel()
    prompt = input("Pose une question : ")
    print(chatbot.generate_response(prompt))
