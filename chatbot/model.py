from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pandas as pd
import re
import os

class ChatbotModel:
    def __init__(self, model_name='google/flan-t5-base', device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model.to(self.device)

        self.df = self.load_csv("chatbot/data_mspr.csv") # Partie à modifier quand l'api sera fini, pour l'instant en brut dans le code

    def load_csv(self, path):
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, low_memory=False)
            return df
        except Exception as e:
            print(f"[ERREUR CSV] {e}")
            return None

    def extract_country_data(self, prompt):
        if self.df is None:
            return ""

        prompt_lower = prompt.lower()
        for country in self.df["Country/Region"].dropna().unique():
            if country.lower() in prompt_lower:
                row = self.df[self.df["Country/Region"].str.lower() == country.lower()].iloc[0]
                deaths = row["Deaths"]
                context = f"Le nombre de décès en {country} est {deaths}."
                return context

        # prompt_lower = prompt.lower()
        # for country in self.df["Country/Region"].dropna().unique():
        #     if country.lower() in prompt_lower:
        #         row = self.df[self.df["Country/Region"].str.lower() == country.lower()].iloc[0]
        #         selected_cols = [
        #             "Country/Region", "Confirmed", "Deaths", "Recovered",
        #             "Active", "New cases", "New deaths", "Deaths / 100 Cases",
        #             "Recovered / 100 Cases", "Date", "Population"
        #         ]
        #         context = "\n".join([f"{col}: {row[col]}" for col in selected_cols if col in row and pd.notna(row[col])])
        #         return context
        return ""

    def math_verified(self, prompt):
        return bool(re.fullmatch(r'^[\d\s+\-*/().]+$', prompt))

    def generate_response(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, num_return_sequences=1):
        if self.math_verified(prompt):
            try:
                result = eval(prompt)
                return str(result)
            except:
                return "Désolé, je n'ai pas pu résoudre cette opération."

        context_info = self.extract_country_data(prompt)
        if context_info:
            enriched_prompt = f"Context:\n{context_info}\n\nQuestion: {prompt}"
        else:
            enriched_prompt = f"Question: {prompt}"

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

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

if __name__ == "__main__":
    chatbot = ChatbotModel()
    prompt = input("Pose une question : ")
    print(chatbot.generate_response(prompt))
