from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

class ChatbotModel:
    def __init__(self, model_name='gpt2', device=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model.to(self.device)

    def math_verified(self, prompt):
        # Vérifie si la question contient des signes mathématiques
        return bool(re.search(r'[+\-*/]', prompt))

    def generate_response(self, prompt, max_length=500, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, num_return_sequences=1):
        if self.math_verified(prompt):
            try:
                result = eval(prompt)
                return str(result)
            except:
                return "Désolé, je n'ai pas pu résoudre cette opération."
        else:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            attention_mask = torch.ones(inputs.shape, device=self.device)

            pad_token_id = self.tokenizer.eos_token_id

            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                attention_mask=attention_mask
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()

if __name__ == "__main__":
    chatbot = ChatbotModel(model_name='gpt2')
    prompt = ""
    response = chatbot.generate_response(prompt)
    print(response)
