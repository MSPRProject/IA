from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

class ChatbotModel:
    def __init__(self, model_name='google/flan-t5-base', device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model.to(self.device)

    def math_verified(self, prompt):
        return bool(re.search(r'^[\d\s+\-*/().]+$', prompt))

    def generate_response(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, num_return_sequences=1):
        if self.math_verified(prompt):
            try:
                result = eval(prompt)
                return str(result)
            except:
                return "Désolé, je n'ai pas pu résoudre cette opération."
        
        task_prompt = f"Question : {prompt}".strip()
        inputs = self.tokenizer(task_prompt, return_tensors="pt").to(self.device)

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
    prompt = ""
    print(chatbot.generate_response(prompt))
