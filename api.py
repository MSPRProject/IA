from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot.model import ChatbotModel
import torch

api = Flask(__name__)
CORS(api)  # CORS activé pour accepter les appels cross-origin

chatbot = ChatbotModel(model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu')

@api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    # Vérification que le prompt est bien dans les données
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 1.0)  # Température prise depuis la requête
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    do_sample = data.get('do_sample', True)
    num_return_sequences = data.get('num_return_sequences', 1)

    # Appel à la méthode generate_response avec la température
    response = chatbot.generate_response(
        prompt,
        max_length=max_length,
        temperature=temperature,  # Température passée ici
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences
    )
    
    return jsonify({'response': response})

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=5000)
