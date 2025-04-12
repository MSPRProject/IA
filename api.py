from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot.model import ChatbotModel
import torch

api = Flask(__name__)
CORS(api)

chatbot = ChatbotModel(
    model_name='google/flan-t5-base',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

@api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({'success': False, 'error': 'No question provided'}), 400

    prompt = data['question']

    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    do_sample = data.get('do_sample', False)
    num_return_sequences = data.get('num_return_sequences', 1)

    response = chatbot.generate_response(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences
    )

    if isinstance(response, dict) and response.get("action") == "chart":
        return jsonify({
            'success': True,
            'response': 'Voici le graphique demandé.',
            'chart': response.get("data", {}),
            'chart_type': response.get("chart_type", "bar"),
        })

    return jsonify({'success': True, 'response': response})


if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=5000)
