from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils.link_detector import contains_links

app = Flask(__name__)
CORS(app)

# Load models
vectorizer_with_links = joblib.load('model/Link_Analysis/tfidf_vectorizer.pkl')
model_with_links = joblib.load('model/Link_Analysis/phishing_url_model.pkl')

bert_model = BertForSequenceClassification.from_pretrained("model/NLP_Model", local_files_only=True)
bert_tokenizer = BertTokenizer.from_pretrained("model/NLP_Model", local_files_only=True)

label_map = {0: "ham", 1: "phishing"}

# Link extractor
def extract_links(text):
    return re.findall(r'https?://\S+|www\.\S+', text)

@app.route('/api/classify', methods=['POST'])
def classify_email():
    data = request.get_json()
    email_text = data.get('text', '').strip()

    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    # 1. Always run BERT model
    inputs = bert_tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        bert_pred_id = torch.argmax(logits, dim=1).item()
    email_classification = label_map.get(bert_pred_id, bert_pred_id)

    # 2. If links found, classify the first one (or all, if you prefer)
    links = extract_links(email_text)
    link_analysis_result = None

    if links:
        first_link = links[0]
        X = vectorizer_with_links.transform([first_link])
        link_prediction = model_with_links.predict(X)[0]
        link_analysis_result = {
            'link': first_link,
            'prediction': str(link_prediction)
        }

    # 3. Return both results
    return jsonify({
        'email_body_classification': email_classification,
        'link_analysis': link_analysis_result
    })

if __name__ == '__main__':
    app.run(debug=True)
