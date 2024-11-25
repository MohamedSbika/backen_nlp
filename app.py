from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import pdfplumber
import os

app = Flask(__name__)
CORS(app)  

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator_en_to_fr = pipeline("translation_en_to_fr")
translator_fr_to_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
translator_ar_to_en = pipeline("translation_ar_to_en", model="Helsinki-NLP/opus-mt-ar-en")
translator_en_to_ar = pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")
translator_ar_to_fr = pipeline("translation_ar_to_fr", model="Helsinki-NLP/opus-mt-ar-fr")
translator_fr_to_ar = pipeline("translation_fr_to_ar", model="Helsinki-NLP/opus-mt-fr-ar")


@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "File is not a PDF"}), 400
    
    pdf_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  
                pdf_text += page_text

    if not pdf_text:
        return jsonify({"error": "No text found in the PDF"}), 400

    input_length = len(pdf_text.split())
    max_length = min(130, int(input_length * 0.6))
    min_length = max(30, int(input_length * 0.3))  
    summary = summarizer(pdf_text, max_length=max_length, min_length=min_length, do_sample=False)

    return jsonify({"summary": summary[0]['summary_text']})

@app.route('/translate_en_to_fr', methods=['POST'])
def translate_en_to_fr():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_en_to_fr(text)
    return jsonify({"translated_text": translation[0]['translation_text']})

@app.route('/translate_fr_to_en', methods=['POST'])
def translate_fr_to_en():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_fr_to_en(text)
    return jsonify({"translated_text": translation[0]['translation_text']})

@app.route('/translate_ar_to_en', methods=['POST'])
def translate_ar_to_en():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_ar_to_en(text)
    return jsonify({"translated_text": translation[0]['translation_text']})

@app.route('/translate_en_to_ar', methods=['POST'])
def translate_en_to_ar():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_en_to_ar(text)
    return jsonify({"translated_text": translation[0]['translation_text']})

@app.route('/translate_ar_to_fr', methods=['POST'])
def translate_ar_to_fr():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_ar_to_fr(text)
    return jsonify({"translated_text": translation[0]['translation_text']})

@app.route('/translate_fr_to_ar', methods=['POST'])
def translate_fr_to_ar():
    data = request.json  
    text = data.get('text')  
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translation = translator_fr_to_ar(text)
    return jsonify({"translated_text": translation[0]['translation_text']})



# Run the Flask server
if __name__ == '__main__':
    app.run(debug=False)