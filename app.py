from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_perplexity(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None

def detect_ai_text(text):
    perplexity = calculate_perplexity(text)
    if perplexity is None:
        return None
    
    # Thresholds calibrated for GPT-2 detection
    if perplexity < 20:
        return {
            "isHuman": False, 
            "confidence": min(0.95, (20 - perplexity)/20 * 0.95),
            "perplexity": perplexity
        }
    elif perplexity < 40:
        confidence = (40 - perplexity)/40 * 0.95
        return {
            "isHuman": np.random.choice([True, False], p=[confidence, 1-confidence]), 
            "confidence": max(confidence, 1-confidence),
            "perplexity": perplexity
        }
    else:
        return {
            "isHuman": True, 
            "confidence": min(0.95, (perplexity - 40)/40 * 0.95),
            "perplexity": perplexity
        }

def calculate_text_stats(text):
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Simple readability score (Flesch-Kincaid approximation)
    readability = 0
    if word_count > 0 and sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        avg_syllables_per_word = min(2.5, (word_count / len(text) * 10))
        readability = max(0, min(100, 
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        ))
    
    return {
        'wordCount': word_count,
        'sentenceCount': sentence_count,
        'avgWordLength': round(avg_word_length, 1),
        'readability': round(readability, 1)
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api')
def api_docs():
    return render_template('api.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detect_ai_text(text)
    if result is None:
        return jsonify({'error': 'Analysis failed'}), 500
    
    explanation = (f"The text perplexity score is {result['perplexity']:.2f}. " +
                  "Lower perplexity (typically <30) often indicates AI-generated content, " +
                  "while higher values suggest human writing.")
    
    return jsonify({
        'isHuman': result['isHuman'],
        'confidence': result['confidence'],
        'perplexity': result['perplexity'],
        'explanation': explanation,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['POST'])
def api_stats():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    return jsonify(calculate_text_stats(text))

@app.route('/api/dashboard')
def api_dashboard():
    # Mock data - in a real app, get from database
    return jsonify({
        'totalAnalyses': 127,
        'humanTexts': 84,
        'aiTexts': 43,
        'accuracyRate': '92%',
        'recentActivity': [
            {
                'type': 'ai',
                'text': 'Detected AI-generated content in "Marketing Copy.docx"',
                'time': '2 hours ago',
                'confidence': 87
            },
            {
                'type': 'human',
                'text': 'Verified human-written content in "Essay.txt"',
                'time': '5 hours ago',
                'confidence': 72
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
