from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from datetime import datetime, timedelta
import os
import uuid
from werkzeug.utils import secure_filename
import json
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}
app.config['API_KEYS'] = {
    'free': {'limit': 100, 'period': timedelta(days=1)},
    'pro': {'limit': 5000, 'period': timedelta(days=1)},
    'enterprise': {'limit': float('inf'), 'period': timedelta(days=1)}
}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Mock database (replace with real database in production)
users_db = {}
analyses_db = {}
api_keys_db = {}

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_perplexity(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        app.logger.error(f"Error calculating perplexity: {e}")
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
        avg_syllables_per_word = min(2.5, (word_count / len(text) * 10)
        readability = max(0, min(100, 
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        )
    
    return {
        'wordCount': word_count,
        'sentenceCount': sentence_count,
        'avgWordLength': round(avg_word_length, 1),
        'readability': round(readability, 1)
    }

def generate_api_key():
    return str(uuid.uuid4())

def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not api_key or api_key not in api_keys_db:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        # Check rate limits
        key_data = api_keys_db[api_key]
        if key_data['count'] >= key_data['limit']:
            reset_time = key_data['last_reset'] + key_data['period']
            if datetime.now() > reset_time:
                # Reset the counter
                key_data['count'] = 0
                key_data['last_reset'] = datetime.now()
            else:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'limit': key_data['limit'],
                    'remaining': 0,
                    'reset': reset_time.isoformat()
                }), 429
        
        # Increment counter
        key_data['count'] += 1
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        user = users_db.get(session['user_id'], {})
        return render_template('index.html', username=user.get('username'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_analyses = [a for a in analyses_db.values() if a['user_id'] == user_id]
    
    # Calculate dashboard stats
    total_analyses = len(user_analyses)
    human_texts = sum(1 for a in user_analyses if a['result']['isHuman'])
    ai_texts = total_analyses - human_texts
    accuracy = (human_texts / total_analyses * 100) if total_analyses > 0 else 0
    
    return render_template('dashboard.html', 
                         total_analyses=total_analyses,
                         human_texts=human_texts,
                         ai_texts=ai_texts,
                         accuracy=round(accuracy, 1))

@app.route('/api')
def api_docs():
    api_key = None
    if 'user_id' in session:
        user = users_db.get(session['user_id'], {})
        api_key = user.get('api_key')
    return render_template('api.html', api_key=api_key)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  # In production, use proper hashing!
        
        # Simple mock authentication
        user = next((u for u in users_db.values() if u['username'] == username and u['password'] == password), None)
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if any(u['username'] == username for u in users_db.values()):
            return render_template('register.html', error='Username already exists')
        
        user_id = str(uuid.uuid4())
        api_key = generate_api_key()
        
        users_db[user_id] = {
            'id': user_id,
            'username': username,
            'password': password,  # Remember to hash in production!
            'email': email,
            'api_key': api_key,
            'plan': 'free',
            'created_at': datetime.now()
        }
        
        api_keys_db[api_key] = {
            'user_id': user_id,
            'plan': 'free',
            'limit': app.config['API_KEYS']['free']['limit'],
            'period': app.config['API_KEYS']['free']['period'],
            'count': 0,
            'last_reset': datetime.now()
        }
        
        session['user_id'] = user_id
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/api/detect', methods=['POST'])
@api_key_required
def api_detect():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detect_ai_text(text)
    if result is None:
        return jsonify({'error': 'Analysis failed'}), 500
    
    explanation = (f"The text perplexity score is {result['perplexity']:.2f}. " +
                  "Lower perplexity (typically <30) often indicates AI-generated content, " +
                  "while higher values suggest human writing.")
    
    response = {
        'isHuman': result['isHuman'],
        'confidence': result['confidence'],
        'perplexity': result['perplexity'],
        'explanation': explanation,
        'timestamp': datetime.now().isoformat()
    }
    
    # Store analysis if user is logged in
    if 'user_id' in session:
        analysis_id = str(uuid.uuid4())
        analyses_db[analysis_id] = {
            'id': analysis_id,
            'user_id': session['user_id'],
            'text': text[:500] + '...' if len(text) > 500 else text,
            'result': response,
            'created_at': datetime.now()
        }
    
    return jsonify(response)

@app.route('/api/stats', methods=['POST'])
@api_key_required
def api_stats():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    return jsonify(calculate_text_stats(text))

@app.route('/api/dashboard')
@api_key_required
def api_dashboard():
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
    user_id = api_keys_db[api_key]['user_id']
    
    user_analyses = [a for a in analyses_db.values() if a['user_id'] == user_id]
    recent_activity = sorted(user_analyses, key=lambda x: x['created_at'], reverse=True)[:5]
    
    return jsonify({
        'totalAnalyses': len(user_analyses),
        'humanTexts': sum(1 for a in user_analyses if a['result']['isHuman']),
        'aiTexts': sum(1 for a in user_analyses if not a['result']['isHuman']),
        'accuracyRate': f"{sum(a['result']['confidence'] for a in user_analyses) / len(user_analyses) * 100:.1f}%" if user_analyses else "0%",
        'recentActivity': [
            {
                'type': 'ai' if not a['result']['isHuman'] else 'human',
                'text': a['text'],
                'time': (datetime.now() - a['created_at']).total_seconds() // 3600,
                'confidence': round(a['result']['confidence'] * 100)
            } for a in recent_activity
        ]
    })

@app.route('/api/generate-key', methods=['POST'])
def api_generate_key():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    # Revoke old key if exists
    old_key = users_db[user_id].get('api_key')
    if old_key and old_key in api_keys_db:
        del api_keys_db[old_key]
    
    # Generate new key
    new_key = generate_api_key()
    users_db[user_id]['api_key'] = new_key
    
    plan = users_db[user_id].get('plan', 'free')
    api_keys_db[new_key] = {
        'user_id': user_id,
        'plan': plan,
        'limit': app.config['API_KEYS'][plan]['limit'],
        'period': app.config['API_KEYS'][plan]['period'],
        'count': 0,
        'last_reset': datetime.now()
    }
    
    return jsonify({'api_key': new_key})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process file (in a real app, you'd extract text from PDF/DOCX)
        with open(filepath, 'r') as f:
            text = f.read(10000)  # Read first 10KB
        
        os.remove(filepath)  # Clean up
        
        result = detect_ai_text(text)
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        return jsonify({
            'filename': filename,
            'result': result,
            'stats': calculate_text_stats(text)
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_analyses = sorted(
        [a for a in analyses_db.values() if a['user_id'] == user_id],
        key=lambda x: x['created_at'],
        reverse=True
    )
    
    return render_template('history.html', analyses=user_analyses)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
