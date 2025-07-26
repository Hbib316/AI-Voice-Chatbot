from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import json
import requests
from auth import init_db, register_user, login_user, get_user_conversations, save_conversation_to_db, verify_reset_credentials, update_password

app = Flask(__name__)
app.secret_key = 'habib'

class HuggingFaceChatbot:
    def __init__(self):
        self.api_token = os.environ.get('HUGGINGFACE_API_TOKEN')  # Set this in Render
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def generate_response(self, user_input):
        """Generate response using Hugging Face Inference API"""
        if not self.api_token:
            return "API token not configured"
        
        try:
            payload = {"inputs": user_input}
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No response generated')
                return "No response generated"
            else:
                return "API request failed"
        except Exception as e:
            return "Error generating response"

# Usage in your app:
chatbot = HuggingFaceChatbot() 

# File to store conversations
CONVERSATION_FILE = "conversations.json"

def load_conversations():
    """Load conversations from file"""
    conversations = []
    
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r") as f:
                conversations.extend(json.load(f))
        except:
            pass
    
    return conversations

def save_conversation(user_input, bot_response):
    """Save conversation to file and database"""
    conversations = load_conversations()
    conversations.append({"user": user_input, "bot": bot_response})
    
    # Keep only last 100 conversations to save space
    if len(conversations) > 100:
        conversations = conversations[-100:]
    
    try:
        with open(CONVERSATION_FILE, "w") as f:
            json.dump(conversations, f, indent=2)
    except:
        pass
    
    # Save to database if user is logged in
    if 'user_id' in session:
        save_conversation_to_db(session['user_id'], user_input, bot_response)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        result = login_user(username, password)
        if result['success']:
            session['user_id'] = result['user_id']
            session['username'] = result['username']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=result['message'])
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        preferred_animal = request.form.get('preferred_animal')
        preferred_color = request.form.get('preferred_color')
        
        if not preferred_animal or not preferred_color:
            return render_template('register.html', error="Preferred animal and color are required")
        
        result = register_user(username, password, preferred_animal, preferred_color)
        if result['success']:
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error=result['message'])
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')

@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    conversations = get_user_conversations(user_id, limit=20)
    
    return jsonify({
        'conversations': [
            {
                'user_message': conv[0],
                'bot_response': conv[1],
                'timestamp': conv[2]
            } for conv in conversations
        ],
        'total_conversations': len(conversations),
        'last_activity': conversations[0][2] if conversations else 'Never'
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    if len(user_input) > 500:
        return jsonify({'error': 'Input too long'}), 400

    # Generate response using simple chatbot
    bot_response = chatbot.generate_response(user_input)
    
    # Save conversation
    save_conversation(user_input, bot_response)

    return jsonify({'response': bot_response})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify({'message': 'Chat history cleared'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_type': 'rule-based',
        'memory_optimized': True
    })

@app.route('/new_chat', methods=['POST'])
def new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify({'message': 'New chat session started'})

@app.route('/api/search_history', methods=['GET'])
def search_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    keyword = request.args.get('keyword', '').strip()
    date_filter = request.args.get('date', '')
    
    try:
        with sqlite3.connect('chatbot.db') as conn:
            cursor = conn.cursor()
            query = "SELECT user_message, bot_response, timestamp FROM conversations WHERE user_id = ?"
            params = [session['user_id']]
            
            if keyword:
                query += " AND (user_message LIKE ? OR bot_response LIKE ?)"
                params.extend([f'%{keyword}%', f'%{keyword}%'])
            
            if date_filter:
                query += " AND DATE(timestamp) = ?"
                params.append(date_filter)
            
            query += " ORDER BY timestamp DESC LIMIT 20"
            cursor.execute(query, params)
            conversations = cursor.fetchall()
            
            return jsonify({
                'conversations': [
                    {
                        'user_message': conv[0],
                        'bot_response': conv[1],
                        'timestamp': conv[2]
                    } for conv in conversations
                ],
                'total_conversations': len(conversations),
                'last_activity': conversations[0][2] if conversations else 'Never'
            })
    except Exception as e:
        return jsonify({'error': f'Error searching history: {str(e)}'}), 500

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        preferred_animal = request.form.get('preferred_animal')
        preferred_color = request.form.get('preferred_color')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, preferred_animal, preferred_color, new_password, confirm_password]):
            return render_template('forgot_password.html', error="All fields are required")
        
        if new_password != confirm_password:
            return render_template('forgot_password.html', error="Passwords do not match")
        
        if len(new_password) < 8:
            return render_template('forgot_password.html', error="Password must be at least 8 characters long")
        
        result = verify_reset_credentials(username, preferred_animal, preferred_color)
        if not result['success']:
            return render_template('forgot_password.html', error=result['message'])
        
        update_result = update_password(result['user_id'], new_password)
        if update_result['success']:
            return redirect(url_for('login', message="Password reset successfully"))
        else:
            return render_template('forgot_password.html', error=update_result['message'])
    
    return render_template('forgot_password.html', message=request.args.get('message'))

@app.route('/model_info')
def model_info():
    conversations = load_conversations()
    
    return jsonify({
        'is_fine_tuned': False,
        'model_type': 'rule-based',
        'conversation_count': len(conversations),
        'memory_optimized': True
    })

# Remove fine-tuning related endpoints to save memory
# @app.route('/fine_tune', methods=['POST']) - REMOVED
# @app.route('/reload_model', methods=['POST']) - REMOVED

if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    
    print("Memory-optimized chatbot initialized")
    print("Using rule-based responses instead of transformer models")
    
    conversations = load_conversations()
    print(f"Found {len(conversations)} conversations")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)