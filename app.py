from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import json
from auth import init_db, register_user, login_user, get_user_conversations, save_conversation_to_db, verify_reset_credentials, update_password
from chatbot_model import MLChatbot

app = Flask(__name__)
app.secret_key = 'habib'

# Global chatbot instance
chatbot = None

# File to store conversations
CONVERSATION_FILE = "conversations.json"

def initialize_chatbot():
    """Initialize the chatbot with automatic model training if needed"""
    global chatbot
    try:
        print("Initializing chatbot...")
        chatbot = MLChatbot()
        print("✅ Chatbot initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        # Create a fallback chatbot
        chatbot = FallbackChatbot()

class FallbackChatbot:
    """Fallback chatbot if main chatbot fails"""
    def generate_response(self, text):
        responses = [
            "I'm having some technical difficulties. Please try again later.",
            "Sorry, I'm not feeling well right now. Can you rephrase that?",
            "I'm currently learning. Could you try a simpler question?",
            "My brain is updating. Please be patient with me!"
        ]
        import random
        return random.choice(responses)
    
    def get_model_info(self):
        return {
            "model_type": "fallback",
            "status": "limited_functionality"
        }

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
    global chatbot
    
    user_input = request.json.get('message', '').strip()

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    if len(user_input) > 500:
        return jsonify({'error': 'Input too long'}), 400

    try:
        # Generate response using chatbot
        bot_response = chatbot.generate_response(user_input)
    except Exception as e:
        print(f"Error generating response: {e}")
        bot_response = "Sorry, I encountered an error. Please try again."
    
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
    global chatbot
    model_info = chatbot.get_model_info() if hasattr(chatbot, 'get_model_info') else {}
    
    return jsonify({
        'status': 'healthy', 
        'model_type': model_info.get('model_type', 'unknown'),
        'memory_optimized': True,
        'model_info': model_info
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
    global chatbot
    conversations = load_conversations()
    model_info = chatbot.get_model_info() if hasattr(chatbot, 'get_model_info') else {}
    
    return jsonify({
        'conversation_count': len(conversations),
        'memory_optimized': True,
        **model_info
    })

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model if needed"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    global chatbot
    try:
        print("Retraining model...")
        chatbot.train_model()
        return jsonify({'message': 'Model retrained successfully!'})
    except Exception as e:
        return jsonify({'error': f'Error retraining model: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Memory-Optimized Chatbot Server...")
    print("=" * 50)
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    
    # Initialize chatbot (will auto-train if model doesn't exist)
    initialize_chatbot()
    
    # Load existing conversations
    conversations = load_conversations()
    print(f"📝 Found {len(conversations)} existing conversations")
    
    print("=" * 50)
    print("✅ Server ready! Using lightweight ML model instead of transformers")
    print("💾 Memory usage optimized for Render free tier")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)