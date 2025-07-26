from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlite3
import os
import json
# Remove this line:
# from passlib.hash import bcrypt

# Change the import from auth to:
from auth import init_db, register_user, login_user, get_user_conversations, save_conversation_to_db, verify_reset_credentials, update_password


app = Flask(__name__)
app.secret_key = 'habib'

# Model configuration
FINE_TUNED_MODEL_PATH = "fine_tuned_model"
# BASE_MODEL_NAME = "microsoft/DialoGPT-medium"
BASE_MODEL_NAME = "microsoft/DialoGPT-small"  
# Global model variables
tokenizer = None
model = None
is_fine_tuned = False

# Load model and tokenizer
def load_model():
    global tokenizer, model, is_fine_tuned
    
    if os.path.exists(FINE_TUNED_MODEL_PATH) and os.path.exists(os.path.join(FINE_TUNED_MODEL_PATH, "config.json")):
        print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        is_fine_tuned = True
    else:
        print(f"Fine-tuned model not found. Loading base model: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        is_fine_tuned = False
    tokenizer.pad_token="[PAD]"
    
    print(f"Model loaded successfully: {'Fine-tuned' if is_fine_tuned else 'Base'}")
    return tokenizer, model, is_fine_tuned

# Initialize model and tokenizer
load_model()

# File to store conversations
CONVERSATION_FILE = "conversations.json"

def get_user_chat_history():
    if 'chat_history_ids' not in session:
        session['chat_history_ids'] = None
    return session['chat_history_ids']

def set_user_chat_history(chat_history_ids):
    if chat_history_ids is not None:
        session['chat_history_ids'] = chat_history_ids.tolist()
    else:
        session['chat_history_ids'] = None

def tensor_from_session(chat_history_list):
    if chat_history_list is None:
        return None
    return torch.tensor(chat_history_list)

def generate_response(user_input, chat_history_ids=None):
    global model, tokenizer
    
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        early_stopping=True
    )

    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response, chat_history_ids

def load_intents_data():
    """Load and process intents.json file"""
    try:
        with open("intents.json", "r") as f:
            intents_data = json.load(f)
        return intents_data
    except FileNotFoundError:
        print("intents.json not found!")
        return {"intents": []}
    except json.JSONDecodeError:
        print("Error parsing intents.json!")
        return {"intents": []}

def convert_intents_to_conversations():
    """Convert intents.json to conversation format for fine-tuning"""
    intents_data = load_intents_data()
    conversations = []
    
    for intent in intents_data.get("intents", []):
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        
        # Create conversations by pairing each pattern with each response
        for pattern in patterns:
            for response in responses:
                # Clean HTML tags from responses for training
                clean_response = response.replace('<a target="_blank" href="', '').replace('">', ' ').replace('</a>', '')
                conversations.append({
                    "user": pattern.strip(),
                    "bot": clean_response.strip(),
                    "tag": intent.get("tag", "unknown")
                })
    
    return conversations

def load_conversations():
    """Load conversations from both sources"""
    conversations = []
    
    # First try to load from existing conversations.json
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r") as f:
                conversations.extend(json.load(f))
        except:
            pass
    
    # Then add conversations from intents.json
    intent_conversations = convert_intents_to_conversations()
    conversations.extend(intent_conversations)
    
    return conversations

def save_conversation(user_input, bot_response):
    conversations = []
    
    # Load existing conversations
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r") as f:
                conversations = json.load(f)
        except:
            conversations = []
    
    # Add new conversation
    conversations.append({"user": user_input, "bot": bot_response})
    
    # Save back to file
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(conversations, f, indent=2)
    
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
            session['chat_history_ids'] = None
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

    chat_history_list = get_user_chat_history()
    chat_history_ids = tensor_from_session(chat_history_list)

    bot_response, new_chat_history_ids = generate_response(user_input, chat_history_ids)

    set_user_chat_history(new_chat_history_ids)
    save_conversation(user_input, bot_response)

    return jsonify({'response': bot_response})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session['chat_history_ids'] = None
    
    return jsonify({'message': 'Chat history cleared'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'fine_tuned': is_fine_tuned,
        'model_path': FINE_TUNED_MODEL_PATH if is_fine_tuned else BASE_MODEL_NAME
    })

@app.route('/fine_tune', methods=['POST'])
def trigger_fine_tuning():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conversations = load_conversations()
        if len(conversations) < 10:
            return jsonify({'error': f'Not enough conversation data for fine-tuning (minimum 10 required, found {len(conversations)})'}), 400
        
        result = fine_tune_model()
        
        if result['success']:
            load_model()
            session.clear()
            
            return jsonify({
                'message': 'Fine-tuning completed successfully. Please log in again.',
                'model_path': FINE_TUNED_MODEL_PATH,
                'redirect': '/login'
            })
        else:
            return jsonify({'error': f'Fine-tuning failed: {result["error"]}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Fine-tuning error: {str(e)}'}), 500

def fine_tune_model():
    try:
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        conversations = load_conversations()
        if len(conversations) < 10:
            return {'success': False, 'error': f'Not enough conversation data (found {len(conversations)}, need at least 10)'}
        
        print(f"Starting fine-tuning with {len(conversations)} conversations...")
        
        # Create training texts
        texts = []
        for conv in conversations:
            # Format: user_input<eos>bot_response<eos>
            text = conv["user"] + tokenizer.eos_token + conv["bot"] + tokenizer.eos_token
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Set labels for language modeling
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True
        )
        
        # Create output directory
        os.makedirs(FINE_TUNED_MODEL_PATH, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_PATH,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=True,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=2,
            fp16=True,
            learning_rate=5e-5,
            weight_decay=0.01,
            report_to=None,  # Disable wandb and other reporting
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        print(f"Saving fine-tuned model to {FINE_TUNED_MODEL_PATH}")
        model.save_pretrained(FINE_TUNED_MODEL_PATH)
        tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
        
        print("Fine-tuning completed successfully!")
        return {'success': True, 'model_path': FINE_TUNED_MODEL_PATH}
        
    except Exception as e:
        print(f"Fine-tuning error: {str(e)}")
        return {'success': False, 'error': str(e)}

@app.route('/model_info')
def model_info():
    conversations = load_conversations()
    intents_data = load_intents_data()
    
    return jsonify({
        'is_fine_tuned': is_fine_tuned,
        'model_path': FINE_TUNED_MODEL_PATH if is_fine_tuned else BASE_MODEL_NAME,
        'conversation_count': len(conversations),
        'intents_count': len(intents_data.get('intents', [])),
        'fine_tuned_model_exists': os.path.exists(FINE_TUNED_MODEL_PATH),
        'intents_file_exists': os.path.exists('intents.json')
    })

@app.route('/reload_model', methods=['POST'])
def reload_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        load_model()
        return jsonify({
            'message': 'Model reloaded successfully',
            'is_fine_tuned': is_fine_tuned,
            'model_path': FINE_TUNED_MODEL_PATH if is_fine_tuned else BASE_MODEL_NAME
        })
    except Exception as e:
        return jsonify({'error': f'Error reloading model: {str(e)}'}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session['chat_history_ids'] = None
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

# New route to view intents data
@app.route('/api/intents')
def api_intents():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    intents_data = load_intents_data()
    return jsonify(intents_data)

if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    
    print(f"Model loaded: {'Fine-tuned' if is_fine_tuned else 'Base'} model")
    print(f"Model path: {FINE_TUNED_MODEL_PATH if is_fine_tuned else BASE_MODEL_NAME}")
    
    # Load and display conversation data info
    conversations = load_conversations()
    intents_data = load_intents_data()
    
    print(f"Found {len(conversations)} total conversations for training")
    print(f"Found {len(intents_data.get('intents', []))} intents in intents.json")
    
    if len(conversations) >= 10:
        print("Sufficient training data found. You can trigger fine-tuning via /fine_tune endpoint")
    else:
        print(f"Need at least 10 conversations for fine-tuning (current: {len(conversations)})")
        print("The app will use the base model for now.")
    
    # app.run(debug=True, host="0.0.0.0", port=5000)
    port = int(os.environ.get("PORT", 5000))  # ← Critical for Render
    app.run(host='0.0.0.0', port=port)