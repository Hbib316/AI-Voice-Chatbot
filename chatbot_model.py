import json
import random
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

class MLChatbot:
    def __init__(self, model_path="model"):
        self.model_path = model_path
        self.model_file = os.path.join(model_path, "intent_model.pkl")
        self.intents = self.load_intents()
        
        # Load or train model
        if os.path.exists(self.model_file):
            self.load_model()
        else:
            print("Model not found. Training new model...")
            self.train_model()

    def load_intents(self):
        """Load intents from JSON file"""
        try:
            with open("intents.json") as f:
                return json.load(f)["intents"]
        except FileNotFoundError:
            print("Warning: intents.json not found. Using default intents.")
            return self.get_default_intents()

    def get_default_intents(self):
        """Default intents if file is missing"""
        return [
            {
                "tag": "greeting",
                "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good afternoon"],
                "responses": ["Hello! How can I help you?", "Hi there!", "Hey! What's up?"]
            },
            {
                "tag": "goodbye",
                "patterns": ["Bye", "Goodbye", "See you later", "Talk to you later"],
                "responses": ["Goodbye!", "See you later!", "Have a great day!"]
            },
            {
                "tag": "thanks",
                "patterns": ["Thanks", "Thank you", "Thanks a lot", "I appreciate it"],
                "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
            },
            {
                "tag": "default",
                "patterns": ["What can you do", "Help", "What are your features"],
                "responses": ["I'm a chatbot that can help you with basic conversations!"]
            }
        ]

    def preprocess_text(self, text):
        """Simple text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def train_model(self):
        """Train the intent classification model"""
        print("Training intent classifier...")
        
        # Prepare training data
        texts, labels = [], []
        tag_to_id = {}
        
        for i, intent in enumerate(self.intents):
            tag = intent["tag"]
            tag_to_id[tag] = i
            
            for pattern in intent["patterns"]:
                texts.append(self.preprocess_text(pattern))
                labels.append(i)
        
        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Train the model
        self.pipeline.fit(texts, labels)
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        
        # Save the model
        self.save_model()
        print("✅ Model trained and saved successfully!")

    def save_model(self):
        """Save the trained model to disk"""
        os.makedirs(self.model_path, exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'tag_to_id': self.tag_to_id,
            'id_to_tag': self.id_to_tag
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.tag_to_id = model_data['tag_to_id']
            self.id_to_tag = model_data['id_to_tag']
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            self.train_model()

    def predict_intent(self, text):
        """Predict intent for given text"""
        try:
            processed_text = self.preprocess_text(text)
            predicted_id = self.pipeline.predict([processed_text])[0]
            return self.id_to_tag[predicted_id]
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return "default"

    def generate_response(self, text):
        """Generate response based on predicted intent"""
        if not text or not text.strip():
            return "I didn't catch that. Could you please say something?"
        
        intent_tag = self.predict_intent(text)
        
        # Find matching intent and return random response
        for intent in self.intents:
            if intent["tag"] == intent_tag:
                return random.choice(intent["responses"])
        
        # Fallback response
        return "Sorry, I didn't understand that. Could you rephrase your question?"

    def get_model_info(self):
        """Get information about the current model"""
        return {
            "model_type": "sklearn_pipeline",
            "vectorizer": "TfIdf",
            "classifier": "LogisticRegression",
            "num_intents": len(self.intents),
            "model_exists": os.path.exists(self.model_file)
        }