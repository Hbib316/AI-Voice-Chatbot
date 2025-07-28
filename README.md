AI Chatbot
A lightweight, memory-optimized chatbot application specializing in Machine Learning (ML) and Deep Learning (DL) topics. Built with Flask and Scikit-learn, this chatbot provides an interactive platform for users to learn about ML/DL concepts, algorithms, frameworks, and more. It includes user authentication, conversation history, and password reset functionality, all optimized for deployment on Render's free tier.
Features

User Authentication: Secure user registration and login with password hashing using Werkzeug.
Conversation History: Stores user conversations in a SQLite database and a JSON file for persistence.
Password Reset: Allows users to reset their password using preferred animal and color as security credentials.
ML/DL Knowledge Base: Responds to queries about ML/DL concepts, algorithms, frameworks, career advice, and project ideas using a pre-trained Scikit-learn model.
Memory Optimization: Designed to run efficiently on resource-constrained environments like Render's free tier.
Health Check: Provides a /health endpoint to monitor chatbot status and model information.
Model Retraining: Supports retraining the chatbot model via an API endpoint.
Fallback Mechanism: Includes a fallback chatbot for robust operation in case of initialization failures.

Prerequisites

Python 3.9.16
SQLite
Dependencies listed in requirements.txt



Install Dependencies:
pip install -r requirements.txt

Start the Application:
python app.py

Alternatively, for production:
gunicorn -c gunicorn.conf.py app:app



Usage

Access the Application:Open your browser and navigate to [http://localhost:5000 (or the deployed URL if hosted).](https://ai-voice-chatbot-oa9o.onrender.com)

Register/Login:

Register a new account with a username, password, preferred animal, and preferred color.
Log in to access the chatbot interface.


Interact with the Chatbot:

Ask questions about ML/DL topics (e.g., "What is machine learning?", "Explain neural networks").
View conversation history via the /history endpoint.
Start a new chat session or clear history as needed.


Password Reset:

Use the /forgot_password endpoint to reset your password by providing your username, preferred animal, and preferred color.


API Endpoints:

/chat: POST to send a message and receive a chatbot response.
/api/history: GET to retrieve conversation history.
/clear_history: POST to clear conversation history.
/health: GET to check the application's health and model status.
/retrain_model: POST to retrain the chatbot model.
/reinitialize_chatbot: POST to reinitialize the chatbot.



Project Structure

app.py: Main Flask application with routes and chatbot logic.
auth.py: Handles user authentication, database operations, and conversation storage.
chatbot_model.py: Implements the ML chatbot using Scikit-learn's TF-IDF and Logistic Regression.
intents.json: Defines intents and responses for the chatbot's knowledge base.
gunicorn.conf.py: Configuration for Gunicorn to optimize performance in production.
render.yaml: Configuration for deploying the application on Render.
requirements.txt: Lists Python dependencies.
templates/: Contains HTML templates for the web interface (e.g., index.html, login.html, register.html, forgot_password.html, history.html).
static/: Stores static assets like CSS and JavaScript files.


Model Details

Model Type: Scikit-learn pipeline with TF-IDF vectorizer and Logistic Regression classifier.
Training Data: Defined in intents.json with patterns and responses for various ML/DL topics.
Training Process: The model is trained on startup if no pre-trained model exists. It can be retrained via the /retrain_model endpoint.
Storage: The trained model is saved as intent_model.pkl in the model/ directory.




Built with Flask, Scikit-learn, and SQLite for lightweight performance.
Inspired by the need for an accessible ML/DL learning assistant.
Deployed on Render for free-tier compatibility.
