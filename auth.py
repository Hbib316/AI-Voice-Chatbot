import sqlite3
from passlib.hash import bcrypt
from datetime import datetime

# Database file
DATABASE = 'chatbot.db'

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create users table with preferred_animal and preferred_color
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            preferred_animal TEXT NOT NULL,
            preferred_color TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create chat_sessions table to track conversation history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def register_user(username, password, preferred_animal, preferred_color):
    """Register a new user with hashed password, preferred animal, and color."""
    try:
        # Check if username already exists
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return {"success": False, "message": "Username already taken"}
        
        # Hash password and insert new user
        hashed_password = bcrypt.hash(password)
        cursor.execute('INSERT INTO users (username, password, preferred_animal, preferred_color) VALUES (?, ?, ?, ?)', 
                       (username, hashed_password, preferred_animal, preferred_color))
        conn.commit()
        conn.close()
        return {"success": True, "message": "User registered successfully"}
    except sqlite3.IntegrityError:
        conn.close()
        return {"success": False, "message": "Username already taken"}
    except Exception as e:
        conn.close()
        return {"success": False, "message": f"Registration error: {str(e)}"}

def login_user(username, password):
    """Verify credentials to login a user."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return {"success": False, "message": "User not found"}
        
        if not bcrypt.verify(password, user[1]):
            return {"success": False, "message": "Incorrect password"}
        
        return {"success": True, "message": "Login successful", "user_id": user[0], "username": username}
    except Exception as e:
        return {"success": False, "message": f"Login error: {str(e)}"}

def verify_reset_credentials(username, preferred_animal, preferred_color):
    """Verify username, preferred animal, and color for password reset."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ? AND preferred_animal = ? AND preferred_color = ?', 
                       (username, preferred_animal, preferred_color))
        user = cursor.fetchone()
        conn.close()
        if user:
            return {"success": True, "user_id": user[0]}
        return {"success": False, "message": "Invalid credentials"}
    except Exception as e:
        return {"success": False, "message": f"Verification error: {str(e)}"}

def update_password(user_id, new_password):
    """Update the user's password in the database."""
    try:
        hashed_password = bcrypt.hash(new_password)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_password, user_id))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Password updated successfully"}
    except Exception as e:
        return {"success": False, "message": f"Password update error: {str(e)}"}

def save_conversation_to_db(user_id, user_message, bot_response):
    """Save conversation to database."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, user_message, bot_response) 
            VALUES (?, ?, ?)
        ''', (user_id, user_message, bot_response))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")
        return False

def get_user_conversations(user_id, limit=10):
    """Get recent conversations for a user."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_message, bot_response, timestamp 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        conversations = cursor.fetchall()
        conn.close()
        return conversations
    except Exception as e:
        print(f"Error retrieving conversations: {str(e)}")
        return []

def clear_user_conversations(user_id):
    """Clear all conversations for a user."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing conversations: {str(e)}")
        return False