# app.py - Complete Diabetes Prediction System with Beautiful UI
from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import tensorflow as tf
import joblib
import sqlite3
import hashlib
from functools import wraps
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'

print("="*70)
print("DIABETES PREDICTION SYSTEM")
print("="*70)

# ==================== LOAD MODEL AND SCALER ====================
print("\n" + "="*70)
print("LOADING MODEL AND SCALER")
print("="*70)

# Load model
model = None
if os.path.exists('models/BCANN_model.h5'):
    try:
        model = tf.keras.models.load_model('models/BCANN_model.h5')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)
else:
    print("❌ Model file not found!")
    exit(1)

# Load scaler
scaler = None
if os.path.exists('models/scaler.pkl'):
    try:
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Scaler loaded successfully!")
        
        # Verify scaler is trained
        if hasattr(scaler, 'mean_'):
            print(f"   Scaler is trained with {len(scaler.mean_)} features")
            print(f"   Feature means: {scaler.mean_}")
        else:
            print("⚠️ Warning: Scaler is not trained!")
            print("Please run train_scaler_and_verify.py first!")
            exit(1)
            
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        print("Please run train_scaler_and_verify.py first!")
        exit(1)
else:
    print("❌ Scaler file not found!")
    print("Please run train_scaler_and_verify.py first!")
    exit(1)

print("\n" + "="*70)
print("✅ System Ready!")
print("="*70 + "\n")

# ==================== DATABASE SETUP ====================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if len(username) < 3:
            flash('Username must be at least 3 characters', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('signup'))
        
        hashed_password = hash_password(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                     (username, hashed_password, email))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        hashed_password = hash_password(password)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", 
                 (username, hashed_password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('main'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
@login_required
def main():
    if request.method == 'POST':
        try:
            # Get form data
            features = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['hypertension']),
                float(request.form['heart_disease']),
                float(request.form['smoking_history']),
                float(request.form['bmi']),
                float(request.form['HbA1c_level']),
                float(request.form['blood_glucose_level'])
            ]
            
            # Validate input ranges
            if features[1] < 0 or features[1] > 120:
                flash('Age must be between 0 and 120', 'error')
                return redirect(url_for('main'))
            
            if features[5] < 10 or features[5] > 50:
                flash('BMI must be between 10 and 50', 'error')
                return redirect(url_for('main'))
            
            if features[6] < 3 or features[6] > 15:
                flash('HbA1c level must be between 3 and 15', 'error')
                return redirect(url_for('main'))
            
            if features[7] < 50 or features[7] > 300:
                flash('Blood glucose level must be between 50 and 300', 'error')
                return redirect(url_for('main'))
            
            # Create dataframe with features
            feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 
                           'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            features_array = np.array([features])
            
            # Scale features using trained scaler
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled, verbose=0)[0][0]
            
            # Interpret result
            if prediction > 0.5:
                result = "POSITIVE for diabetes"
                confidence = f"{prediction*100:.2f}%"
                result_class = "positive"
                advice = "⚠️ Please consult a healthcare provider for proper diagnosis and treatment."
                risk_level = "High Risk"
                icon = "fa-exclamation-triangle"
            else:
                result = "NEGATIVE for diabetes"
                confidence = f"{(1-prediction)*100:.2f}%"
                result_class = "negative"
                advice = "✅ Continue maintaining a healthy lifestyle. Regular check-ups are recommended."
                risk_level = "Low Risk"
                icon = "fa-check-circle"
            
            # Get risk factors
            risk_factors = []
            if features[5] > 25:  # BMI > 25
                risk_factors.append("High BMI")
            if features[6] > 5.7:  # HbA1c > 5.7
                risk_factors.append("Elevated HbA1c")
            if features[7] > 100:  # Blood glucose > 100
                risk_factors.append("High Blood Glucose")
            if features[2] == 1:  # Hypertension
                risk_factors.append("Hypertension")
            if features[3] == 1:  # Heart Disease
                risk_factors.append("Heart Disease")
            
            return render_template('main.html', 
                                 result=result, 
                                 confidence=confidence,
                                 result_class=result_class,
                                 advice=advice,
                                 risk_level=risk_level,
                                 risk_factors=risk_factors,
                                 icon=icon,
                                 form_data=request.form)
        
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')
            return redirect(url_for('main'))
    
    return render_template('main.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username, email, created_at FROM users WHERE id=?", (session['user_id'],))
    user = c.fetchone()
    conn.close()
    return render_template('profile.html', user=user)

@app.route('/check_status')
def check_status():
    """Check system status"""
    return {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'scaler_trained': hasattr(scaler, 'mean_') if scaler else False,
        'model_file_exists': os.path.exists('models/BCANN_model.h5'),
        'scaler_file_exists': os.path.exists('models/scaler.pkl')
    }

if __name__ == '__main__':
    init_db()
    print("\n" + "="*70)
    print("🚀 DIABETES PREDICTION SYSTEM STARTED")
    print("="*70)
    print(f"📍 Access at: http://127.0.0.1:5000")
    print(f"🔍 Status check: http://127.0.0.1:5000/check_status")
    print(f"🔴 Press CTRL+C to stop the server")
    print("="*70 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)