from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])
    text = ' '.join(text.split())
    return text

def train_and_save_model():
    """Train the model and save it"""
    print("Training model...")
    
    # Load dataset
    df = pd.read_csv("train-balanced-sarcasm.csv")
    df = df.dropna(subset=['comment', 'label'])
    
    # Clean text
    df['clean_text'] = df['comment'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]
    
    # Split data
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize
    global vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model with hyperparameter tuning
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
    grid_search = GridSearchCV(
        MultinomialNB(), 
        param_grid, 
        cv=5, 
        n_jobs=-1,
        scoring='accuracy'
    )
    grid_search.fit(X_train_vec, y_train)
    
    global model
    model = grid_search.best_estimator_
    
    # Save model and vectorizer
    with open('sarcasm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model trained and saved successfully!")
    return model, vectorizer

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    try:
        with open('sarcasm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        model, vectorizer = train_and_save_model()

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on input text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Please enter some text'
            }), 400
        
        # Clean and vectorize the text
        cleaned_text = clean_text(text)
        text_vec = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        
        # Get confidence scores
        confidence = probabilities[prediction] * 100
        
        result = {
            'text': text,
            'prediction': 'Sarcastic' if prediction == 1 else 'Not Sarcastic',
            'is_sarcastic': bool(prediction == 1),
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_sarcastic': round(probabilities[0] * 100, 2),
                'sarcastic': round(probabilities[1] * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load or train the model
    load_model()
    
    # Run the Flask app
    print("\n" + "="*50)
    print("🚀 Sarcasm Detection Web App is running!")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)
