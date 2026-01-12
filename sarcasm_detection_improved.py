import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
df = pd.read_csv("train-balanced-sarcasm.csv")

# Data exploration
print("\n=== Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nClass distribution:")
print(df['label'].value_counts())

# Handle missing values
print("\n=== Data Cleaning ===")
initial_rows = len(df)
df = df.dropna(subset=['comment', 'label'])
print(f"Rows removed due to missing values: {initial_rows - len(df)}")

# Enhanced text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    # Remove special characters but keep some punctuation that might indicate sarcasm
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply text cleaning
df['clean_text'] = df['comment'].apply(clean_text)

# Remove empty strings after cleaning
df = df[df['clean_text'].str.len() > 0]
print(f"Final dataset shape: {df.shape}")

# Split data into features and labels
X = df['clean_text']
y = df['label']

# Split data into train and test (80% train, 20% test)
print("\n=== Splitting Data ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Vectorization with TF-IDF (with improved parameters)
print("\n=== Vectorizing Text ===")
vectorizer = TfidfVectorizer(
    max_features=5000,      # 5000 most common words
    min_df=2,               # Ignore terms that appear in less than 2 documents
    max_df=0.95,            # Ignore terms that appear in more than 95% of documents
    ngram_range=(1, 2)      # Use unigrams and bigrams
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_vec.shape}")

# Train baseline model
print("\n=== Training Baseline Model ===")
baseline_model = MultinomialNB()
baseline_model.fit(X_train_vec, y_train)
y_pred_baseline = baseline_model.predict(X_test_vec)

print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

# Hyperparameter tuning
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}
grid_search = GridSearchCV(
    MultinomialNB(), 
    param_grid, 
    cv=5, 
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train_vec, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model
print("\n=== Evaluating Tuned Model ===")
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_vec)

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")
print(f"Improvement over baseline: {(tuned_accuracy - accuracy_score(y_test, y_pred_baseline)):.4f}")
print("\nTuned Model Classification Report:")
print(classification_report(y_test, y_pred_tuned))

# Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred_tuned)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix - Sarcasm Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png' in the current directory")

# Test the model with example predictions
print("\n=== Example Predictions ===")
test_examples = [
    "This is the best day ever!",
    "Oh great, another meeting. Just what I needed.",
    "I love working on weekends, said no one ever.",
    "The weather is beautiful today."
]

test_examples_vec = vectorizer.transform(test_examples)
predictions = best_model.predict(test_examples_vec)
probabilities = best_model.predict_proba(test_examples_vec)

for text, pred, prob in zip(test_examples, predictions, probabilities):
    label = "Sarcastic" if pred == 1 else "Not Sarcastic"
    confidence = prob[pred] * 100
    print(f"\nText: '{text}'")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")

print("\n=== Model Training Complete ===")
