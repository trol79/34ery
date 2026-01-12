# 🤖 Sarcasm Detection Web Application

An AI-powered web application that detects sarcasm in text using Machine Learning (Naive Bayes classifier with TF-IDF).

## 📋 Features

- Real-time sarcasm detection
- Confidence scores
- Beautiful, modern UI
- Example sentences to try
- Easy to use interface

## 🚀 Setup Instructions

### Step 1: Install Dependencies

Open your terminal (Command Prompt or PowerShell in Windows) and run:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Make sure you have the `train-balanced-sarcasm.csv` file in the same folder as `app.py`.

### Step 3: Run the Application

Run the Flask app:

```bash
python app.py
```

The first time you run it, the app will automatically train the model (this takes a few minutes). The model will be saved as `sarcasm_model.pkl` and `vectorizer.pkl` for faster loading next time.

### Step 4: Open in Browser

Once you see the message:
```
🚀 Sarcasm Detection Web App is running!
📱 Open your browser and go to: http://127.0.0.1:5000
```

Open your web browser and navigate to: **http://127.0.0.1:5000**

## 💻 Usage

1. Type or paste your text in the input box
2. Click "Detect Sarcasm" or press Enter
3. See the results with confidence score!

## 📁 Project Structure

```
your-project/
│
├── app.py                          # Flask backend
├── templates/
│   └── index.html                  # Frontend HTML
├── requirements.txt                # Python dependencies
├── train-balanced-sarcasm.csv     # Training dataset
├── sarcasm_model.pkl              # Trained model (generated)
└── vectorizer.pkl                  # Text vectorizer (generated)
```

## 🛠️ Troubleshooting

**Problem: "Module not found" error**
- Solution: Make sure you installed all dependencies with `pip install -r requirements.txt`

**Problem: "train-balanced-sarcasm.csv not found"**
- Solution: Make sure the CSV file is in the same folder as app.py

**Problem: Port 5000 already in use**
- Solution: Change the port in app.py: `app.run(debug=True, port=5001)`

**Problem: Model training takes too long**
- Solution: The first run trains the model which can take 5-10 minutes. Subsequent runs will be instant as it loads the saved model.

## 🎨 Customization

### Change the Port
Edit `app.py`, line at the bottom:
```python
app.run(debug=True, port=5000)  # Change 5000 to your preferred port
```

### Modify the UI Colors
Edit `templates/index.html` and change the CSS variables in the `<style>` section.

## 📊 How It Works

1. **Text Preprocessing**: Cleans and normalizes input text
2. **Vectorization**: Converts text to TF-IDF features
3. **Prediction**: Uses trained Naive Bayes model to classify
4. **Confidence**: Shows probability score for the prediction

## 📝 Notes

- The model is trained on Reddit comments
- Accuracy depends on the training data quality
- Works best with casual, conversational text
- May not detect very subtle sarcasm

## 🤝 Contributing

Feel free to improve this project! Some ideas:
- Add more ML models (SVM, Neural Networks)
- Improve text preprocessing
- Add sentiment analysis
- Export prediction history

Enjoy detecting sarcasm! 😏
