# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load your trained model (you must have saved it as 'model.pkl')
MODEL_PATH = 'model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/')
def home():
    return jsonify({
        "status": "Running",
        "message": "Credit Card Fraud Detection API",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Please train and save model.pkl"}), 500

    try:
        data = request.get_json()
        # Expect a list of feature values in correct order (e.g., V1, V2, ..., Amount)
        features = data.get('features')
        if not features or len(features) != 30:
            return jsonify({"error": "Expected 30 features (V1-V28, Amount, Time)"}), 400

        df = pd.DataFrame([features], columns=[f'V{i}' for i in range(1,29)] + ['Amount', 'Time'])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()

        return jsonify({
            "prediction": int(prediction),  # 0 = legit, 1 = fraud
            "fraud_probability": probability[1]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
