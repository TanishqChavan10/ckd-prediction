from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# Enable CORS so our frontend can make requests to this backend seamlessly
CORS(app)

# Resolve path to the model (server.py is in /backend, we will move models here)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')

try:
    rf_model = joblib.load(MODEL_PATH)
    print("SYS.DIAG_CKD // RF Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading RF model: {e}")
    rf_model = None

try:
    NB_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'naive_baysien_model.pkl')
    nb_model = joblib.load(NB_MODEL_PATH)
    print("SYS.DIAG_CKD // NB Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading NB model: {e}")
    nb_model = None

# This is the exact order of features that the models expect based on feature_names_in_
FEATURE_NAMES = [
    "sc", "su", "bgr", "bp", "sg", "appet", "sod", "al", "pc", "ba", 
    "pe", "bu", "ane", "pot", "pcc", "hemo", "cad", "rbc", "age", "wc", "pcv"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not rf_model or not nb_model:
        return jsonify({'error': 'Server Error: Models not fully loaded.'}), 500

    try:
        # Build the input array properly mapped
        input_data = []
        for feature in FEATURE_NAMES:
            val = data.get(feature, 0)
            input_data.append(float(val))
        
        # Reshape to 2D array for scikit-learn
        X_new = np.array(input_data).reshape(1, -1)
        
        # RF Prediction
        rf_prob = rf_model.predict_proba(X_new)[0][1]
        rf_risk = round(rf_prob * 100, 1)
        rf_is_ckd = bool(rf_prob >= 0.5)

        # NB Prediction
        nb_prob = nb_model.predict_proba(X_new)[0][1]
        nb_risk = round(nb_prob * 100, 1)
        nb_is_ckd = bool(nb_prob >= 0.5)
        
        return jsonify({
            'success': True,
            'rf_prediction': rf_is_ckd,
            'rf_risk_percentage': rf_risk,
            'nb_prediction': nb_is_ckd,
            'nb_risk_percentage': nb_risk
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)