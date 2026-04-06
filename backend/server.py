from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# Enable CORS so our frontend can make requests to this backend seamlessly
CORS(app)

# Resolve path to the model (server.py is in /backend, model is in root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')

try:
    rf_model = joblib.load(MODEL_PATH)
    print("SYS.DIAG_CKD // RF Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading RF model: {e}")
    rf_model = None

# This is the exact order of features that helper.py and the Random Forest/CART models expect
FEATURE_NAMES = [
    'sc', 'su', 'bgr', 'bp', 'sg', 'appet', 'sod', 'al', 'pc', 'ba', 
    'pot', 'wc', 'hemo', 'rbc', 'age', 'ane', 'pcc', 'bu', 'pe', 'cad', 'pcv'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not rf_model:
        return jsonify({'error': 'Server Error: RF Model not loaded.'}), 500

    try:
        # Build the input array properly mapped
        input_data = []
        for feature in FEATURE_NAMES:
            val = data.get(feature, 0)
            input_data.append(float(val))
        
        # Reshape to 2D array for scikit-learn
        X_new = np.array(input_data).reshape(1, -1)
        
        # Predict Proba returns array of probabilities e.g., [[prob_0, prob_1]]
        probability = rf_model.predict_proba(X_new)[0][1]
        risk_percentage = round(probability * 100, 1)
        
        is_ckd = bool(probability >= 0.5)
        
        return jsonify({
            'success': True,
            'prediction': is_ckd,
            'risk_percentage': risk_percentage,
            'message': 'CKD DETECTED' if is_ckd else 'NO CKD DETECTED'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
