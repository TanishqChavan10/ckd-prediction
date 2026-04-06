import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Columns from frontend/server.py
FEATURE_NAMES = ['sc', 'su', 'bgr', 'bp', 'sg', 'appet', 'sod', 'al', 'pc', 'ba', 
                 'pe', 'bu', 'ane', 'pot', 'pcc', 'hemo', 'cad', 'rbc', 'age', 'wc', 'pcv']

print("Loading raw dataset...")
df = pd.read_csv('datasetVariations/CKDF_noQmarks_unindexed.csv')

# Feature Mapping logic for our specific frontend setup
# The backend expects numeric 0.0 or 1.0 for these:
# In the raw CSV, categorical columns have text ('normal', 'abnormal', 'present', 'notpresent', 'good', 'poor', 'yes', 'no')
# We need to map them to 1 and 0 based on what is generally "normal" or the dataset mode, matching our frontend.
# Let's see:
df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
df['ba'] = df['ba'].map({'notpresent': 1, 'present': 0})
df['pe'] = df['pe'].map({'no': 1, 'yes': 0})
df['ane'] = df['ane'].map({'no': 1, 'yes': 0})
df['pcc'] = df['pcc'].map({'notpresent': 1, 'present': 0})
df['cad'] = df['cad'].map({'no': 1, 'yes': 0})
df['rbc'] = df['rbc'].map({'normal': 1, 'abnormal': 0})
# Map target class
df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

y = df['class']
X = df[FEATURE_NAMES]

# Build a clean scikit-learn pipeline!
# Decision Trees handle unscaled raw data perfectly.
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Automatically fill NaNs
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training production Random Forest pipeline...")
pipeline.fit(X, y)

print(f"Accuracy on training set: {pipeline.score(X, y):.4f}")

joblib.dump(pipeline, 'models/rf_model.pkl')
print("Successfully exported raw-data compatible rf_model.pkl!")
