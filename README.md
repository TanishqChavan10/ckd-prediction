# SYS.DIAG_CKD

A machine learning web application for predicting the presence of Chronic Kidney Disease (CKD) in patients. The project utilizes a Random Forest algorithm and provides a web-based diagnostic interface to evaluate clinical data.

## Architecture

- `backend/`: A Flask-based REST API that utilizes a pre-trained scikit-learn Random Forest pipeline (`rf_model.pkl`) to calculate CKD risk probability.
- `frontend/`: A pure HTML, CSS, and vanilla JavaScript interface for data entry and prediction visualization.

## Installation and Setup

### 1. Backend Setup
Initialize the Python environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Running the Server
Keep the virtual environment active and start the Flask API server:

```bash
python backend/server.py
```
The server will start locally on `http://127.0.0.1:5000`.

### 3. Running the Frontend
There is no local web server required for the client application.
1. Navigate to the `frontend/` directory.
2. Open `index.html` directly in any modern web browser.

## Application Features
- Real-time, percentage-based risk assessments for CKD using a custom prediction pipeline.
- Built-in sample data injection ("Load Healthy" and "Load At-Risk") for rapid system testing.
- Included `deploy_model.py` script for re-compiling and exporting the underlying model pipeline.
