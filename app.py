from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
from flask_cors import CORS
import os   # ✅ ADD THIS

app = Flask(__name__)
CORS(app)

# ✅ Get current backend folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Go one folder up (project main folder)
model_path = os.path.join(BASE_DIR, "random_forest_model.joblib")
dataset_path = os.path.join(BASE_DIR, "staff_performance_dataset_500.csv")

# ✅ Load files
model = joblib.load(model_path)
df = pd.read_csv(dataset_path)

# Assume preprocessing steps (e.g., encoders) are also loaded here if needed

# Prepare data for SHAP - typically features used for training
# Exclude the target variable 'salary_credit_score' and 'staff_id'
X = df[[
    'attendance_percentage',
    'avg_working_hours',
    'avg_minutes_late',
    'logbook_submissions',
    'task_completion_rate',
    'feedback_score'
]]

# Create a TreeExplainer for the Random Forest model
explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return "Staff Performance Evaluation Backend with XAI"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming incoming JSON has keys matching feature names
    # Ensure the order of columns matches the training data features
    features = [
        'attendance_percentage',
        'avg_working_hours',
        'avg_minutes_late',
        'logbook_submissions',
        'task_completion_rate',
        'feedback_score'
    ]
    # Convert incoming data to DataFrame
    input_data = pd.DataFrame([data], columns=features)

    # Make prediction
    prediction = model.predict(input_data)

    # Calculate SHAP values
    # SHAP expects the input data in the same format as the training features
    shap_values = explainer.shap_values(input_data)

    # Return the prediction and SHAP values
    # For a single prediction, shap_values will be a list of arrays (one array per output). For regression, it's one array.
    # Convert the SHAP values (numpy array) to a list for JSON serialization
    shap_values_list = shap_values[0].tolist() # Get SHAP values for the first (and only) sample and convert to list

    response = {
        'salary_credit_score': prediction[0],
        'shap_values': shap_values_list,
        'feature_names': features # Also send feature names for frontend mapping
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True) 