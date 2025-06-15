from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Set up paths for models and columns file ---
# Assuming this script is in the same directory as your saved files
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_RF_PATH = os.path.join(SCRIPT_DIR, 'random_forest_model.joblib')
MODEL_KNN_PATH = os.path.join(SCRIPT_DIR, 'knn_model.joblib')
COLUMNS_PATH = os.path.join(SCRIPT_DIR, 'X_train_columns.csv')


# --- Load the trained models and the list of training columns ---
print("Loading models and training columns...")
try:
    rf_model = joblib.load(MODEL_RF_PATH)
    knn_model = joblib.load(MODEL_KNN_PATH)
    train_columns = pd.read_csv(COLUMNS_PATH)['columns'].tolist()
    print("Models and training columns loaded successfully.")
    models_loaded = True
except FileNotFoundError as e:
    print(f"Error loading necessary files: {e}")
    print(f"Please ensure 'random_forest_model.joblib', 'knn_model.joblib', and 'X_train_columns.csv' exist in the same directory as api_app.py: {SCRIPT_DIR}")
    models_loaded = False # Flag to indicate if models were loaded


app = Flask(__name__)

# --- Function to preprocess input data ---
def preprocess_input(job_data_dict, train_columns):
    """
    Preprocesses a dictionary of job data to match the training data format.

    Args:
        job_data_dict (dict): Dictionary containing job posting details.
        train_columns (list): List of column names used during model training.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame ready for prediction.
        str: An error message string if preprocessing fails, otherwise None.
    """
    try:
        # Convert the job data to a pandas DataFrame
        input_df = pd.DataFrame([job_data_dict])

        # Apply the same preprocessing steps as training (fill missing, create country, etc.)
        # Ensure these steps exactly match your training preprocessing
        input_df['location'] = input_df['location'].fillna(value='Other')
        input_df['department'] = input_df['department'].fillna(value='Other') # Assuming a default if not provided
        input_df['company_profile'] = input_df['company_profile'].fillna(value='') # Fill dropped text columns with empty string
        input_df['description'] = input_df['description'].fillna(value='')
        input_df['requirements'] = input_df['requirements'].fillna(value='')
        input_df['benefits'] = input_df['benefits'].fillna(value='')
        input_df['employment_type'] = input_df['employment_type'].fillna(value='Other')
        input_df['required_experience'] = input_df['required_experience'].fillna(value='Not Applicable')
        input_df['required_education'] = input_df['required_education'].fillna(value='Unspecified')
        input_df['industry'] = input_df['industry'].fillna(value='Other')
        input_df['function'] = input_df['function'].fillna(value='Other')
        # Add salary_range placeholder as it was dropped
        input_df['salary_range'] = np.nan

        # Drop salary_range again (it was added as NaN placeholder)
        input_df = input_df.drop('salary_range', axis=1, errors='ignore')

        # Drop text columns that were dropped during training
        input_df = input_df.drop(['company_profile','description','benefits','requirements'], axis=1, errors='ignore')


        # Create 'country' column (if needed for dummy variables)
        def split_location(location):
            l = str(location).split(',') # Handle potential non-string input
            return l[0]

        input_df['country'] = input_df['location'].apply(split_location)


        # Apply one-hot encoding (pd.get_dummies)
        # Crucially, align with the training columns to ensure consistent features
        input_df = pd.get_dummies(input_df, prefix_sep='_', drop_first=True)

        # Align input columns with training columns and fill missing with 0
        # This is essential to handle cases where the input data doesn't have all the categories
        # that were present in the training data.
        input_df = input_df.reindex(columns=train_columns, fill_value=0)

        return input_df, None # Return the processed DataFrame and no error

    except Exception as e:
        # Catch any error during preprocessing and return an error message
        return None, f"Preprocessing failed: {e}"


# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    # Check if models were loaded successfully
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Cannot make predictions.'}), 500

    # Get the job data from the request (assuming JSON format)
    job_data = request.get_json(force=True)

    # Validate input format
    if not isinstance(job_data, dict):
         return jsonify({'error': 'Invalid data format. Expected a single JSON object.'}), 400

    # --- Preprocess the input data ---
    processed_input_df, preprocessing_error = preprocess_input(job_data, train_columns)

    if preprocessing_error:
         print(f"Preprocessing error: {preprocessing_error}")
         return jsonify({'error': 'Failed to preprocess input data.', 'details': preprocessing_error}), 400

    # --- Get model choice (optional - can default or require in input) ---
    # You could pass the model choice in the JSON request body, e.g., job_data['model']
    # For simplicity here, let's default to Random Forest or get it from query param
    model_choice = request.args.get('model', 'Random Forest').lower() # Get model from query parameter ?model=knn or ?model=randomforest

    model_to_use = None
    if 'random forest' in model_choice:
        model_to_use = rf_model
        model_name = "Random Forest"
    elif 'knn' in model_choice:
        model_to_use = knn_model
        model_name = "KNN"
    else:
         return jsonify({'error': f'Invalid model choice: {model_choice}. Choose "Random Forest" or "KNN".'}), 400


    # --- Make Prediction ---
    try:
        prediction = model_to_use.predict(processed_input_df)
        # Get the probability of the positive class (fraudulent = 1)
        try:
             prediction_proba = model_to_use.predict_proba(processed_input_df)[:, 1]
             fraud_probability = float(prediction_proba[0])
        except AttributeError:
             # KNN predict_proba might not always be available or reliable depending on n_neighbors
             print(f"Warning: Could not get prediction probability for {model_name}.")
             fraud_probability = -1.0 # Use -1.0 to indicate probability is not available


        is_fraudulent = bool(prediction[0])

        # --- Format the response ---
        result = {
            'model_used': model_name,
            'prediction_class': int(prediction[0]), # 0 or 1
            'is_fraudulent': is_fraudulent,
        }
        if fraud_probability != -1.0:
             result['fraud_probability'] = fraud_probability

        return jsonify(result), 200 # Return 200 OK


    except Exception as e:
        # Catch any error during prediction
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.', 'details': str(e)}), 500

# --- Running the Flask App ---
# To run this locally for testing:
# Save the code above as a Python file (e.g., api_app.py)
# Make sure your model files and X_train_columns.csv are in the same directory.
# Open your terminal, navigate to that directory.
# Run: python api_app.py

# By default, Flask runs on http://127.0.0.1:5000/
# You can send POST requests to http://127.0.0.1:5000/predict
# with JSON data representing a job posting.

# Example of how to send a POST request using Python 'requests' library:
import requests
job_data = {
    "title": "Senior Data Scientist",
    "location": "London, UK",
    "employment_type": "Full-time",
    "required_experience": "Mid-Senior level",
    "required_education": "Master's Degree",
    "industry": "Technology",
    "function": "Data Science",
    "has_company_logo": 1,
    "has_questions": 1
    # Add other relevant fields from your original dataset structure, even if empty strings
    # e.g., "department": "", "company_profile": "", "description": "", etc.
}
response = requests.post("http://127.0.0.1:5000/predict?model=randomforest", json=job_data)
print(response.json())


if __name__ == '__main__':
    # Run the Flask development server
    # debug=True allows for automatic reloading and debugging
    app.run(debug=True)