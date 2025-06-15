import streamlit as st
import pandas as pd
import joblib
import numpy as np
import smtplib
from email.mime.text import MIMEText
import os # Import the os module to help with paths

# --- Set up paths for model and columns file ---
# Get the directory of the current script
script_dir = os.path.dirname(__file__)
model_path_rf = os.path.join(script_dir, 'random_forest_model.joblib')
model_path_knn = os.path.join(script_dir, 'knn_model.joblib')
columns_path = os.path.join(script_dir, 'X_train_columns.csv')


# --- Load your models and training columns ---
try:
    rf_model = joblib.load(model_path_rf)
    knn_model = joblib.load(model_path_knn)
    train_columns = pd.read_csv(columns_path)['columns'].tolist()
    st.success("Models and training columns loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading necessary files: {e}")
    st.error(f"Please ensure 'random_forest_model.joblib', 'knn_model.joblib', and 'X_train_columns.csv' are in the same directory as app2.py: {script_dir}")
    st.stop() # Stop the app if models can't be loaded

# --- Email Configuration (Use Streamlit Secrets for security) ---
# Instructions on how to use Streamlit Secrets: https://docs.streamlit.io/library/api-reference/utilities/st.secrets
# Create a .streamlit folder in your app's directory, then create a secrets.toml file inside it.
# Add your email credentials like this:
# [email]
# recipient_email = "your_alert_email@example.com"
# sender_email = "your_sending_email@example.com"
# smtp_server = "your_smtp_server.com"
# smtp_port = 587 # or 465 for SSL
# smtp_user = "your_smtp_username"
# smtp_password = "your_smtp_password"

try:
    recipient_email = st.secrets["email"]["recipient_email"]
    sender_email = st.secrets["email"]["sender_email"]
    smtp_server = st.secrets["email"]["smtp_server"]
    smtp_port = st.secrets["email"]["smtp_port"]
    smtp_user = st.secrets["email"]["smtp_user"]
    smtp_password = st.secrets["email"]["smtp_password"]
except KeyError as e:
    st.error(f"Error loading email secret: {e}.")
    st.error("Please ensure your .streamlit/secrets.toml file is configured correctly with the [email] section and all required keys.")
    st.stop()


# Define the fraud probability threshold for sending an alert
FRAUD_ALERT_THRESHOLD = 0.8 # Adjust this threshold (0.0 to 1.0) as needed

# --- Email Sending Function ---
def send_fraud_alert(job_data, fraud_probability):
    """
    Sends an email alert for a potentially fraudulent job listing.
    Uses Streamlit Secrets for email configuration.

    Args:
        job_data (dict): The dictionary containing the job listing information.
        fraud_probability (float): The predicted probability of the job being fraudulent.
    """
    try:
        # Create the email message
        subject = f"Fraud Alert: High-Risk Job Listing Detected (Probability: {fraud_probability:.2f}%)"
        # Construct a more detailed body using available job_data
        body = f"""
        A potentially fraudulent job listing has been detected with a predicted probability of {fraud_probability:.2f}%.

        Job Details:
        Title: {job_data.get('title', 'N/A')}
        Location: {job_data.get('location', 'N/A')}
        Employment Type: {job_data.get('employment_type', 'N/A')}
        Required Experience: {job_data.get('required_experience', 'N/A')}
        Required Education: {job_education: job_data.get('required_education', 'N/A')}
        Industry: {job_data.get('industry', 'N/A')}
        Function: {job_data.get('function', 'N/A')}
        Has Company Logo: {'Yes' if job_data.get('has_company_logo', 0) == 1 else 'No'}
        Has Questions: {'Yes' if job_data.get('has_questions', 0) == 1 else 'No'}
        # You can add other relevant fields from job_data here

        Please review this listing carefully.
        """
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        st.info(f"Fraud alert email sent to {recipient_email} for job: {job_data.get('title', 'N/A')}")

    except Exception as e:
        st.error(f"Error sending fraud alert email: {e}")
        # Log the error for debugging
        print(f"Error sending fraud alert email: {e}")


# --- Function to preprocess user input ---
def preprocess_input(user_input_dict, train_columns):
    """
    Preprocesses the user input dictionary to match the training data format.

    Args:
        user_input_dict (dict): Dictionary containing user input job details.
        train_columns (list): List of column names used during model training.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame ready for prediction.
    """
    input_df = pd.DataFrame([user_input_dict])

    # Apply the same preprocessing steps as training
    input_df['location'] = input_df['location'].fillna(value='Other')
    input_df['department'] = input_df['department'].fillna(value='Other')
    input_df['company_profile'] = input_df['company_profile'].fillna(value='')
    input_df['description'] = input_df['description'].fillna(value='')
    input_df['requirements'] = input_df['requirements'].fillna(value='')
    input_df['benefits'] = input_df['benefits'].fillna(value='')
    input_df['employment_type'] = input_df['employment_type'].fillna(value='Other')
    input_df['required_experience'] = input_df['required_experience'].fillna(value='Not Applicable')
    input_df['required_education'] = input_df['required_education'].fillna(value='Unspecified')
    input_df['industry'] = input_df['industry'].fillna(value='Other')
    input_df['function'] = input_df['function'].fillna(value='Other')

    # Drop salary_range again (it was added as NaN placeholder if not in input)
    input_df = input_df.drop('salary_range', axis=1, errors='ignore')

    # Drop text columns that were dropped during training
    input_df = input_df.drop(['company_profile','description','benefits','requirements'], axis=1, errors='ignore')

    # Create 'country' column
    def split_location(location):
        l = str(location).split(',')
        return l[0]
    input_df['country'] = input_df['location'].apply(split_location)

    # Apply one-hot encoding
    input_df = pd.get_dummies(input_df, prefix_sep='_', drop_first=True)

    # Align input columns with training columns and fill missing with 0
    # This is crucial to handle cases where the input data doesn't have all the categories
    # that were present in the training data.
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    return input_df


# --- Your Streamlit App Code ---
st.title("Job Posting Fraud Detector")

st.write("Enter the details of the job posting to predict if it's fraudulent.")

# --- User Input Section ---
title = st.text_input("Job Title")
location = st.text_input("Location (e.g., City, State, Country)")
employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other', 'Casual', 'Freelance']) # Added more types based on typical data
required_experience = st.selectbox("Required Experience", ['Not Applicable', 'Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive'])
required_education = st.selectbox("Required Education", ['Unspecified', 'High School or Equivalent', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctorate', 'Vocational', 'Certification'])
industry = st.text_input("Industry")
function = st.text_input("Job Function")
has_company_logo = st.checkbox("Does the posting have a company logo?")
has_questions = st.checkbox("Does the posting have application questions?")

# --- Model Selection (Allow user to choose between RF and KNN) ---
selected_model = st.radio("Select Model for Prediction:", ("Random Forest", "KNN"))

# --- Prediction Button ---
if st.button("Predict if Fraudulent"):
    # Store original user input for the email alert
    user_input_for_alert = {
        'title': title,
        'location': location,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function,
        'has_company_logo': 1 if has_company_logo else 0,
        'has_questions': 1 if has_questions else 0,
        # Include other potentially useful original fields for the alert
        'company_profile': '', # Add placeholders if these
        # ... (previous code for imports, loading models, secrets config, send_fraud_alert, preprocess_input, Streamlit app title/intro, user input section, model selection)

        # Include other potentially useful original fields for the alert
        'company_profile': '', # Add placeholders if these
        'description': '',
        'benefits': '',
        'requirements': '',
        'department': 'Other',
        'salary_range': np.nan
    }


    # Preprocess the user input
    processed_input_df = preprocess_input(user_input_for_alert, train_columns)

    # --- Make Prediction ---
    model_to_use = None
    if selected_model == "Random Forest":
        model_to_use = rf_model
    elif selected_model == "KNN":
        model_to_use = knn_model

    if model_to_use:
        prediction = model_to_use.predict(processed_input_df)
        # Get the probability of the positive class (fraudulent = 1)
        # KNN.predict_proba might behave differently with limited n_neighbors
        # but it generally provides probabilities.
        try:
            prediction_proba = model_to_use.predict_proba(processed_input_df)[:, 1]
            fraud_probability = float(prediction_proba[0])
        except AttributeError:
            # Some models might not have predict_proba (less common for these)
            st.warning("Could not get prediction probability for the selected model.")
            fraud_probability = -1 # Indicate probability is not available

        is_fraudulent = bool(prediction[0])

        # --- Display Prediction Results ---
        st.subheader(f"Prediction Result ({selected_model}):")
        if is_fraudulent:
            st.error(f"This job posting is predicted to be **Fraudulent**.")
        else:
            st.success(f"This job posting is predicted to be **Legitimate**.")

        if fraud_probability != -1:
             st.write(f"Fraud Probability: {fraud_probability:.2f}%")

        # --- Trigger Email Alert if High Risk and probability is available ---
        if fraud_probability != -1 and fraud_probability > FRAUD_ALERT_THRESHOLD:
            st.warning(f"This is a high-risk job listing (Probability > {FRAUD_ALERT_THRESHOLD:.2f}%). Attempting to send an alert...")
            # Pass the original user input dictionary and the calculated probability
            send_fraud_alert(user_input_for_alert, fraud_probability * 100)
        elif fraud_probability != -1:
             st.info(f"Fraud probability ({fraud_probability:.2f}%) is below the alert threshold ({FRAUD_ALERT_THRESHOLD:.2f}%). No alert sent.")
    else:
        st.error("Please select a model to use.")