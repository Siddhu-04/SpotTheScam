import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import io

# --- Streamlit App Configuration (Dark Theme) ---
st.set_page_config(
    page_title="Job Fraud Detector",
    page_icon=":briefcase:",
    layout="wide", # Use a wide layout
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# Custom CSS for dark mode aesthetics (adjust as needed)
st.markdown("""
<style>
body {
    color: #eee;
    background-color: #111;
}
.stSelectbox label, .stFileUploader label, .stRadio label {
    color: #eee;
}
.stButton>button {
    color: #eee;
    background-color: #333;
}
h1, h2, h3, h4, h5, h6 {
    color: #fff;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
  font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# Set seaborn style for dark background
plt.style.use('seaborn-v0_8-darkgrid') # Use a dark grid style
plt.rcParams.update({
    'figure.facecolor': '#1e1e1e', # Dark background for figures
    'axes.facecolor': '#1e1e1e',    # Dark background for axes
    'axes.edgecolor': '#eee',     # Light edges
    'axes.labelcolor': '#eee',    # Light labels
    'xtick.color': '#eee',        # Light ticks
    'ytick.color': '#eee',        # Light ticks
    'text.color': '#eee',         # Light text
    'grid.color': '#333',         # Darker grid lines
    'figure.dpi': 150             # Increase DPI for better resolution
})


# --- Load Models ---
@st.cache_resource # Cache the models to avoid reloading on every interaction
def load_models():
    try:
        loaded_rf_model = joblib.load('random_forest_model.joblib')
        loaded_knn_model = joblib.load('knn_model.joblib')
        return loaded_rf_model, loaded_knn_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'random_forest_model.joblib' and 'knn_model.joblib' are in the same directory.")
        st.stop()

loaded_rf_model, loaded_knn_model = load_models()

# --- Load Trained Columns ---
@st.cache_data # Cache the column list
def load_trained_columns():
    try:
        trained_columns = pd.read_csv('X_train_columns.csv')['columns'].tolist()
        return trained_columns
    except FileNotFoundError:
        st.error("X_train_columns.csv not found. Cannot align data correctly.")
        st.stop()

trained_columns_during_notebook_training = load_trained_columns()

# --- Preprocessing Function ---
# This function needs to exactly replicate the preprocessing in your notebook
def preprocess_data(df, trained_columns):
    # Handle missing values (replicate notebook's fillna)
    df['location'] = df['location'].fillna(value='Other')
    df['department'] = df['department'].fillna(value='Other')
    df['company_profile'] = df['company_profile'].fillna(value='')
    df['description'] = df['description'].fillna(value='')
    df['requirements'] = df['requirements'].fillna(value='')
    df['benefits'] = df['benefits'].fillna(value='')
    df['employment_type'] = df['employment_type'].fillna(value='Other')
    df['required_experience'] = df['required_experience'].fillna(value='Not Applicable')
    df['required_education'] = df['required_education'].fillna(value='Unspecified')
    df['industry'] = df['industry'].fillna(value='Other')
    df['function'] = df['function'].fillna(value='Other')

    # Drop columns (replicate notebook's drop)
    cols_to_drop = ['job_id', 'salary_range', 'company_profile', 'description', 'benefits', 'requirements']
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    # Create dummy variables (replicate notebook's get_dummies)
    df = pd.get_dummies(df, prefix_sep = '_', drop_first=True)

    # Align columns with the training data columns
    df, _ = df.align(pd.DataFrame(columns=trained_columns), join='right', axis=1, fill_value=0)

    return df

# --- Streamlit App ---
st.title("🕵️ Job Posting Fraud Detection Dashboard")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Model Selection in Sidebar
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Model for Prediction:",
    ('Random Forest', 'K Nearest Neighbors')
)

# Display Data Head (optional, you can put this elsewhere)
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        df_original = df_input.copy() # Store original data

        # --- Data Processing and Prediction ---
        with st.spinner("Processing data and making predictions..."):
            df_processed = preprocess_data(df_input.copy(), trained_columns_during_notebook_training)

            # Make predictions based on selected model
            if selected_model == 'Random Forest':
                predictions = loaded_rf_model.predict(df_processed)
                probabilities = loaded_rf_model.predict_proba(df_processed)[:, 1]
                df_original['Predicted_Fraudulent'] = predictions
                df_original['Fraud_Probability'] = probabilities
            elif selected_model == 'K Nearest Neighbors':
                predictions = loaded_knn_model.predict(df_processed)
                # KNN predict_proba can be less reliable, or not available depending on implementation.
                # We'll just use the prediction for KNN in the main table and visualizations.
                df_original['Predicted_Fraudulent'] = predictions
                df_original['Fraud_Probability'] = None # No probability for KNN in this view

        st.success("Processing and Prediction Complete!")

        # --- Dashboard Visualizations ---
        st.subheader(f"Analysis based on {selected_model} Predictions")

        tab1, tab2, tab3 = st.tabs(["Prediction Results", "Overall Visualizations", "Detailed Plots"])

        with tab1:
            st.write("### Prediction Results Table")
            st.dataframe(df_original)

            # Download link for results
            csv_export = df_original.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv_export,
                file_name='job_fraud_predictions.csv',
                mime='text/csv',
            )

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                # Pie Chart of Fake vs Real Jobs
                st.write("### Predicted Fake vs. Real Jobs")
                fake_real_counts = df_original['Predicted_Fraudulent'].value_counts().sort_index()
                if not fake_real_counts.empty:
                    fig1, ax1 = plt.subplots()
                    ax1.pie(fake_real_counts, labels=['Real (0)', 'Fake (1)'], autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c']) # Using dark theme friendly colors
                    ax1.axis('equal')
                    st.pyplot(fig1)
                else:
                    st.info("No predictions to display in pie chart.")

            with col2:
                # Histogram of Fraud Probabilities (Only for Random Forest)
                if selected_model == 'Random Forest' and 'Fraud_Probability' in df_original.columns and df_original['Fraud_Probability'].notna().any():
                    st.write("### Distribution of Fraud Probabilities (Random Forest)")
                    fig, ax = plt.subplots()
                    sns.histplot(df_original['Fraud_Probability'], bins=20, kde=True, ax=ax, color='#2ecc71') # Green color
                    ax.set_title("Histogram of Random Forest Fraud Probabilities")
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                     st.info(f"Probability distribution is primarily shown for Random Forest and requires probability scores.")


            # Top-10 Most Suspicious Listings (Based on Probability if available, otherwise just predicted fraudulent)
            st.write(f"### Top-10 Most Suspicious Listings ({selected_model})")
            if 'Fraud_Probability' in df_original.columns and selected_model == 'Random Forest' and df_original['Fraud_Probability'].notna().any():
                top_10_suspicious = df_original.sort_values(by='Fraud_Probability', ascending=False).head(10)
                st.dataframe(top_10_suspicious[['title', 'location', 'Fraud_Probability']])
            else:
                 # For KNN or if probability is not used, show top based on predicted fraudulent
                 top_10_suspicious = df_original[df_original['Predicted_Fraudulent'] == 1].head(10)
                 if not top_10_suspicious.empty:
                     st.dataframe(top_10_suspicious[['title', 'location', 'Predicted_Fraudulent']])
                 else:
                     st.info("No listings predicted as fraudulent by this model in the top 10.")


        with tab3:
            st.write("### Detailed Feature Visualizations")

            # Additional Visualizations Dropdown (Now within a tab)
            visualization_option = st.selectbox(
                "Select a Visualization:",
                ('None', 'Fraud by Required Education', 'Fraud by Function', 'Fraud by Employment Type', 'Fraud by Company Logo', 'Fraud by Questions')
            )

            if visualization_option != 'None':
                st.write(f"#### Visualization: {visualization_option}")

                # Use the original dataframe with predictions for visualizations
                df_viz = df_original.copy()

                if visualization_option == 'Fraud by Required Education':
                    plt.figure(figsize=(12, 7))
                    sns.countplot(palette='dark:#5A9_r', hue='Predicted_Fraudulent', y='required_education', data=df_viz)
                    plt.title('Predicted Fraudulent Jobs by Required Education', color='#eee')
                    plt.xlabel('Count', color='#eee')
                    plt.ylabel('Required Education', color='#eee')
                    st.pyplot(plt)

                elif visualization_option == 'Fraud by Function':
                    plt.figure(figsize=(15, 8))
                    sns.countplot(palette='dark:#5A9_r', hue='Predicted_Fraudulent', y='function', data=df_viz, order=df_viz['function'].value_counts().index[:15]) # Show top 15 functions
                    plt.title('Predicted Fraudulent Jobs by Function', color='#eee')
                    plt.xlabel('Count', color='#eee')
                    plt.ylabel('Function', color='#eee')
                    st.pyplot(plt)

                elif visualization_option == 'Fraud by Employment Type':
                    plt.figure(figsize=(10, 5))
                    sns.countplot(palette='dark:#5A9_r', hue='Predicted_Fraudulent', y='employment_type', data=df_viz)
                    plt.title('Predicted Fraudulent Jobs by Employment Type', color='#eee')
                    plt.xlabel('Count', color='#eee')
                    plt.ylabel('Employment Type', color='#eee')
                    st.pyplot(plt)

                elif visualization_option == 'Fraud by Company Logo':
                    plt.figure(figsize=(8, 5))
                    sns.countplot(palette='dark:#5A9_r', hue='Predicted_Fraudulent', x='has_company_logo', data=df_viz)
                    plt.title('Predicted Fraudulent Jobs by Company Logo', color='#eee')
                    plt.xlabel('Has Company Logo', color='#eee')
                    plt.ylabel('Count', color='#eee')
                    plt.xticks([0, 1], ['No', 'Yes'])
                    st.pyplot(plt)

                elif visualization_option == 'Fraud by Questions':
                    plt.figure(figsize=(8, 5))
                    sns.countplot(palette='dark:#5A9_r', hue='Predicted_Fraudulent', x='has_questions', data=df_viz)
                    plt.title('Predicted Fraudulent Jobs by Questions', color='#eee')
                    plt.xlabel('Has Questions', color='#eee')
                    plt.ylabel('Count', color='#eee')
                    plt.xticks([0, 1], ['No', 'Yes'])
                    st.pyplot(plt)


    except Exception as e:
        st.error(f"An error occurred during processing or prediction: {e}")
        st.info("Please check the uploaded file format and ensure the necessary model and column files are present.")

else:
    st.info("Please upload a CSV file through the sidebar to get started.")