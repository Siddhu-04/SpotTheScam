Here's your README professionally formatted with better structure, consistent markdown syntax, and clear instructions:

---

# 🕵️‍♂️ Spot The Scam

## 🔍 Job Posting Fraud Detection

---

## 📌 Project Overview

This project develops a machine learning solution to detect whether a job posting is **fraudulent or legitimate**. By analyzing structured and unstructured data from job listings, the system flags potential scams to protect users and platforms.

Key components include:

* Data exploration and preprocessing
* Binary classification using ML models
* A user-facing dashboard (Streamlit)
* Optional real-time API and email alerting system

---

## 🚀 Features & Technologies

### ✅ **Core Features**

* **Fraud Detection:** Binary classifier to detect fake job listings
* **Data Analysis:** In-depth exploratory data analysis (EDA)
* **Model Training:** Uses **Random Forest** and **K-Nearest Neighbors (KNN)**
* **Preprocessing:**

  * Handling missing values
  * Feature engineering (e.g., extract country from location)
  * One-hot encoding of categorical features
* **Imbalanced Data Handling:** Over-sampling / Under-sampling using `imblearn`
* **Evaluation Metrics:**

  * Accuracy, Precision, Recall, F1-score, ROC AUC
  * Confusion matrix for visualization
* **Model Persistence:** Models saved using `joblib` for reuse
* **Streamlit Dashboard:** Interactive interface for real-time predictions
* **Email Alerts:** Optional alerts for high-risk predictions
* **REST API (Flask):** JSON-based prediction endpoint
* **Model Retraining Script:** Update the model as new data becomes available
* **SHAP Interpretability (Optional):** Feature importance insights using SHAP values

---

### ⚙️ **Technologies Used**

* **Python**
* **Pandas**, **NumPy** – Data handling
* **Scikit-learn**, **imblearn** – Machine learning & imbalance techniques
* **Matplotlib**, **Seaborn** – Data visualization
* **Streamlit** – Dashboard UI
* **Flask** – REST API
* **Joblib** – Model serialization
* **smtplib / email** – Send alerts via email
* **SHAP** – Model interpretability (optional)

---

## 🛠️ Setup Instructions

### 🔧 Prerequisites

* Python 3.6+
* `pip` installed

---

### 📁 1. Clone the Repository

```bash
git clone https://github.com/your-username/Job-Posting-Fraud-Detection.git
cd Job-Posting-Fraud-Detection
```

---

### 📦 2. Install Required Packages

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit flask joblib
```

---

### 🧠 3. Train and Save Models

Make sure your models are trained and saved before running the app:

```bash
python retrain_model.py
```

This script should save:

* `random_forest_model.joblib`
* `knn_model.joblib`
* `vectorizer.joblib`
* `X_train_columns.csv`

These files are required for both the dashboard and the API.

---

### 📬 4. Email Alerts Setup (Optional)

To enable email notifications:

1. Create a folder `.streamlit` in the project root.
2. Inside it, create a file `secrets.toml`.
3. Add the following content (replace placeholders):

```toml
[email]
recipient_email = "receiver@example.com"
sender_email = "your_email@example.com"
smtp_server = "smtp.example.com"
smtp_port = 587
smtp_user = "your_email@example.com"
smtp_password = "your_email_password"
```

> ⚠️ **Security Tip:** Never upload this file to GitHub. Use app-specific passwords or secure vaults in production.

---

### 📊 5. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

* Visit: [http://localhost:8501](http://localhost:8501)
* Features include:

  * Upload CSV file
  * Get predictions with fraud probability
  * View tables, pie charts, and histograms

---

### 🌐 6. Run the Flask API (Optional)

```bash
python api_app.py
```

* Access at: [http://127.0.0.1:5000](http://127.0.0.1:5000)
* Endpoint: `POST /predict`
* Send JSON like:

```json
{
  "title": "Software Engineer",
  "description": "Work remotely with Python...",
  "location": "USA",
  ...
}
```

---

## 📂 Directory Structure (Sample)

```
.
├── app.py
├── api_app.py
├── retrain_model.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
├── models/
│   ├── random_forest_model.joblib
│   ├── knn_model.joblib
│   └── vectorizer.joblib
└── README.md
```

---

## 📈 Future Improvements

* Integrate SHAP visualizations directly into the UI
* Enable multi-model comparison from the dashboard
* Deploy to cloud (e.g., Heroku, Streamlit Cloud, or AWS)

BEST MODEL WAS KNN A LITTLE BIT MORE IMPROVED THAN RANDOM FOREST...SVM WITH rbf kernel WAS TOO GOOD WITH F1 SCORE OF 76 SOMETHING BUT IT WAS TAKING TOO MUCH TIME...SO WE CHOOSE A BALANCED ONE i.e KNN WITH 73.8 SOMETHING F1 SCORE...
