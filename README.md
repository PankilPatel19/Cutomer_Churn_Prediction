# 📡 Customer Churn Prediction System

An end-to-end Machine Learning web application designed to predict telecom customer churn and provide data-driven retention recommendations in real time.

Built with **Python**, **Scikit-Learn**, **Random Forest Classifier**, and **Streamlit**.

---

## 🚀 Live Demo
Deployable with 1-click on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 📊 Model Highlights & Performance

* **Algorithm**: Random Forest Classifier with `class_weight='balanced'` and 5-fold Stratified GridSearchCV.
* **ROC-AUC Score**: **0.8362** (Cross-Validated ROC-AUC: **0.8430**)
* **Churn Recall**: **69.5%** (Identifies ~70% of potential churners)
* **Precision**: **54.3%**
* **Overall Accuracy**: **76.4%**

### Key Features Used:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Data**: Tenure, Contract Type, Payment Method, Paperless Billing, Monthly Charges, Total Charges
- **Subscribed Services**: Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Engineered Features**: `AvgMonthlyCharge`, `ChargePerService`, `TenureGroup`

---

## 🛠️ Project Structure

```
churn-prediction-system/
├── app.py                     # Streamlit frontend & inference pipeline
├── churn_data.db              # Local SQLite database (7,043 customers)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded environments and caches
├── .streamlit/
│   └── config.toml            # Streamlit theme and UI configurations
├── artifacts/                 # Pre-trained models & preprocessing pipelines
│   ├── churn_model.pkl        # Tuned Random Forest model
│   ├── scaler.pkl             # StandardScaler fitted on training split
│   ├── label_encoders.pkl     # Categorical encoders
│   └── feature_columns.pkl    # Serialized feature order (22 features)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory Data Analysis & visual analytics
└── src/
    ├── db_setup.py            # SQLite data pipeline & data cleaning
    └── train_model.py         # Training, tuning & artifact export
```

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd churn-prediction-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ⚙️ Retraining the Model

To rebuild the database and retrain the machine learning model from scratch:

```bash
# 1. Populate SQLite database from raw CSV
python src/db_setup.py

# 2. Run preprocessing, cross-validation tuning, and export artifacts
python src/train_model.py
```

---

## 📄 License
This project is licensed under the MIT License.
