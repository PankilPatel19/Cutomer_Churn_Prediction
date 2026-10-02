import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "churn_data.db")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "churn_model.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

def fetch_data(db_path):
    print("[INFO] Connecting to database:", db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    print("[INFO] Fetched", len(df), "rows and", df.shape[1], "columns.")
    return df

def preprocess(df):
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Domain feature engineering before label encoding
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    active_services = (df[service_cols] == "Yes").sum(axis=1)
    df["ChargePerService"] = df["MonthlyCharges"] / (active_services + 1)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != "Churn"]
    print("[INFO] Categorical columns:", categorical_cols)
    print("[INFO] Numerical columns:", numerical_cols)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders, numerical_cols, categorical_cols

def train_and_evaluate(df):
    TARGET = "Churn"
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    # Split train/test FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("[INFO] Training set:", X_train.shape[0], "| Test set:", X_test.shape[0])

    scaler = StandardScaler()
    numerical_in_X = [c for c in X.columns if X[c].dtype in ["float64", "int64", "int32"]]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_in_X] = scaler.fit_transform(X_train[numerical_in_X])
    X_test[numerical_in_X] = scaler.transform(X_test[numerical_in_X])

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    print("[INFO] Starting GridSearchCV... (this takes 1-2 minutes)")
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("[INFO] Best parameters:", grid_search.best_params_)
    print("[INFO] Best CV ROC-AUC:", round(grid_search.best_score_, 4))

    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("       MODEL EVALUATION REPORT")
    print("=" * 50)
    print("  Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("  Precision:", round(precision_score(y_test, y_pred), 4))
    print("  Recall   :", round(recall_score(y_test, y_pred), 4))
    print("  F1 Score :", round(f1_score(y_test, y_pred), 4))
    print("  ROC-AUC  :", round(roc_auc_score(y_test, y_pred_prob), 4))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("=" * 50)
    return best_model, scaler, features

def save_artifacts(model, scaler, label_encoders, feature_columns):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    print("[SUCCESS] All artifacts saved to:", ARTIFACTS_DIR)
    print("          churn_model.pkl")
    print("          scaler.pkl")
    print("          label_encoders.pkl")
    print("          feature_columns.pkl")

def main():
    df = fetch_data(DB_PATH)
    df, label_encoders, numerical_cols, categorical_cols = preprocess(df)
    model, scaler, feature_columns = train_and_evaluate(df)
    save_artifacts(model, scaler, label_encoders, feature_columns)

if __name__ == "__main__":
    main()