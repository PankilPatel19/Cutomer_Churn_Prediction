"""
db_setup.py
-----------
Reads the Telco Customer Churn CSV and loads it into a
local SQLite database (churn_data.db).
"""

import sqlite3
import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DB_PATH    = os.path.join(BASE_DIR, "churn_data.db")

# ── Step 1: Load raw CSV ───────────────────────────────────────────────────────
def load_csv(path):
    print(f"[INFO] Reading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw dataset shape: {df.shape}")
    return df

# ── Step 2: Clean the data ─────────────────────────────────────────────────────
def clean_dataframe(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    before = len(df)
    df = df.drop_duplicates(subset=["customerID"])
    after  = len(df)
    if before != after:
        print(f"[WARN] Removed {before - after} duplicate rows.")
    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df

# ── Step 3: Write to SQLite ────────────────────────────────────────────────────
def write_to_sqlite(df, db_path, table_name="customers"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"[INFO] Written {count} records to '{table_name}' in {db_path}")
    conn.close()

# ── Step 4: Create Index ───────────────────────────────────────────────────────
def create_index(db_path, table_name="customers"):
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_customerID ON {table_name}(customerID)")
    conn.commit()
    conn.close()
    print("[INFO] Index created on customerID.")

# ── Step 5: Print Schema ───────────────────────────────────────────────────────
def print_schema(db_path, table_name="customers"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    print(f"\n[SCHEMA] Table '{table_name}' columns:")
    for col in cols:
        print(f"    {col[1]:30s}  {col[2]}")
    conn.close()

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}.\n"
            "Please place the CSV file inside the /data folder."
        )
    df = load_csv(CSV_PATH)
    df = clean_dataframe(df)
    write_to_sqlite(df, DB_PATH)
    create_index(DB_PATH)
    print_schema(DB_PATH)
    print("\n[SUCCESS] Database setup complete. File saved at:", DB_PATH)