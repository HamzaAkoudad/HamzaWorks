# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:29:48 2025

@author: Hamza
"""
from astroquery.vizier import Vizier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Configuration
INPUT_FILE = 'Catalogos_descargados/SDSS_2015.csv'
CATALOG_NAME = "V/147"
ROW_LIMIT = 10000

# Ensure directory exists
os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)

def download_catalog(cat: str, lim: int) -> pd.DataFrame:
    """Download a catalog if it does not exist or does not meet row requirements."""
    print(f"Downloading catalog {cat}...")
    Vizier.ROW_LIMIT = lim
    Vizier.columns = ['*']  # Get all columns
    
    try:
        catalogs = Vizier.get_catalogs(cat)
        if not catalogs:
            print("No data found for the specified catalog.")
            return None
        
        df = catalogs[0].to_pandas()
        df.to_csv(INPUT_FILE, index=False)
        print(f"Catalog saved as '{INPUT_FILE}'.")
        return df
    except Exception as e:
        print(f"Error downloading catalog: {e}")
        return None

def load_or_download_catalog() -> pd.DataFrame:
    """Load catalog from file or download if necessary."""
    if os.path.exists(INPUT_FILE):
        print(f"File '{INPUT_FILE}' found locally.")
        try:
            df = pd.read_csv(INPUT_FILE)
            if len(df) != ROW_LIMIT:
                print(f"Row mismatch ({len(df)} found, {ROW_LIMIT} expected). Redownloading...")
                return download_catalog(CATALOG_NAME, ROW_LIMIT)
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}. Redownloading...")
            return download_catalog(CATALOG_NAME, ROW_LIMIT)
    return download_catalog(CATALOG_NAME, ROW_LIMIT)

def preprocess_data(df: pd.DataFrame):
    """Filter and prepare dataset for training."""
    df = df[["umag", "gmag", "rmag", "imag", "zmag", "class"]].dropna()
    X, y = df.drop(columns=["class"]), df["class"]
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def train_model(X_train, y_train) -> MLPClassifier:
    """Train an MLP classifier."""
    clf = MLPClassifier(solver='adam', learning_rate_init=0.001, alpha=1e-5, hidden_layer_sizes=(128, 64), verbose=True, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def main():
    """Main execution function."""
    df = load_or_download_catalog()
    if df is None:
        print("Failed to load or download catalog.")
        return
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot Class Distribution
    y_train.value_counts().plot(kind='bar', color='b', alpha=0.7)
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
