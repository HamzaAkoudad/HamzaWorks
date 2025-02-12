# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:59:02 2025

@author: Hamza
"""

from astroquery.vizier import Vizier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Configuration
INPUT_FILE = 'Catalogos_descargados/SDSS_2015.csv'  # Fixed path format
CATALOG_NAME = "V/147"  # Name of catalog
ROW_LIMIT = 9000  # Number of rows to download
FEATURES = ["umag", "gmag", "rmag", "imag", "zmag"]
TARGET = "class"

# Ensure directory exists
os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)


def download_catalog(catalog_name, row_limit):
    """
    Download a catalog if it does not exist or does not meet row requirements.

    Parameters:
        - catalog_name: Catalog to download (str)
        - row_limit: Row limit (int)

    Returns:
        - DataFrame containing the catalog
    """
    print(f"Downloading catalog {catalog_name}...")

    # Vizier configurations
    Vizier.ROW_LIMIT = row_limit
    Vizier.columns = ['*']  # Download all columns

    try:
        catalogs = Vizier.get_catalogs(catalog_name)
        if catalogs:
            df = catalogs[0].to_pandas()
            print("First rows of the catalog:")
            print(df.head())
            df.to_csv(INPUT_FILE, index=False)
            print(f"The catalog was saved as '{INPUT_FILE}'.")
            return df
        else:
            print("No data was found for the specified catalog.")
            return None
    except Exception as e:
        print(f"Error downloading the catalog: {e}")
        return None


def load_or_download_data():
    """
    Load data from file or download it if necessary.

    Returns:
        - DataFrame containing the catalog
    """
    if os.path.exists(INPUT_FILE):
        print(f"The file '{INPUT_FILE}' already exists locally.")
        df = pd.read_csv(INPUT_FILE)
        if df.shape[0] != ROW_LIMIT:
            print(f"Row count mismatch! Expected {ROW_LIMIT}, but found {df.shape[0]}.")
            df = download_catalog(CATALOG_NAME, ROW_LIMIT)
    else:
        print(f"File '{INPUT_FILE}' does not exist.")
        df = download_catalog(CATALOG_NAME, ROW_LIMIT)
    return df


def preprocess_data(df):
    """
    Preprocess the data by selecting relevant columns and removing NaNs.

    Parameters:
        - df: Raw DataFrame

    Returns:
        - X: Features DataFrame
        - y: Target Series
    """
    df = df[FEATURES + [TARGET]].dropna()
    return df[FEATURES], df[TARGET]


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print the accuracy and classification report.

    Parameters:
        - model: Trained model
        - X_test: Test features
        - y_test: Test target
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))


def plot_class_distribution(y):
    """
    Plot the distribution of classes in the target data.

    Parameters:
        - y: Target Series
    """
    y.value_counts().plot(kind='bar', color='b', alpha=0.7)
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def main():
    # Load or download data
    df = load_or_download_data()

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # Train Random Forest model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model
    evaluate_model(clf, X_test, y_test)

    # Plot class distribution
    plot_class_distribution(y_train)


if __name__ == "__main__":
    main()


# Visualization of decision tree from random forest

# tree = clf.estimators_[0]
# fig = plt.figure(figsize=(20,10))  # Set figure size to make the tree more readable
# features= ["umag", "gmag", "rmag", "imag", "zmag"]
# class_names = y_df.unique().astype(str)  # Ensure class names are strings
# plot_tree(tree, 
#           feature_names=features,  # Use the feature names from the dataset
#           class_names=class_names,  # Use class names (species names)
#           filled=True,              # Fill nodes with colors for better visualization
#           rounded=True)             # Rounded edges for nodes
# plt.title("Decision Tree from the Random Forest")
# plt.show()
# fig.savefig('rf_5trees.png')
