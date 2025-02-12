# Astronomical Data Classification with Machine Learning

This repository contains two Python scripts that download, preprocess, and analyze astronomical data from the **Sloan Digital Sky Survey (SDSS)** catalog using the `astroquery` library to access data from **Vizier**. The scripts train machine learning models to classify astronomical objects based on their photometric features (e.g., magnitudes in different bands) and evaluate their performance.

## Scripts Overview

### 1. Random Forest Classifier
- **Model**: Trains a **Random Forest Classifier**.
- **Process**:
  - Downloads the SDSS catalog from Vizier if it doesn't exist locally or if the row count doesn't match the expected limit.
  - Preprocesses the data by selecting relevant features (`umag`, `gmag`, `rmag`, `imag`, `zmag`) and the target variable (`class`).
  - Splits the data into training and testing sets.
  - Trains the Random Forest model with 200 estimators.
  - Evaluates the model using accuracy and a classification report.
  - Visualizes the class distribution in the training data.

### 2. Multi-Layer Perceptron (MLP) Neural Network
- **Model**: Trains a **Multi-Layer Perceptron (MLP) Neural Network**.
- **Process**:
  - Similar to the Random Forest script, it downloads and preprocesses the data.
  - Uses an MLP classifier with a specific architecture (`hidden_layer_sizes=(128, 64)`) and training parameters (`learning_rate_init`, `alpha`, etc.).
  - Evaluates the model's performance and visualizes the class distribution in the training data.

## Key Features
- **Data Download**: Automatically downloads the SDSS catalog using `astroquery.Vizier` if not already available locally.
- **Preprocessing**: Filters relevant columns and removes rows with missing values.
- **Model Training**:
  - Random Forest: Ensemble learning with 200 decision trees.
  - MLP: Deep learning with a neural network architecture.
- **Evaluation**: Provides accuracy and a detailed classification report for each model.
- **Visualization**: Plots the class distribution in the training set to understand data balance.

## Requirements
- Python 3.x
- Libraries: `astroquery`, `pandas`, `scikit-learn`, `matplotlib`

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
