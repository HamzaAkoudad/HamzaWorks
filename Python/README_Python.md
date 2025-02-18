# Using Astronomical Data with Machine Learning Classification and Markov Chain Monte Carlo Method

This repository contains Python scripts to download, preprocess, and analyze astronomical data from the **Sloan Digital Sky Survey (SDSS)** catalog using the `astroquery` library to access data from **Vizier**. The scripts train machine learning models to classify astronomical objects based on their photometric features (e.g., magnitudes in different bands) and evaluate their performance.

Additionally, the repository includes a script that uses **Markov Chain Monte Carlo (MCMC)**, specifically the `emcee` package, to estimate the **Cosmic Microwave Background (CMB) temperature** from data obtained by **COBE** and **ARCADE**. This analysis involves fitting a model to the observed data, calculating the posterior distribution of the temperature, and providing an estimate of the CMB temperature with its associated uncertainty.

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

### 3. CMB Temperature Estimation with MCMC
- **Model**: Uses **Markov Chain Monte Carlo (MCMC)** with the `emcee` package.
- **Process**:
  - Downloads CMB temperature data from **COBE** and **ARCADE**.
  - Defines a simple model assuming a constant temperature for the CMB across different frequencies.
  - Uses MCMC to sample from the posterior distribution of the temperature and estimate its value, including its uncertainty.

## Key Features
- **Data Download**: Automatically downloads the SDSS catalog using `astroquery.Vizier` if not already available locally.
- **Preprocessing**: Filters relevant columns and removes rows with missing values.
- **Model Training**:
  - Random Forest: Ensemble learning with 200 decision trees.
  - MLP: Deep learning with a neural network architecture.
  - MCMC for CMB: Uses the `emcee` package to estimate the CMB temperature.
- **Evaluation**: Provides accuracy and a detailed classification report for each model.
- **Visualization**: Plots the class distribution in the training set to understand data balance.

## Requirements
- Python 3.x
- Libraries: `astroquery`, `pandas`, `scikit-learn`, `matplotlib`, `emcee`

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
