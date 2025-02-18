# **Using Computational Tools for Scientific Analysis (Machine Learning Classification, Markov Chain Monte Carlo Method)**

This repository showcases the use of **Python and MATLAB** for scientific computing, including **Machine Learning**, **Markov Chain Monte Carlo (MCMC)**, and **Orbital Simulations**.

## **Repository Structure**

```
/
|-- python/        # Python scripts for Machine Learning models for astronomical object classification and modeling
|   |-- Random_Forest_Query_SDSS.py  # Random Forest classifier
|   |
|   |-- neural_network_Query_SDSS.py # Multi-Layer Perceptron classifier
|   |
|   |-- BlackBody_Radiation_with_MCMC_Hamza.py  # Uses emcee to estimate CMB temperature
|
|-- matlab/       # MATLAB scripts for orbital simulations
|   |-- orbital_simulation.m  # Jupiter-star 1-star 2 system simulation
|
|-- data/         # Folder for dataset storage
|    |-- ARCADE_data.txt  # ARCADE data
|    |-- COBE_data.txt   # COBE observational data
|
|-- README.md # Description of the different scripts
```

## **1. Astronomical Object Classification with Machine Learning (Python)**

These scripts utilize data from the **Sloan Digital Sky Survey (SDSS)** catalog to train classification models for identifying different types of astronomical objects.

### **1.1. Random Forest Classifier**

- **Model:** Trains a **Random Forest Classifier** with 200 estimators.
- **Process:**
  - Downloads the SDSS catalog from Vizier if not already available locally.
  - Preprocesses the data, selecting relevant magnitudes (`umag`, `gmag`, `rmag`, `imag`, `zmag`) and the target variable (`class`).
  - Splits the data into training and testing sets.
  - Evaluates the model using **accuracy** and a **classification report**.
  - Generates class distribution plots for the training dataset.

### **1.2. Multi-Layer Perceptron (MLP) Neural Network**

- **Model:** Trains an **MLP neural network**.
- **Process:**
  - Downloads and preprocesses the SDSS data.
  - Configures and trains an **MLP with architecture `(128, 64)`**.
  - Evaluates model accuracy and generates a classification report.

---

## **2. CMB Temperature Estimation with MCMC (Python)**

This script uses **Markov Chain Monte Carlo (MCMC)** with `emcee` to estimate the **Cosmic Microwave Background (CMB) temperature** using data from **COBE** and **ARCADE**.

### **Process:**

- Downloads CMB temperature data from **COBE** and **ARCADE**.
- Defines a model based on Planck's law for black-body radiation.
- Uses **MCMC** to estimate the CMB temperature and calculate its uncertainty.
- Generates plots comparing observational data with the theoretical fit.

---

## **3. Orbital Simulation of Jupiter, Saturn, and the Sun (MATLAB)**

This **MATLAB** script simulates the **orbital dynamics of Jupiter, Saturn, and the Sun** using the **4th-order Runge-Kutta method** (`odeRK4`).

### **Key Features**

- **Realistic gravitational model**: Considers the gravitational interaction between the celestial bodies.
- **High-precision numerical integration**: Uses **Runge-Kutta 4** with **1,000,000 steps** to simulate 12 years.
- **Dynamic visualization**:
  - **Red**: Sun.
  - **Black**: Jupiter.
  - **Blue**: other star.
  - Trajectories update in real-time.

### **Equations and Parameters**

- **Gravitational constant (G):** 5168.6 (units in thousand km, hours, and Earth masses).
- **Masses**:
  - **Jupiter:** 318 Earth masses.
  - **other star:** \(10^6\) Earth masses.
  - **Sun:** \(10^6\) Earth masses.

### **Usage Example**

Run the script in **MATLAB**, ensuring that the function `termino_dcha_tres` is correctly implemented.

---

## **Requirements**

- **Python 3.x**
- Required libraries: `astroquery`, `pandas`, `scikit-learn`, `matplotlib`, `emcee`
- **MATLAB** for running the orbital simulation.

---

## **Usage**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Run Python scripts**:
   ```bash
   cd python
   python Random_Forest_Query_SDSS.py
   ```
   ```bash
   python neural_network_Query_SDSS.py
   ```
   ```bash
   python BlackBody_Radiation_with_MCMC_Hamza.py
   ```
3. **Run MATLAB script**:
   Open `matlab/orbital_simulation.m` in MATLAB and execute it.

