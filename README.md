 heart-disease-classification

 # Heart Disease Classification using PyTorch

## Overview

This project builds a neural network model to predict the presence of heart disease using clinical patient data. The model is implemented using **PyTorch**, with data preprocessing and analysis performed using **Pandas**, **NumPy**, and **Scikit-learn**.

The goal of this project is to demonstrate a complete machine learning workflow, including:

* Data preprocessing and cleaning
* Exploratory visualization
* Classical machine learning concepts (Perceptron / step function)
* Neural network implementation with PyTorch
* Model training and evaluation
* Performance analysis using accuracy and confusion matrix

---

## Dataset

The dataset used in this project is the **UCI Heart Disease Dataset**, which contains medical attributes related to heart health.

Some of the key features include:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Maximum heart rate achieved (thalach)
* ST depression induced by exercise (oldpeak)

The target variable indicates whether a patient **has heart disease or not**.

---

## Project Workflow

### 1. Data Loading and Cleaning

The dataset is loaded using Pandas and cleaned by removing rows with missing values.

A binary target variable is created:

* **0 → No Heart Disease**
* **1 → Heart Disease**

---

### 2. Exploratory Data Visualization

Scatter plots are used to visualize relationships between features such as:

* Maximum heart rate (`thalach`)
* ST depression (`oldpeak`)

This helps understand how these features relate to heart disease classification.

---

### 3. Perceptron Implementation (NumPy)

A simple **step-function based perceptron** is implemented from scratch using NumPy to classify the dataset based on two features.

The perceptron is trained using:

* Random weight initialization
* Iterative weight updates
* Misclassification tracking across epochs

This step demonstrates the **fundamental idea behind neural networks**.

---

### 4. Feedforward Neural Network (Conceptual Visualization)

A small feedforward neural network is implemented using NumPy with:

* One hidden layer
* ReLU activation
* Random weights

The output is visualized to understand how neural networks create **non-linear decision boundaries**.

---

### 5. Data Preprocessing for Deep Learning

Before training the PyTorch model:

* Features are standardized using **StandardScaler**
* Data is split into **training and testing sets (80/20)**

---

### 6. PyTorch Neural Network Model

The neural network architecture used for classification:

Input Layer: **13 features**
Hidden Layer: **16 neurons (ReLU activation)**
Output Layer: **1 neuron (Sigmoid activation)**

Example architecture:

```
Input (13 features)
       ↓
Linear Layer (13 → 16)
       ↓
ReLU Activation
       ↓
Linear Layer (16 → 1)
       ↓
Sigmoid Output
```

---

### 7. Model Training

The model is trained using:

* **Loss Function:** Binary Cross Entropy (BCELoss)
* **Optimizer:** Adam
* **Epochs:** 200

Training loss is tracked over time to monitor learning progress.

---

### 8. Model Evaluation

The trained model is evaluated using:

* **Test Accuracy**
* **Confusion Matrix**

Predictions are converted to binary labels using a **0.5 probability threshold**.

Example evaluation output:

```
Test Accuracy: ~XX%
```

---

### 9. Visualization

The project includes visualizations such as:

* Training loss curve
* Decision boundary plots
* Confusion matrix

These help interpret how well the model performs.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* PyTorch


## Learning Outcomes

This project demonstrates:

* Implementation of machine learning models from scratch
* Neural network fundamentals
* Practical deep learning using PyTorch
* Model evaluation and visualization techniques

---

## Future Improvements

Possible extensions of this project include:

* Adding more advanced neural network architectures
* Hyperparameter tuning
* Cross-validation
* Deploying the model as a web application

---

 Author

Nishka Mehta

Computer Science student focusing on **AI, Data Science, and Machine Learning**.
