# Network Intrusion Detection Using Machine Learning

## Overview
With the rapid growth of digital networks, detecting malicious activity in network traffic has become increasingly critical. Traditional rule-based intrusion detection systems often fail to adapt to new or evolving attack patterns.

This project implements a **machine learning–based network intrusion detection system** that classifies network traffic as either **normal** or **malicious** using structured traffic data. The focus is on building a clear, reproducible pipeline and comparing baseline and ensemble models.


## Problem Statement
Network intrusion detection involves identifying unauthorized or malicious activities within network traffic.  
Manual analysis and static rule-based systems do not scale well with high-volume or complex traffic patterns.

The challenge is to design a data-driven approach that can:
- Automatically detect intrusions
- Generalize across different attack behaviors
- Provide measurable and interpretable performance

## Objective
The primary objective of this project is to:
- Build machine learning models to detect network intrusions
- Compare linear and ensemble-based approaches
- Evaluate model performance using standard classification metrics and ROC-AUC

## Dataset
- **Source:** KDD intrusion detection dataset  
- **Data Type:** Tabular network traffic records  
- **Target Variable:** Binary classification  
  - `0` → Normal traffic  
  - `1` → Attack traffic  

The dataset includes a mix of numerical and categorical features representing network behavior.

## Methodology

### 1. Data Preprocessing
- Removed constant (non-informative) features
- Standardized label naming
- Converted multi-class labels into a binary target
- Applied one-hot encoding to categorical features
- Handled missing and infinite values

### 2. Feature Engineering
- Separated features and target variables
- Scaled numerical features using `StandardScaler`
- Ensured stratified train-test splitting to preserve class distribution

### 3. Model Development
Two models were trained and evaluated:
- **Logistic Regression** (baseline, interpretable model)
- **Random Forest Classifier** (ensemble model for capturing non-linear patterns)

### 4. Model Evaluation
Models were evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- ROC curve visualization

## Results & Insights
- Both models successfully differentiated between normal and malicious traffic.
- Logistic Regression provided a strong and interpretable baseline.
- Random Forest achieved superior performance by capturing complex feature interactions.
- ROC-AUC scores confirmed the effectiveness of both approaches, with Random Forest performing better overall.

## Tools & Technologies
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - Matplotlib  
  - Joblib
    
## Project Structure
