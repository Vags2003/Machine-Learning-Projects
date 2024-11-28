# Parkinson's Disease Detection using XGBoost
This project aims to predict Parkinson's Disease using the XGBoost algorithm, based on biomedical voice measurements. The model is deployed on a web application using Flask, enabling users to input voice-related features and receive a prediction about the likelihood of Parkinson's Disease.


## Overview
> Parkinson’s Disease is a neurodegenerative disorder that affects motor control. Early detection is crucial for managing the disease. This project uses machine learning, specifically the XGBoost algorithm, to build a model that can predict whether a person has Parkinson's based on various voice measurements.

## Dataset
> The dataset used in this project is from the UCI Machine Learning Repository:
Parkinson’s Disease Data Set
It consists of 195 records from 31 people, 23 of whom have Parkinson's Disease. The features are various biomedical voice measurements.

> Key features:
- MDVP
  (Hz) - Average vocal fundamental frequency
- MDVP
  (Hz) - Maximum vocal fundamental frequency
- MDVP
  (Hz) - Minimum vocal fundamental frequency
- Jitter(%) - Measure of frequency variation
- Shimmer - Measure of amplitude variation
- HNR - Harmonics-to-noise ratio
- RPDE, D2 - Nonlinear dynamical complexity measures
- PPE - A vocal fold measure

## Model
> The XGBoost algorithm is used in this project for its efficiency and performance with structured/tabular data. XGBoost stands for Extreme Gradient Boosting and is an ensemble method based on decision trees.

> Key steps:
1. Data preprocessing (handling missing data, feature scaling)
2. Splitting the dataset into training and testing sets
3. Training the XGBoost model
4. Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1 score.

## Technologies Used
> Python: For data analysis and building the model.
> XGBoost: For implementing the machine learning model.
> Flask: For deploying the web application.
> Pandas: For data manipulation.
> Scikit-learn: For preprocessing and evaluating the model.
> Numpy: For numerical operations.

## Results
> The XGBoost model achieved an accuracy of ~92% on the test data. Below are the evaluation metrics:
Accuracy: 92%
Precision: 91%
Recall: 93%
F1 Score: 92%


