# Titanic Survival Prediction Using Machine Learning

This project aims to predict whether a passenger on the Titanic survived or not using machine learning algorithms. The dataset used includes features such as age, sex, passenger class, and fare, which are used to predict survival.

## Table of Contents

- [Project Description](#project-description)
- [Data Description](#data-description)
- [Libraries Used](#libraries-used)
- [Model Building](#model-building)
- [Results](#results)
- [How to Run the Code](#how-to-run-the-code)

## Project Description

In this project, we predict the survival status of passengers aboard the Titanic using various machine learning algorithms. The project involves:
- Exploring the dataset with visualizations and summary statistics.
- Preprocessing the data by handling missing values and performing feature engineering.
- Training several models including **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**, and **Decision Tree** classifiers.
- Evaluating the models' performance using accuracy, cross-validation, and confusion matrix.

The objective of this project is to build a predictive model that can accurately predict whether a passenger survived the Titanic disaster based on available features.

## Data Description

The Titanic dataset consists of the following features:

- **PassengerId**: Unique ID for each passenger
- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger's name
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Fare**: Fare paid by the passenger
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- **Cabin**: Cabin number
- **Ticket**: Ticket number
- **Survived**: Survival status (0 = No, 1 = Yes)

## Libraries Used

The following Python libraries are used in the project:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computing
- **matplotlib** & **seaborn**: For data visualization
- **scikit-learn**: For machine learning algorithms and model evaluation
- **xgboost**: For the XGBoost classifier
- **lightgbm**: For the LightGBM classifier

## Model Building

The following models are built and evaluated for Titanic survival prediction:

1. **Logistic Regression**: A statistical model used for binary classification tasks.
2. **Decision Tree Classifier**: A model that splits the data into subsets based on feature values.
3. **Random Forest Classifier**: An ensemble method using multiple decision trees for better generalization.
4. **XGBoost**: A gradient boosting model known for high accuracy and efficiency.
5. **LightGBM**: Another gradient boosting model, optimized for speed and efficiency.

## Results

The models were evaluated using accuracy, cross-validation scores, and confusion matrix. The **Random Forest** model provided the best performance with an accuracy of around 81.6% and a cross-validation score of 80.7%.

### Model Performance:

- **Decision Tree Classifier**:
  - Accuracy: 77.6%
  - CV Score: 76.6%
  
- **LightGBM Classifier**:
  - Accuracy: 80.7%

- **XGBoost Classifier**:
  - Accuracy: 80.3%
  - CV Score: 81.3%
  
- **Random Forest Classifier**:
  - Accuracy: 81.6%
  - CV Score: 80.7%
  
- **Logistic Regression**:
  - Accuracy: 81.2%
  - CV Score: 78.3%

## How to Run the Code

### 1. Clone this repository:

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
