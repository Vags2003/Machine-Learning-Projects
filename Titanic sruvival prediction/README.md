# Titanic Survival Prediction using Logistic Regression

This project aims to predict the survival of passengers aboard the Titanic using the Logistic Regression algorithm. The model is trained on passenger data, including features such as age, sex, and ticket class, to determine whether a passenger survived or not.

## Overview
> The Titanic disaster, one of the deadliest maritime tragedies in history, has been the subject of various analyses. This project uses machine learning to predict whether a passenger survived or not based on available demographic and ticket-related features. The Logistic Regression algorithm is employed to classify the survival outcome, using historical data of Titanic passengers.

## Dataset
> The dataset used in this project is from Kaggle’s Titanic competition:
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

The dataset contains information about passengers aboard the Titanic, including demographic data and other characteristics. Key features include:

- **Pclass**: Passenger class (1, 2, or 3)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: The fare the passenger paid for the ticket
- **Embarked**: The port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- **Survived**: Whether the passenger survived (1) or not (0)

## Model
> The Logistic Regression algorithm is used in this project to predict the survival of Titanic passengers. Logistic Regression is a linear model used for binary classification problems, making it suitable for this task (survived or not). The model predicts the probability of survival based on the features provided.

> Key steps:
1. **Data preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
2. **Feature engineering**: Transforming features like `Fare` to improve the model's performance.
3. **Splitting the dataset**: Dividing the dataset into training and testing sets.
4. **Training the model**: Using Logistic Regression to train on the training data.
5. **Model evaluation**: Using accuracy as the evaluation metric to assess model performance.

## Technologies Used
> **Python**: For data analysis, preprocessing, and building the model.  
> **Logistic Regression**: For classification.  
> **Pandas**: For data manipulation and analysis.  
> **Scikit-learn**: For machine learning algorithms, model training, and evaluation.  
> **NumPy**: For numerical operations.  
> **Matplotlib** and **Seaborn**: For data visualization.

## Results
> The Logistic Regression model achieved an accuracy of **81.11%** on the test data. Below are the evaluation metrics:

- **Accuracy**: 81.11%
- **Precision**: 78.6%
- **Recall**: 83.4%
- **F1 Score**: 80.0%

> The model was able to correctly classify the survival of passengers with a high degree of accuracy, making it a viable tool for predictions in similar scenarios.
