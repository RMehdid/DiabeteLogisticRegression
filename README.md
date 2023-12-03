# Diabetes Binary Classification Project

## Overview

This project focuses on binary classification to predict the presence or absence of diabetes based on various health-related features. The Logistic Regression model is employed for its simplicity and effectiveness in binary classification tasks.

## Dataset

The dataset used for this project is sourced from [diabetes.csv](link_to_dataset). It includes information about several health-related features such as pregnancies, glucose level, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. The target variable is whether a patient has diabetes or not (Outcome).

### Dataset Source
[Dataset Source](link_to_dataset)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/diabetes-binary-classification.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load and Explore the Dataset:

   ```python
   import pandas as pd

   diabete_df = pd.read_csv("diabetes.csv")
   ```

2. Train the Logistic Regression Model:

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
   outcome = ['Outcome']

   X = diabete_df[features]
   y = diabete_df[outcome]

   X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2)

   lg_model = LogisticRegression()
   lg_model.fit(X_train, y_train)
   ```

3. Evaluate Model Performance:

   ```python
   predictions = lg_model.predict(X_test)
   accuracy = lg_model.score(X_test, y_test)
   ```

   Explore the results and model performance in the Jupyter Notebook.

## Results

The Logistic Regression model achieves an accuracy of X% on the test set. Further details, including confusion matrix, precision, recall, and F1-score metrics, are provided in the notebook.

## Future Improvements

1. Feature Engineering: Experiment with creating new features or transforming existing ones for better model performance.
2. Hyperparameter Tuning: Fine-tune the logistic regression model by optimizing hyperparameters.