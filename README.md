# Project README

## Overview

This project focuses on developing a predictive model to analyze and predict contract numbers based on a dataset. It employs a RandomForestClassifier from the scikit-learn library, along with comprehensive data preprocessing, visualization of feature importances, ROC curves, confusion matrices, and decision trees. The primary goal is to achieve accurate predictions on whether the contract numbers match, utilizing various features derived from supplier and building information.

## Prerequisites

Before running this project, ensure you have the following packages installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

These can be installed via pip using the command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

The `preprocess` function prepares the dataset for model training by:

- Filling missing values with column means.
- Encoding categorical variables related to supplier prediction methods and building prediction methods into binary variables.
- Deriving a binary target variable based on the match between predicted and target contract numbers.

Additionally, the `preprocces_creditor` function cleans creditor information for further analysis.

## Model Training

A RandomForestClassifier is trained on the preprocessed dataset with custom class weights to address any imbalance in the target variable. The model's performance is evaluated using accuracy and ROC-AUC scores.

## Visualization

The project includes functions to visualize:

- Feature importances to understand which features contribute most to the model's predictions.
- The ROC curve to evaluate the model's ability to distinguish between the classes.
- A confusion matrix to see the model's performance in detail.
- Decision trees within the random forest for insight into how decisions are made.

## Usage

1. Load your dataset as a pandas DataFrame.
2. Preprocess the dataset using the `preprocess` function.
3. Train the RandomForestClassifier with the preprocessed data.
4. Evaluate the model using the provided visualization functions.

Ensure you have a CSV file with the required format and columns similar to those mentioned in the preprocessing function. 

## Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Preprocess the data
X, y = preprocess(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, class_weight={0: 9, 1: 1})
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions_proba = rf.predict_proba(X_test)
predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

roc_auc = roc_auc_score(y_test, predictions_proba[:, 1])
print("ROC-AUC Score:", roc_auc)

# Visualizations
plot_feature_importances(12)
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cm, rf.classes_)
```

Note: Modify the paths and parameters according to your dataset and requirements.
