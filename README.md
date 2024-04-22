### Overview

This project predicts contract numbers based on supplied and building information, utilizing a RandomForestClassifier from scikit-learn, along with data preprocessing, visualization of feature importances, ROC curves, confusion matrices, and decision trees. The goal is to accurately classify contract number matches using features extracted from the dataset.

### Environment Setup

1. **Requirements File:** Ensure you have a `requirements.txt` file in your project directory with the specified library versions. 

2. **Installation:** Install the required packages by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Data Preprocessing

The `preprocess` function prepares the dataset for model training by:

- Filling missing values with column means.
- Encoding categorical variables related to supplier prediction methods and building prediction methods into binary variables.
- Creating a binary target variable based on the match between predicted and target contract numbers.

### Model Training

A RandomForestClassifier is trained on the preprocessed dataset with custom class weights to address any imbalance in the target variable. The model's performance is evaluated using accuracy and ROC-AUC scores.

### Visualization

The project includes functions to visualize:

- Feature importances to identify the most influential features for the model's predictions.
- The ROC curve to assess the model's discriminative ability.
- A confusion matrix for detailed performance analysis.
- Decision trees to gain insights into the decision-making process of the model.

### Usage

Follow the step-by-step guide in the README to preprocess your data, train the RandomForestClassifier, and apply the visualization functions to evaluate the model's performance. Ensure your dataset is formatted correctly as indicated in the preprocessing function. 

### Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Preprocess the data
X, y = preprocess(df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, class_weight={0: 9, 1: 1})
rf.fit(X_train, y_train)

# Evaluate the model
predictions_proba = rf.predict_proba(X_test)
predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

roc_auc = roc_auc_score(y_test, predictions_proba[:, 1])
print("ROC-AUC Score:", roc_auc)

# Visualization
plot_feature_importances(12)
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cm, rf.classes_)
```

Note: Modify paths and parameters as needed for your dataset and specific requirements.
