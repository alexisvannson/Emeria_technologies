import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import re
from sklearn.tree import plot_tree


def preprocess(custom_df):
    # fill NaN with mean; custu
    custom_df.fillna(custom_df.mean(), inplace=True)
    #custom_df = custom_df[custom_df['TTC_AMOUNT_CONFIDENCE'] > 0.7]
    #'PRIMARY_REFERENCE_MATCH' 'SECONDARY_REFERENCE_MATCH' 'SUPPLIER_PREDICTION_root_email'
    custom_df['SUPPLIER_PREDICTION_root_email'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'root_email' == x else 0)
    custom_df['SUPPLIER_PREDICTION_siret'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'siret' == x else 0)
    custom_df['SUPPLIER_PREDICTION_METHOD_siren'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'siren' == x else 0)
    custom_df['SUPPLIER_PREDICTION_email'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'email' == x else 0)
    custom_df['SUPPLIER_PREDICTION_phone'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'phone' == x else 0)
    custom_df['SUPPLIER_PREDICTION_siren'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'siren' == x else 0)
    custom_df['SUPPLIER_PREDICTION_iban'] = custom_df['SUPPLIER_PREDICTION_METHOD'].apply(lambda x: 1 if 'iban' == x else 0)
    custom_df['BUILDING_PREDICTION_geolocation'] = custom_df['BUILDING_PREDICTION_METHOD'].apply(lambda x: 1 if 'geolocation' == x else 0)
    custom_df['BUILDING_PREDICTION_name'] = custom_df['BUILDING_PREDICTION_METHOD'].apply(lambda x: 1 if 'name' == x else 0)
    custom_df['BUILDING_PREDICTION_address'] = custom_df['BUILDING_PREDICTION_METHOD'].apply(lambda x: 1 if 'address' == x else 0)
    custom_df['BUILDING_PREDICTION_METHOD_service_number']  = custom_df['BUILDING_PREDICTION_METHOD'].apply(lambda x: 1 if 'service_number' == x else 0)
    custom_df = custom_df.replace({True: 1, False: 0})
    custom_df['CONTRACT_NUMBER_MATCH'] = np.where(custom_df['PREDICTED_CONTRACT_NUMBER'] == custom_df['TARGET_CONTRACT_NUMBER'], 1, 0)
    #custom_df = custom_df[ custom_df['CREDIT_NOTE_CONFIDENCE'] == 1 ]
    #custom_df = custom_df[  (custom_df['SECONDARY_REFERENCE_MATCH'] == 1) & (custom_df['BUILDING_PREDICTION_METHOD_service_number'] == 0)  & (custom_df['SUPPLIER_PREDICTION_phone'] == 0)] # (custom_df['SECONDARY_REFERENCE_MATCH'] == 1)  &  & (custom_df['PRIMARY_REFERENCE_MATCH'] == 1) & (custom_df['BUILDING_PREDICTION_METHOD_service_number'] == 0) 
    #custom_df = custom_df[  (custom_df['CREDIT_NOTE_CONFIDENCE'] == 1)& (custom_df['DOCUMENT_SPLIT_CONFIDENCE'])]
    return custom_df[['STATUS','CREDIT_NOTE_CONFIDENCE','TTC_AMOUNT_CONFIDENCE','HT_AMOUNT_CONFIDENCE','DOCUMENT_SPLIT_CONFIDENCE','DOCUMENT_TYPE_CONFIDENCE', 'SUPPLIER_PREDICTION_iban', 'SUPPLIER_PREDICTION_siren', 'SUPPLIER_PREDICTION_root_email', 'SUPPLIER_PREDICTION_siret', 'SUPPLIER_PREDICTION_email', 'SUPPLIER_PREDICTION_phone', 'BUILDING_PREDICTION_name', 'BUILDING_PREDICTION_address', 'PRIMARY_REFERENCE_MATCH', 'SECONDARY_REFERENCE_MATCH']], custom_df['CONTRACT_NUMBER_MATCH']



#NUUUUUl service number, building name et address: buildings
# bieng geoloc: buildings

#NUUUUUl email, root_email (service number):supplier
#bieng siren siret iban 



def preprocces_creditor(df):
    info = df['CREDITOR_INFO']
    unique_creditors = pd.DataFrame({'Creditor_Clean': info.unique()})
    unique_creditors['Creditor_Clean'] = unique_creditors['Creditor_Clean'].str.strip().str.upper().str.replace(r'[^A-Z\s]', '', regex=True)
    return unique_creditors['Creditor_Clean']

def plot_feature_importances(n=10):
    feature_importances = rf.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_importances = feature_importances[sorted_indices][:n]
    sorted_feature_names = feature_names[sorted_indices][:n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
    plt.xticks(range(len(sorted_feature_importances)), sorted_feature_names, rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'Top {n} Feature Importance')
    plt.tight_layout()
    plt.show()
    return feature_importances

def plot_single_tree(decision_tree, feature_names=None, class_names=None, figsize=(20, 10)):
    """
    Plots a single decision tree.
    
    Parameters:
    - decision_tree: The decision tree classifier to be visualized.
    - feature_names: list, The names of the features.
    - class_names: list, The names of the target classes.
    - figsize: tuple, The figure size of the plot (width, height in inches).
    """
    plt.figure(figsize=figsize)
    plot_tree(decision_tree, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

def plot_roc_curve(y_test, predicted_probabilities):
    # Extract predicted probabilities of the positive class
    positive_class_probabilities = predicted_probabilities[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test, positive_class_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def calculate_partial_dependence(model, X, feature_name, grid_resolution=50):
    X_temp = X.copy()
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_resolution)
    predictions = []
    for val in feature_values:
        X_temp[feature_name] = val
        y_pred = model.predict_proba(X_temp)[:, 1]
        predictions.append(np.mean(y_pred))
    return feature_values, predictions

def plot_partial_dependence(feature_values, pdp_predictions, feature_name):
    plt.plot(feature_values, pdp_predictions)
    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence')
    plt.title('Partial Dependence Plot for ' + feature_name)
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, cmap='Blues', title='Confusion Matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        thresh = 0.5
    else:
        fmt = 'd'
        thresh = cm.max() / 2

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, cmap=cmap, fmt=fmt, cbar=False, xticklabels=classes, yticklabels=classes)

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


df = pd.read_csv('2024-04-15 11_14am.csv')


X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

class_weights = {0: 9, 1: 1}
rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, class_weight=class_weights)
rf.fit(X_train, y_train)

predictions_proba = rf.predict_proba(X_test)
predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

roc_auc = roc_auc_score(y_test, predictions)
print("ROC-AUC Score:", roc_auc)

plot_feature_importances(12)
cm = confusion_matrix(y_test, predictions)

plot_confusion_matrix(cm, rf.classes_)

# Plot the first tree of the random forest
first_tree = rf.estimators_[0]

#plot_single_tree(first_tree, feature_names=X_train.columns, class_names=['Class1', 'Class2'])

