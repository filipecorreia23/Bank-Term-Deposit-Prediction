#!/usr/bin/env python
# coding: utf-8

# # Dependencies loading

# In[300]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings('ignore')


# # Data loading

# In[303]:


base_url = "https://raw.githubusercontent.com/filipecorreia23/Bank-Term-Deposit-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{base_url}/X_train_fe.csv")
X_test = pd.read_csv(f"{base_url}/X_test_fe.csv")
y_train = pd.read_csv(f"{base_url}/y_train.csv")
y_test = pd.read_csv(f"{base_url}/y_test.csv")

print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


# In[304]:


X_train


# # Model

# ## Encoding

# As already explained, XGBoost cannot handle categorical data directly. Therefore, we converted all categorical variables into numeric format using label encoding. Each category was replaced with a unique numeric value, ensuring the data is compatible with XGBoost for effective processing.

# In[308]:


# creating a dictionary to store LabelEncoder objects
label_encoders = {}

# lets slice the last 3 characters for 'contact_date'
X_train['contact_date'] = X_train['contact_date'].str[-3:]
X_test['contact_date'] = X_test['contact_date'].str[-3:]

# lets first create LabelEncoder for 'contact_date'
label_encoders['contact_date'] = LabelEncoder()

X_train['contact_date'] = label_encoders['contact_date'].fit_transform(X_train['contact_date'])
X_test['contact_date'] = label_encoders['contact_date'].transform(X_test['contact_date'])

# then, get all the categorical columns of object type
categorical_columns = X_train.select_dtypes(include=['object']).columns.difference(['contact_date'])

# encoding each categorical column
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    
    # fit and transform for X_train
    X_train[column] = label_encoders[column].fit_transform(X_train[column])
    
    # transform X_test using the same encoder
    X_test[column] = label_encoders[column].transform(X_test[column])

# to print Encoding and Decoding
for column, encoder in label_encoders.items():
    print(f"\n{column} Encoding and Decoding:")
    
    encoded_values = range(len(encoder.classes_))
    decoded_values = encoder.classes_
    
    for enc, dec in zip(encoded_values, decoded_values):
        print(f"{enc} -> {dec}")


# In[309]:


# initialize the model
xgb_model = XGBClassifier()

# fit the model
xgb_model.fit(X_train, y_train)


# In[310]:


# and make predictions
y_pred = xgb_model.predict(X_test)


# ## Model Evaluation

# ### Accuracy:

# In[313]:


# lets start with accuracy
xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_accuracy


# The model achieved 91% accuracy, showing good initial performance. Next, we should check other metrics like precision and recall to ensure it handles imbalances well.

# ### Classification Report:

# In[316]:


# prediction probabilities
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# classification report
report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
print(report)


# The model performs well for non-subscribers (class 0) with high precision and recall. However, it struggles to identify subscribers (class 1) with lower recall and f1-score. Overall accuracy is 91%, but improvements are needed for class 1 predictions.

# ### ROC AUC score:

# In[319]:


auc_score = roc_auc_score(y_test, y_test_proba)
print(f"Test AUC: {auc_score:.4f}")


# The Test AUC score of 0.9222 shows that the model performs very well in distinguishing between classes. A high AUC score reflects a strong ability to rank positive and negative instances correctly.

# ### Consusion Matrix:

# In[322]:


cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Again, the confusion matrix shows the model is great at predicting non-subscribers (class 0), but it struggles with subscribers (class 1).

# ### ROC curve:

# In[325]:


# prediction probabilities
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_proba)

# plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Again, the ROC curve shows the model does a great job overall, with an AUC score of 0.9222. This means it’s good at telling the two classes apart. But, there’s a catch—it struggles with the smaller group (subscribers). Since most people in the dataset didn’t subscribe, the model gets really good at predicting them, but it misses more subscribers. To fix this, we might need to balance the data or tweak how the model learns.

# ## Hyperparameter tunning

# In[328]:


# Define the updated parameter grid to search
param_dist = {
    'n_estimators': [50, 100, 300, 500],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],   
    'max_depth': [2, 3, 5, 7],                    
    'min_child_weight': [1, 5, 10, 15],           
    'subsample': [0.2, 0.3, 0.5, 0.7, 0.9],           
    'colsample_bytree': [0.2, 0.3, 0.5, 0.7, 0.9],    
    'gamma': [1.0, 2.0, 5.0],                 
    'scale_pos_weight': [1, 2, 3, 4],       
    'reg_alpha': [50, 100, 200], # L1 regularization
    'reg_lambda': [50, 100, 200] # L2 regularization
}

# lets initialize the XGBoost model
xgb_model = xgb.XGBClassifier()

# perform RandomizedSearchCV with early stopping
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist,
                                   n_iter=50, scoring='average_precision', cv=10, random_state=42)

# fit the model using RandomizedSearchCV
random_search.fit(X_train, y_train)

# get the best hyperparameters
best_params = random_search.best_params_
print("Best Parameters: ", best_params)

# train the final model with the best parameters
bst = xgb.XGBClassifier(**best_params)

bst.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="aucpr",             
    early_stopping_rounds=20, 
    verbose=True
)

# lets make predictions
bst_preds = bst.predict(X_test)


# In[329]:


# plot the learning curve for PR AUC
def plot_pr_auc_curve(bst):
    results = bst.evals_result()
    train_aucpr = results['validation_0']['aucpr']
    test_aucpr = results['validation_1']['aucpr']

    plt.figure(figsize=(12, 5))
    plt.plot(train_aucpr, label="Training PR AUC")
    plt.plot(test_aucpr, label="Validation PR AUC")
    plt.xlabel("Iterations")
    plt.ylabel("PR AUC")
    plt.title("Learning Curve (PR AUC with Early Stopping)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_pr_auc_curve(bst)


# ## Model Evaluation

# ### Classification report:

# In[333]:


bst_preds = bst.predict(X_test)

report = classification_report(y_test, bst_preds, target_names=["No", "Yes"])
print(report)


# The F1-score improvements, while not remarkable, are still meaningful. Accuracy saw a slight decline, but the impact is minimal. Most importantly, recall for true positives—our main focus—rose significantly, showing the model's better ability to identify potential subscribers. Precision, indicating the proportion of correctly predicted subscribers, decreased. However, our priority remains recall to ensure we capture as many potential subscribers as possible. A few false positives are acceptable in this trade-off!

# ### ROC curve:

# In[338]:


# prediction probabilities
y_test_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]

# ROC AUC score
auc_score = roc_auc_score(y_test, y_test_proba)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)

# plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# The AUC score slightly decreased, but it’s nothing to be concerned about. The trade-off is worth it as the recall for true positives—our primary objective—improved significantly.

# ### Confusion Matrix

# In[346]:


# make predictions
y_pred = random_search.best_estimator_.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# After hyperparameter tuning, the matrix shows an increase in correctly identified subscribers, though some misclassifications remain.

# ## Feature selection:

# Feature selection is all about finding the most important pieces of data to focus on. It helps simplify the model, makes it run faster, and improves it by cutting out the noise. It’s like keeping only what truly matters!

# In[352]:


# get feature importances from the hp tunned model
headers = ["name", "score"]
values = sorted(zip(X_train.columns, bst.feature_importances_), key=lambda x: x[1] * -1)
feature_importances = pd.DataFrame(values, columns=headers)

features = feature_importances.head(12)

# plot
plt.figure(figsize=(10, 6))
plt.barh(features['name'], features['score'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances (tunned XGB model)')

plt.gca().invert_yaxis()
plt.show()


# Let's filter using a 5% threshold:

# In[354]:


important_features = feature_importances[feature_importances['score'] > 0.05]['name'].tolist()
important_features


# In[355]:


# let's filter on important features
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]

# initialize the tunned model
xgb_model_filtered = xgb.XGBClassifier(**random_search.best_params_)

# and retrain the model on the filtered dataset
xgb_model_filtered.fit(X_train_filtered, y_train)

# lets make predictions
y_pred_filtered = xgb_model_filtered.predict(X_test_filtered)


# ## Model Evaluation

# ### Classification report:

# In[358]:


filtered_report = classification_report(y_test, y_pred_filtered, target_names=["No", "Yes"])
print(filtered_report)


# **Got worst!** Did not improve at all with feature selection unfortunately.

# ### ROC curve:

# In[361]:


# get predicted probabilities for the filtered model
y_test_proba_filtered = xgb_model_filtered.predict_proba(X_test_filtered)[:, 1]

# ROC AUC score
auc_score_filtered = roc_auc_score(y_test, y_test_proba_filtered)

# ROC curve
fpr_filtered, tpr_filtered, _ = roc_curve(y_test, y_test_proba_filtered)

# plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_filtered, tpr_filtered, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score_filtered:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Filtered Model)')
plt.legend(loc="lower right")
plt.show()


# Also got worst!

# ### Confusion matrix:

# In[364]:


# make predictions
y_pred_filtered = xgb_model_filtered.predict(X_test_filtered)

# confusion matrix (filtered)
cm_filtered = confusion_matrix(y_test, y_pred_filtered)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Filtered Model)')
plt.show()


# # Summary

# After fine-tuning, the XGBoost model strikes a better balance between "Yes" (subscribers) and "No" (non-subscribers). Accuracy dropped slightly to 88% from 91%, a small trade-off for improved handling of the minority class. Precision for "Yes" is 49%, meaning some false positives, but recall improved to 73%, capturing more true subscribers—key for identifying leads. For "No," the model maintains strong precision (96%) and recall (90%). The ROC AUC score of 0.91 confirms solid overall performance.
# 
# Feature selection didn’t help, so we’re using the tuned model without it as our final choice.
