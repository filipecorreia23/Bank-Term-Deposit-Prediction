#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')


# # Data loading

# In[72]:


base_url = "https://raw.githubusercontent.com/filipecorreia23/Bank-Term-Deposit-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{base_url}/X_train_fe.csv")
X_test = pd.read_csv(f"{base_url}/X_test_fe.csv")
y_train = pd.read_csv(f"{base_url}/y_train.csv")
y_test = pd.read_csv(f"{base_url}/y_test.csv")

print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


# In[73]:


X_train


# # Baseline Model

# ## Encoding

# In[76]:


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


# In[77]:


# initialize the model
baseline_model = GradientBoostingClassifier()

# fit the model
baseline_model.fit(X_train, y_train)


# In[78]:


# and make predictions
y_pred = baseline_model.predict(X_test)


# ## Model Evaluation

# ### Classification Report:

# In[81]:


y_proba = baseline_model.predict_proba(X_test)[:, 1]

# classification Report
report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
print(report)


# ### Confusion Matrix

# In[83]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Baseline Model)')
plt.show()


# ### ROC Curve

# In[85]:


auc_score = roc_auc_score(y_test, y_test_proba)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Baseline Model)')
plt.legend(loc="lower right")
plt.show()


# # Baseline Model Summary

# The baseline model shows solid overall performance for an initial implementation. The accuracy stands at a respectable 90%, and the ROC AUC score of 0.9197 highlights its capability to distinguish between positive and negative cases effectively.
# 
# However, the recall for class "Yes" (true positives) is relatively low at 40%, meaning the model misses many opportunities to identify actual term deposit subscriptions. Despite this, the model's precision for class "Yes" (63%) ensures that the predictions for term deposits are relatively reliable.
# 
# For a baseline model, these results are promising!!
