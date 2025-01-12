#!/usr/bin/env python
# coding: utf-8

# # Data loading

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score,f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')


# In[5]:


base_url = "https://raw.githubusercontent.com/filipecorreia23/Bank-Term-Deposit-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{base_url}/X_train_fe.csv")
X_test = pd.read_csv(f"{base_url}/X_test_fe.csv")
y_train = pd.read_csv(f"{base_url}/y_train.csv")
y_test = pd.read_csv(f"{base_url}/y_test.csv")

print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


# Ensuring that the appropriate columns are of type categorical:

# In[7]:


category_cols = ['job', 'marital', 'education', 'contact','age_bin', 'previous_bin', 'poutcome','contact_date',
                 'last_contact_days_bin']

for col in category_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')


# # Light GBM

# In[9]:


# initialize the model
lgb_model = LGBMClassifier()

# fit the model
lgb_model.fit(
    X_train, y_train, 
    categorical_feature=category_cols,
    eval_set=[(X_test, y_test)])

# make predictions
y_pred = lgb_model.predict(X_test)


# ### Classification report

# In[11]:


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# The model does a great job identifying non-subscribers (Class 0) with high precision and recall. However, it struggles with subscribers (Class 1), showing lower recall and F1-scores. While the overall accuracy is 91%, there's room to improve predictions for Class 1.

# ### ROC AUC score

# In[14]:


# prediction probabilities
y_test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

# ROC AUC score
test_auc = roc_auc_score(y_test, y_test_proba_lgb)
print(f"Test AUC: {test_auc:.4f}")


# Test AUC score of 0.9309 shows that our LightGBM model performs very well in distinguishing between the two classes.

# ### Confusion Matrix

# In[17]:


# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# The confusion matrix also shows that the model performs exceptionally well in predicting non-subscribers (Class 0) but struggles to accurately identify subscribers (Class 1).

# ### ROC Curve

# In[21]:


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba_lgb)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# The curve being so close to the top-left corner shows that the model has a high true positive rate (sensitivity) while keeping the false positive rate pretty low. This means it’s good at telling the two classes apart. However, there is a lot of room to improve Class 1 results.

# # Hyperparameter tuning

# In[25]:


# defining parameter grid to search
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10], 
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "n_estimators": [50, 100, 200, 250],
    "max_bin": [255, 512, 1024],
    "num_leaves": [15, 31, 63, 127, 255],
    "min_data_in_leaf": [10, 20, 30, 40, 50],
    "lambda_l1": [0.0, 0.1, 1.0, 5.0, 10.0],  # L1 regularization
    "lambda_l2": [0.0, 0.1, 1.0, 5.0, 10.0],  # L2 regularization
    "min_split_gain": [0.0, 0.01, 0.1, 0.5, 1.0],  # Min loss reduction to split
    "bagging_fraction": [0.4, 0.6, 0.8, 1.0],
    "bagging_freq": [1, 3, 5, 7],  
    "feature_fraction": [0.4, 0.6, 0.8, 1.0],
    "scale_pos_weight": [1.0, 2.0, 3.0, 4.0]  # For imbalanced dataset
}

# lets initialize the LGBM model
lgb_model_ = LGBMClassifier(objective='binary')

# performing RandomizedSearchCV with early stopping
random_search = RandomizedSearchCV(lgb_model_, param_distributions=param_grid, n_iter=50,  
                                    scoring='f1', cv=5, n_jobs=-1, random_state=42)

# fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# get the best hyperparameters
best_params_lgb = random_search.best_params_
best_model_lgb = random_search.best_estimator_

# lets make predictions
y_pred_lgb = best_model_lgb.predict(X_test)


# ### Classification report

# In[30]:


report = classification_report(y_test, y_pred_lgb)
print("Classification Report:\n", report)


# The F1-score improvements were modest but meaningful. Overall accuracy saw a slight drop, but the impact is minimal. Recall for true positives—our main goal—improved significantly, boosting our ability to identify potential subscribers. Precision, which shows how many predicted subscribers were correct, decreased. Still, our focus is on recall to capture as many actual subscribers as possible, even if it means some errors along the way!

# In[34]:


# prediction probabilities
y_test_proba_lgb = best_model_lgb.predict_proba(X_test)[:, 1]

# AUC score
test_auc = roc_auc_score(y_test, y_test_proba_lgb)
print(f"Test AUC: {test_auc:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba_lgb)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# The AUC score is basically the same.

# In[36]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_lgb)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# After hyperparameter tuning, the matrix shows an increase in correctly identified subscribers, though misclassifications remain.

# # Feature selection

# Lets investigate which features contribute the most to our LGBM model. We will select based on **permutation importance**.

# In[40]:


# permutation importance
perm_importance = permutation_importance(best_model_lgb, X_test, y_test, scoring="f1", 
                                         n_repeats=10, random_state=42)

# to convert into df
perm_feature_scores = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

# plot permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_feature_scores["Importance"], y=perm_feature_scores["Feature"])
plt.title("Permutation Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.show()


# ### We'll keep features with importance scores above 5%:

# In[42]:


# calculating permutation importance
perm_result = permutation_importance(
    best_model_lgb, 
    X_test, 
    y_test, 
    scoring='f1', 
    n_repeats=10, 
    random_state=42
)

# converting permutation importance results to a DF
perm_importance = pd.DataFrame({
    'feature': X_test.columns,
    'perm_importance': perm_result.importances_mean
})

# importance threshold
perm_threshold = 0.05
top_features = perm_importance[perm_importance['perm_importance'] > perm_threshold]['feature'].tolist()

# Filter to include only important features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]


# In[43]:


print("Selected Features:")
print(top_features)


# ### Lets retrain the model:

# In[45]:


best_model_lgb_selected = LGBMClassifier(**best_params_lgb)

# retrain the model with selected features
best_model_lgb_selected.fit(X_train_selected, y_train)

# and make predictions
y_pred_selected = best_model_lgb_selected.predict(X_test_selected)


# In[46]:


report = classification_report(y_test, y_pred_selected)
print("\nClassification Report:\n", report)


# No improvement observed!!

# ### Learning curve

# In[49]:


# making sure the metric is set to 'binary_logloss' in the parameters
best_params_lgb['metric'] = 'binary_logloss'

# datasets
dtrain = lgb.Dataset(X_train_selected, label=y_train)
dval = lgb.Dataset(X_test_selected, label=y_test)

# callback to track evaluation metrics
evals_result = {}

# training our model with callbacks
model = lgb.train(
    params=best_params_lgb,
    train_set=dtrain,
    num_boost_round=200,
    valid_sets=[dtrain, dval],
    valid_names=['Train', 'Validation'],
    callbacks=[
        lgb.record_evaluation(evals_result),    # Storing evaluation metrics
        lgb.early_stopping(stopping_rounds=10)  # Early stopping callback
    ],
)

# training and validation losses
train_losses = evals_result['Train']['binary_logloss']
validation_losses = evals_result['Validation']['binary_logloss']

# learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Log Loss', color='blue')
plt.plot(validation_losses, label='Validation Log Loss', color='orange')
plt.xlabel('Boosting Iterations')
plt.ylabel('Log Loss')
plt.title('Learning Curve with Early Stopping')
plt.legend(loc='best')
plt.grid()
plt.show()


# The learning curve shows steady improvement, with log loss dropping quickly at first and leveling off after 30 iterations. Training and validation stay closely aligned, showing no overfitting. Early stopping at 10 iterations ensured a balanced and efficient model.

# ### ROC AUC

# In[53]:


# Predicting probabilities for the Class 1 (yes)
y_test_proba_lgb = best_model_lgb_selected.predict_proba(X_test_selected)[:, 1]

# ROC AUC score
test_auc = roc_auc_score(y_test, y_test_proba_lgb)

print(f"Test AUC: {test_auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba_lgb)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Its worst!!

# ### Confusion matrix

# In[59]:


# confusion matrix
cm = confusion_matrix(y_test, y_pred_selected)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# # Summary

# After tuning, the model achieves a better balance between "Yes" and "No" classes. Accuracy is solid at 89%, reflecting a small trade-off for improved focus on the minority class. Precision for "Yes" is 52%, with some over-prediction, but recall rises to 75%, capturing more true positives. For "No," precision is 97% and recall is 91%, maintaining strong performance. The F1-scores—61% for "Yes" is still not ideal, but we are focusing on improving our recall. An AUC of 0.93 confirms the model's robust class separation. 
# 
# Unfortunately, feature selection did not improve performance. As a result, the hyperparameter-tuned model without feature selection is our final choice for comparison.
