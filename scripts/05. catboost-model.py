#!/usr/bin/env python
# coding: utf-8

# # Dependencies loading

# In[96]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy
import numba
from catboost import CatBoostClassifier, cv, Pool
import optuna
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
import shap

import warnings
warnings.filterwarnings('ignore')  # Suppress warning


# # Data loading

# In[99]:


base_url = "https://raw.githubusercontent.com/filipecorreia23/Bank-Term-Deposit-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{base_url}/X_train_fe.csv")
X_test = pd.read_csv(f"{base_url}/X_test_fe.csv")
y_train = pd.read_csv(f"{base_url}/y_train.csv")
y_test = pd.read_csv(f"{base_url}/y_test.csv")

print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


# In[100]:


X_train


# # Model

# In[102]:


category_cols = ['job', 'marital', 'education', 'contact', 'contact_date', 
                 'poutcome','age_bin','previous_bin','last_contact_days_bin']

# initialize the model
cat_model = CatBoostClassifier()

# fit the model
cat_model.fit(X_train, y_train,cat_features=category_cols, verbose=10)

# and make predictions
y_pred = cat_model.predict(X_test)


# ## Model Evaluation

# ### Accuracy:

# In[105]:


# lets start with accuracy
xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_accuracy


# The model did well with 91% accuracy, which is a great start. Next, we should look at other things like precision and recall, especially for the smaller class, to make sure it handles imbalances properly.

# ### Classification Report:

# In[108]:


# prediction probabilities
y_test_proba = cat_model.predict_proba(X_test)[:, 1]

# classification report
report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
print(report)


# The model performs strongly for non-subscribers (class 0), achieving high precision and recall. While it struggles with identifying subscribers (class 1), showing lower recall and F1-score, it still outperformed other models for the smaller class. Despite an overall accuracy of 91%, there’s room to further improve class 1 predictions for better balance.

# ### ROC AUC score:

# In[111]:


auc_score = roc_auc_score(y_test, y_test_proba)
print(f"Test AUC: {auc_score:.4f}")


# The Test AUC score of 0.9355 shows that the model performs better than the other models in distinguishing between classes. While the other models also performed very well, this one showed an even stronger ability to correctly rank positive and negative instances, reflecting very nice overall performance.

# ### Consusion Matrix:

# In[114]:


cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# The confusion matrix shows once again that the model is great at predicting non-subscribers (class 0) but struggles to identify subscribers (class 1) accurately.

# ### ROC curve:

# In[117]:


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


# The ROC curve shows the model performs well overall, with an AUC score of 0.9355, 
# outperforming its peers. However, it still struggles to fully capture subscribers 
# due to the imbalance, favoring non-subscribers. We'll see if hyperparameters like 'scale_pos_weight' and 'subsample' can help balance the data and improve performance.

# ## Hyperparameter tunning

# In[120]:


# objective function
def objective(trial):
    # defining hyperparameters to optimize
    params = {
        "iterations": trial.suggest_int("iterations", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.05, log=True),
        "depth": trial.suggest_int("depth", 1, 6),
        "subsample": trial.suggest_float("subsample", 0.1, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.9),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 500, 1500),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 4),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 50, 150),
        "random_seed": 56
    }
    
    # lets initialize the CatBoost model
    cat_model = CatBoostClassifier(**params, eval_metric="F1", custom_metric="F1")

    # and train the model with early stopping
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=category_cols, 
                  early_stopping_rounds=30, 
                  verbose=1)

    # return the best F1-score from the validation set
    return max(cat_model.evals_result_["validation"]["F1"])

# create an Optuna study for maximizing F1-score
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=56))
study.optimize(objective, n_trials=100)

# get the best hyperparameters
best_params = study.best_trial.params

# train the final model with the best parameters
cat_model = CatBoostClassifier(**best_params, cat_features=category_cols, eval_metric="F1", custom_metric="F1", verbose=100)
cat_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)


# In[121]:


print("Best Parameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")


#  ### Learning Curve:

# In[123]:


train_scores = cat_model.evals_result_["learn"]["F1"]
validation_scores = cat_model.evals_result_["validation"]["F1"]

# plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_scores, label="Training F1")
plt.plot(validation_scores, label="Validation F1")
plt.xlabel("Iterations")
plt.ylabel("F1 Score")
plt.title("Learning Curve (F1 Score)")
plt.legend()
plt.grid(True)
plt.show()


# It generalized reasonably well!!

# ## Model Evaluation

# ### Classification report:

# In[129]:


cat_preds = cat_model.predict(X_test)

report = classification_report(y_test, cat_preds, target_names=["No", "Yes"])
print(report)


# The improvements in F1-score were not outstanding but still notable. Overall accuracy dropped slightly, but the change is not significant. Recall for true positives—our primary objective—improved significantly, demonstrating better identification of potential subscribers. Precision, which measures how many predicted subscribers are actually correct, declined. However, we are prioritizing recall, which measures how many actual subscribers were successfully identified, to ensure we capture as many potential subscribers as possible. **It's okay to do some wrong calls!!**

# ### ROC curve:

# In[132]:


# prediction probabilities
cat_probs = cat_model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, cat_probs)

# ROC AUC score
auc_score = roc_auc_score(y_test, cat_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# The AUC score is basically the same.

# ### Confusion Matrix

# In[135]:


# make predictions
y_pred = cat_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# After hyperparameter tuning, the matrix shows an increase in identified subscribers, though some misclassifications remain.

# ## Feature selection:

# Understanding what drives our model is essential. We use **two powerful methods**: **Permutation Importance**, which quickly highlights key features by measuring how performance drops when we shuffle them, and **SHAP Violin Plots**, which reveal not only which features matter but also how they influence predictions—whether positively or negatively. Together, these methods provide a clear and well-rounded view of our model’s decision-making process, helping us understand both *what* drives the model and *how* it makes its predictions.

# In[139]:


# permutation importance
perm_importance = permutation_importance(cat_model, X_test, y_test, scoring="f1", 
                                         n_repeats=10, random_state=42)

# to conver into df
perm_feature_scores = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

# plot permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_feature_scores["Importance"], y=perm_feature_scores["Feature"])
plt.title("Permutation - Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[140]:


# to ensure that categorical features are encoded
X_test_encoded = X_test.copy()
for col in category_cols:
    X_test_encoded[col] = X_test_encoded[col].astype('category').cat.codes

# initialize SHAP explainer
explainer = shap.TreeExplainer(cat_model)

# calculate SHAP values
shap_values = explainer.shap_values(X_test_encoded)

# plot (violin plot)
shap.summary_plot(shap_values, X_test_encoded, plot_type="violin")


# The analysis highlights "Duration" and "Contact_date" as the most impactful features, with "Poutcome" and "Contact" following after. SHAP shows that high "Duration" strongly boosts predictions, while "Contact_date" has mixed effects. On the other hand, features like "Previous" and "Age" contribute very little. We'll focus on the top features from SHAP and permutation importance, drop the less important ones, and retrain the model to improve performance and efficiency.

# ### We'll keep features with importance scores above 5% of the maximum, focusing on the most impactful ones:

# In[143]:


perm_importance = pd.DataFrame({
    'feature': X_test.columns, 
    'perm_importance': perm_importance.importances_mean
})

# set threshold to 5% of the maximum importance
perm_threshold = 0.05 * perm_importance['perm_importance'].max()  # Keep features contributing >5%
perm_features = perm_importance[perm_importance['perm_importance'] > perm_threshold]['feature'].tolist()

# filter dataset based on selected features
X_train_filtered = X_train[perm_features]
X_test_filtered = X_test[perm_features]

# Print selected features
print("Selected Features Based on Permutation Importance (>5% threshold):")
print(perm_features)


# In[144]:


category_cols = ['contact_date', 'contact']

# initialize the tunned model
cat_model_filtered = CatBoostClassifier(**best_params, cat_features=category_cols)

# and retrain the model on the filtered dataset
cat_model_filtered.fit(X_train_filtered, y_train)

# lets make predictions
y_pred_filtered = cat_model_filtered.predict(X_test_filtered)


# ## Model Evaluation

# ### Classification report:

# In[147]:


filtered_report = classification_report(y_test, y_pred_filtered, target_names=["No", "Yes"])
print(filtered_report)


# **Slightly worst!** Did not improve at all with feature selection unfortunately.

# ### ROC curve:

# In[150]:


# get predicted probabilities for the filtered model
y_test_proba_filtered = cat_model_filtered.predict_proba(X_test_filtered)[:, 1]

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


# Got worst as well!

# ### Confusion matrix:

# In[153]:


# make predictions
y_pred_filtered = cat_model_filtered.predict(X_test_filtered)

# confusion matrix (filtered)
cm_filtered = confusion_matrix(y_test, y_pred_filtered)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Filtered Model)')
plt.show()


# # Summary of Results
# 
# After parameter tuning, our model achieves a better balance between the minority ("Yes") and majority ("No") classes. The **accuracy decreased slightly to 89%** from 91%, reflecting a small trade-off for improved handling of imbalanced classes. Precision for "Yes" is **51%**, meaning the model's predictions for subscribers include some false positives. However, recall for "Yes" improved significantly to **82%**, showing the model captures more true subscribers—critical for identifying valuable leads. For non-subscribers ("No"), the model maintains high precision (**97%**) and recall (**90%**). The AUC score of **0.94** confirms the model's strong overall ability to separate the classes.
# 
# Unfortunately, feature selection did not improve the model's performance. As a result, we consider the hyperparameter-tuned model without feature selection as our final model for comparison.
