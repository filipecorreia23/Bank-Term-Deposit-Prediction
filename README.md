# Bank Term Deposit Prediction

## How do you turn a "maybe" into a "yes"?

For banks, this is the key to running successful marketing campaigns. Phone calls are a powerful tool for offering financial products like term deposits, but they come with challenges. Calling uninterested clients wastes time and resources, while missing the right ones means losing opportunities. So, how can banks know who to call?

This project predicts whether a client will subscribe to a term deposit ("yes" or "no") based on data from past marketing campaigns conducted by a Portuguese bank. The dataset includes valuable information such as client demographics, previous interactions, and economic conditions. By analyzing this data, we aim to uncover the patterns behind client decisions.

## Project Goal

The goal of this project is to develop machine learning models to predict whether a client will subscribe to a term deposit. By identifying likely subscribers, banks can:

- **Save resources** by targeting the right clients.
- **Boost efficiency** in their marketing efforts.
- **Personalize campaigns** for better client experiences.

We will use the following machine learning models:
- **LightGBM**: Efficient for large datasets, handles categorical features well.
- **CatBoost**: Natively supports categorical features and reduces preprocessing efforts.
- **XGBoost**: A high-performance, scalable gradient boosting framework.
- **Gradient Boosting (Baseline)**: A standard approach for comparison.

## Dataset Overview

The dataset includes 45,211 instances and 16 features, divided into three categories:

### Client Data
- **age**: Age of the client (numeric).
- **job**: Type of job (categorical).
- **marital**: Marital status (categorical).
- **education**: Education level (categorical).
- **default**: Has credit in default? (binary).
- **balance**: Average yearly balance, in euros (numeric).
- **housing**: Has a housing loan? (binary).
- **loan**: Has a personal loan? (binary).

### Campaign Data
- **contact**: Contact communication type (categorical).
- **day**: Last contact day of the month (numeric).
- **month**: Last contact month of the year (categorical).
- **duration**: Duration of the last contact, in seconds (numeric).  
  _Note_: Duration is highly predictive but requires careful use to avoid bias.

### Previous Campaign Data
- **campaign**: Number of contacts during the campaign (numeric).
- **pdays**: Days since the client was last contacted (numeric; -1 indicates no prior contact).
- **previous**: Number of previous contacts (numeric).
- **poutcome**: Outcome of the previous campaign (categorical).

### Target Variable
- **y**: Indicates whether the client subscribed to a term deposit ("yes" or "no").

## Key Challenges

1. **Class Imbalance**: Most clients say "no," making this a challenging classification problem.
2. **Complex Feature Interactions**: Numerical and categorical features interact in non-linear ways, requiring robust algorithms.
3. **Strong Predictive Features**: Features like duration can dominate predictions and introduce bias.

## Why It Matters

- **Better Marketing**: Improve targeting accuracy and reduce wasted efforts.
- **Client Insights**: Understand factors influencing client behavior.
- **Cross-Industry Impact**: Methods and insights can apply to other fields like e-commerce and telecom.

## Let's Find the "Yes" That Matters

Join us as we explore and build machine learning models to make smarter decisions, optimize campaigns, and drive impactful results.

## Project Structure

├── README.md
├── data
│   ├── input
│   ├── models_output
│   └── output
├── models
├── notebooks
│   ├── 01. project-description-&-data-preparation-&-EDA.ipynb
│   ├── 02. feature-engineering-&-feature-selection.ipynb
│   ├── 03. gradient-boosting-model.ipynb
│   ├── 04. lightgbm-model.ipynb
│   ├── 05. catboost-model.ipynb
│   ├── 06. xgboost-model.ipynb
│   ├── 07. final-model-comparison-and-summary.ipynb
└── README.md

---

## How to Run the Project Locally

1. **Install Conda Environment Management System**  
   If you don’t already have Conda installed, you can follow the instructions here:  
   [Install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. **Create Conda Environment**  
   Use the provided `requirements.txt` file to set up your environment.  
   Run the following commands in your terminal or Anaconda Prompt:

   2.1. Add `conda-forge` channel:  
   ```bash
   conda config --append channels conda-forge


