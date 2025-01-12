# Bank Term Deposit Prediction

## How do you turn a "maybe" into a "yes"?

For banks, this is the key to running successful marketing campaigns. Phone calls are a powerful tool for offering financial products like term deposits, but they come with challenges. Calling uninterested clients wastes time and resources, while missing the right ones means losing opportunities. So, how can banks know who to call?

This project predicts whether a client will subscribe to a term deposit ("yes" or "no") based on data from past marketing campaigns conducted by a Portuguese bank. The dataset includes valuable information such as client demographics, previous interactions, and economic conditions. By analyzing this data, we aim to uncover the patterns behind client decisions.

## Project Goal

The goal of this project is to develop machine learning models to predict whether a client will subscribe to a term deposit. By identifying likely subscribers, banks can:

- **Save resources** by targeting the right clients.
- **Increased efficiency** in their marketing efforts.
- **Personalized campaigns** for better client experiences.

We will use the following machine learning models:
- **LightGBM**: efficient for large datasets, handles categorical features well.
- **CatBoost**: natively supports categorical features and reduces preprocessing efforts.
- **XGBoost**: a high-performance, scalable gradient boosting framework.
- **Gradient Boosting (Baseline)**: a standard approach for comparison.

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

## Main Challenges

1. **Class Imbalance**: most clients say "no," making this a challenging classification problem.
2. **Feature Interactions**: numerical and categorical features interact in non-linear ways, requiring robust algorithms.
3. **Strong Predictive Features**: features like duration can dominate predictions and introduce bias.

## Why It Matters

- **Better Marketing**: improves targeting accuracy and reduces wasted efforts.
- **Customer Insights**: understand factors influencing client behavior.
- **Multi-Industry Impact**: methods and insights can apply to other fields like e-commerce and telecom.

## Let's Find the "Yes" That Matters

Join us as we explore and build machine learning models to make smarter decisions, optimize campaigns, and drive impactful results.

## Project Structure

```bash
├── README.md
├── input
├── output
├── notebooks
│   ├── 01. project-description-&-data-preparation-&-EDA.ipynb
│   ├── 02. feature-engineering.ipynb
│   ├── 03. baseline-GBM.ipynb
│   ├── 04. xgboost-model.ipynb
│   ├── 05. catboost-model.ipynb
│   ├── 06. lightgbm-model.ipynb
│   ├── 07. final-model-comparison-and-summary.ipynb
├── scripts
│   ├── 01. project-description-&-data-preparation-&-EDA.py
│   ├── 02. feature-engineering.py
│   ├── 03. baseline-GBM.py
│   ├── 04. xgboost-model.py
│   ├── 05. catboost-model.py
│   ├── 06. lightgbm-model.py
│   ├── 07. final-model-comparison-and-summary.py
├── requirements.txt
└── requirements_pypi.txt
```
---

## How to run the project locally

1. Install conda environment management system: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create conda environment using provided requirements.txt file (paste the following commands in Anaconda Prompt or Terminal):

   2.0 Navigate in Anaconda Prompt (Windows) or Terminal (macOS/Linux) into the project directory, for instance:  

   **On Windows:**  
   ```bash
   cd C:\Users\YourUsername\Documents\MyProject
   ```
   **On macOS:**
   ```bash
   cd /Users/YourUsername/Documents/MyProject
   ```
   **On Linux:**
   ```bash
   cd /home/YourUsername/MyProject
   ```
   Then run the following commands:
   
   2.1. `conda config --append channels conda-forge`

   2.2. `conda create --name xxxx_env --file requirements.txt`

   2.3. `conda activate xxxx_env`

   2.4. `pip install -r requirements_pypi.txt`

  Run the project (using your xxxx_env) in your favorite IDE which supports notebooks for instance in Jupyter Notebook, for instance run in Anaconda Prompt:

  3.1 `conda activate xxxx_env`

  3.2 `jupyter notebook`

  ## How to run the project remotely

  1. **Clone the Repository**
     Copy the URL of this GitHub project.

  2. **Open in a Web-Based IDE**
     Replace github.com in the URL with github.dev to open the project in GitHub's web-based IDE.

  ## **Notes**

  - Replace `xxxx_env` with the name of your environment (e.g., `catboost_env` or `my_project_env`).
  - Ensure Conda is correctly installed and accessible in your system’s PATH.
  - The `requirements.txt` file should include Conda-specific dependencies, while `requirements_pypi.txt` should handle pip-only dependencies.
