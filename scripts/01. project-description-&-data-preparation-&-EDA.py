#!/usr/bin/env python
# coding: utf-8

# # Problem Description

# ![Bank Marketing Campaign](https://github.com/filipecorreia23/Bank-Term-Deposit-Prediction/blob/main/input/bankmarketingcampaign.jpeg?raw=true)

# **How do you turn a "maybe" into a "yes"?**
# 
# For banks, this is the key to running successful marketing campaigns. Phone calls are a powerful tool for offering financial products like term deposits, but they come with challenges. Contacting uninterested clients wastes time, resources, and trust. Missing the right clients means losing valuable opportunities. So, how can banks know who to call?
# 
# This project focuses on **predicting whether a client will subscribe to a term deposit ("yes" or "no")**, based on past marketing campaigns conducted by a Portuguese bank. The dataset includes rich information about clients, such as their demographics, previous interactions, and economic conditions. By analyzing this data, we can uncover the patterns behind client decisions.
# 
# However, the problem isn’t as simple as it seems. The dataset is **imbalanced**, meaning most clients say "no," and client profiles are highly **diverse**, making accurate predictions tricky. For example, a younger client might respond differently to a call than a retiree, and these differences can complicate modeling. Overcoming these hurdles requires careful feature engineering and robust machine learning techniques.
# 
# The rewards, however, are worth it. Solving this problem can help banks run **smarter, more personalized campaigns**, saving money, improving efficiency, and delivering a better experience for clients. Beyond banking, these insights can inspire other industries—like e-commerce or healthcare—to adopt data-driven approaches for better decision-making. 
# 
# **The question remains:** *Can we uncover the "yes" that makes the difference?*
# 

# # Dataset Description

# The dataset provides detailed information about clients and their interactions with a Portuguese banking institution’s marketing campaigns. The goal is to predict whether a client will subscribe to a term deposit (**"yes" or "no"**). This data includes a variety of features, ranging from demographic details to historical campaign outcomes, as well as economic indicators.
# 
# ---
# 
# ### **Columns Description**
# 
# #### **Baseline Data**
# 1. **`age`** – Age of the client (numeric).  
# 2. **`job`** – Type of job (categorical):  
#    - Values: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services".  
# 3. **`marital`** – Marital status (categorical):  
#    - Values: "married", "divorced" (includes widowed), "single".  
# 4. **`education`** – Education level (categorical):  
#    - Values: "unknown", "secondary", "primary", "tertiary".  
# 5. **`default`** – Has credit in default? (binary):  
#    - Values: "yes", "no".  
# 6. **`balance`** – Average yearly balance, in euros (numeric).  
# 7. **`housing`** – Has a housing loan? (binary):  
#    - Values: "yes", "no".  
# 8. **`loan`** – Has a personal loan? (binary):  
#    - Values: "yes", "no".  
# 
# #### **Campaign Data**
# 9. **`contact`** – Contact communication type (categorical):  
#    - Values: "unknown", "telephone", "cellular".  
# 10. **`day`** – Last contact day of the month (numeric).  
# 11. **`month`** – Last contact month of the year (categorical):  
#    - Values: "jan", "feb", "mar", ..., "nov", "dec".  
# 12. **`duration`** – Last contact duration, in seconds (numeric).  
#    - **Important Note:** Duration is a highly predictive feature for the target variable but may introduce bias if used without caution.
# 
# #### **Previous Campaign Data**
# 13. **`campaign`** – Number of contacts performed during this campaign (numeric, includes last contact).  
# 14. **`pdays`** – Number of days since the client was last contacted from a previous campaign (numeric; `-1` indicates no prior contact).  
# 15. **`previous`** – Number of contacts performed before this campaign for this client (numeric).  
# 16. **`poutcome`** – Outcome of the previous marketing campaign (categorical):  
#    - Values: "unknown", "other", "failure", "success".  
# 
# #### **Target Variable**
# 17. **`y`** – Did the client subscribe to a term deposit? (binary):  
#    - Values: "yes", "no".  
# 
# ---
# 
# ### **Dataset Characteristics**
# - **Type**: Multivariate  
# - **Subject Area**: Business, Marketing  
# - **Associated Task**: Classification  
# - **Feature Types**: Categorical, Numeric  
# - **Instances**: 45,211  
# - **Features**: 16  
# 
# ---
# 
# ### **Challenges**
# 1. **Class Imbalance**:  
#    - Most clients do not subscribe to term deposits (`"no"`), making this a highly imbalanced dataset. Handling this imbalance is key to building accurate and fair models.
# 2. **Feature Interactions**:  
#    - Numerical features (e.g., `balance`, `duration`) and categorical features (e.g., `job`, `contact`) interact in complex ways, requiring thoughtful feature engineering.  
# 3. **Bias in Predictive Features**:  
#    - Features like `duration` are strong predictors but can introduce bias if interpreted improperly. Models need to consider their practical relevance.
# 
# ---
# 
# ### **Potential Applications**
# 1. **Smarter Marketing Strategies**:  
#    - Improve campaign targeting by identifying clients most likely to subscribe, saving resources and time.
# 2. **Customer Insights**:  
#    - Analyze demographic and behavioral factors influencing client decisions.
# 3. **Multi-Industry Impact**:  
#    - Insights from this dataset can also apply to industries like e-commerce and telecom, where customer retention and product adoption are key.

# # Dependencies loading

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Utilities
import os
import warnings
warnings.filterwarnings('ignore')


# # Project setup

# In[100]:


bank_full_url = 'https://raw.githubusercontent.com/filipecorreia23/Bank-Term-Deposit-Prediction/main/input/bank-full.csv'


# # Data preparation

# ## Data loading

# In[104]:


data = pd.read_csv(bank_full_url, sep=';')
df = data.copy()


# In[105]:


df.sample(10)


# # Dataset adjustment

# ## Removing Redundant Variables
# 
# The `day` and `month` columns are combined into a single variable, `contact_date`, to simplify the dataset and capture temporal information more effectively. This helps reduce redundancy without losing any meaningful data.
# 

# In[108]:


df.columns


# In[109]:


# Combine 'day' and 'month' into a string
df['contact_date'] = df['day'].astype(str) + '-' + df['month']

# Drop the original 'day' and 'month' columns
df = df.drop(columns=['day', 'month'])

df.sample(10)


# ## Handling 'Unknown' values
# 
# In dataset adjustment, it's important to find any missing values, including placeholders like `"unknown"`. This helps decide whether to fill them, remove them, or leave them as they are before moving on to further analysis.
# 

# In[111]:


unknown_counts = df.apply(lambda x: (x == 'unknown').sum())
total_counts = df.shape[0]
unknown_percentage = (unknown_counts / total_counts) * 100

print("Count of 'unknown' values per column:")
print(unknown_counts)


# In[115]:


print("Percentage of 'unknown' values per column:")
print(unknown_percentage)


# In[118]:


df.shape


# In[121]:


print("Value counts for 'contact':")
df.contact.value_counts()


# In[123]:


print("Value counts for 'poutcome':")
df.poutcome.value_counts()


# **Removing 'unknown' values**: We will drop rows where important columns like `education` and `job` have the value `'unknown'` to keep the data clean and reliable.
# 

# In[126]:


#drop the unknown values in education and job
df = df[df['education'] != 'unknown']
df = df[df['job'] != 'unknown']


# **Handling 'unknown' values in `poutcome`**: The majority of values in the `poutcome` column are labeled as `'unknown'`, outnumbering all other categories. To simplify and retain this data, we are replacing `'unknown'` with `'other'`.

# In[129]:


# Replace the 'unknown' values in the 'poutcome' column with 'other'
df['poutcome'] = df['poutcome'].replace('unknown', 'other')

# Check the updated value counts in 'poutcome'
print(df['poutcome'].value_counts())


# In[131]:


df.head(10)


# In[133]:


df.shape


# **Keeping 'unknown' values in `contact`:** For now, we will keep the `'unknown'` values in the `contact` column as a separate category.

# ## Converting Target Variable (`y`)
# 
# We’re transforming the `y` column (our target variable) into numbers: `1` for "yes" and `0` for "no". 
# 
# This makes it easier to work with machine learning algorithms, simplifies analysis, and ensures smooth handling in tasks like visualization, data splitting, and modeling. Doing this upfront keeps the workflow clean and efficient!

# In[137]:


# Convert the target variable 'y' to binary 0 and 1
df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

df['y'].value_counts()


# ## Handling Binary Variables (`yes`/`no`)
# 
# We are transforming binary features (`yes`/`no`) into numeric values (`1` for `yes` and `0` for `no`) to ensure compatibility across different machine learning models. Here's how the different boosting algorithms handle categorical variables:
# 
# - **Gradient Boosting:** Does not handle categorical variables, requiring all features to be numeric.
# - **XGBoost:** Requires all features to be numeric, making transformation mandatory.
# - **LightGBM:** Supports categorical variables natively, but transforming binary features is still useful for simplicity and consistency.
# - **CatBoost:** Handles categorical variables natively, but binary features are simple to convert and are often treated as numeric.
# 
# By converting these binary variables now, we prepare the dataset for compatibility with all algorithms, ensuring clarity and flexibility for future analysis and modeling steps.

# In[140]:


binary_columns = ['default', 'housing', 'loan']

for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df.head(10)


# For the other categorical features, we chose to retain them as they are for now, planning to handle them natively in models that support it and numerically for XGBoost and the baseline model.

# ## Sorting values
# We sort the dataset by age and balance to make the data easier to understand and ensure it's organized for analysis.

# In[144]:


df.sort_values(by=['age', 'balance'], inplace=True)
df.head(10)


# ### Reviewing the Data
# 
# A final look at the dataset ensures it's clean and ready for splitting and further analysis.

# In[147]:


df.info()


# In[151]:


df.isnull().sum()


# In[153]:


df.shape


# # Dataset Splitting
# 
# We split the dataset into:
# - **Training (& Validation) dataset**: 80% of the data.
# - **Test dataset**: 20% of the data for final evaluation.
# 
# Test dataset will be used only for the final predictions! We assume that during the entire study they do not have access to it and do not study its statistical properties.

# In[156]:


X = df.drop('y', axis=1)  # Features
y = df['y']               # Target


# **Description:** This separates the dataset into the input features X and the target variable y. The target variable y will be used for predictions, while X includes all other columns.

# In[159]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# **Description:** The `train_test_split` function splits the dataset into training and test sets:
# 
# - **test_size=0.2**: 20% of the data is reserved for testing.
# - **random_state=42**: Ensures reproducibility of the split.
# - **stratify=y**: Maintains the proportion of target classes in both training and testing datasets, ensuring a balanced representation.

# In[162]:


print(X_train.shape, X_test.shape)


# In[164]:


print(y_train.shape, y_test.shape)


# # EDA (Exploratory Data Analysis)

# ## Initial descriptive analyses of the data

# ### Viewing the first few rows of the training dataset:

# In[169]:


X_train.head(10)


# In[171]:


y_train.head(10)


# ### Overview of data types and non-null counts in the training dataset:

# In[174]:


X_train.info()


# In[176]:


y_train.info()


# ### Summary statistics for the training dataset:

# In[179]:


X_train.describe().T


# In[181]:


y_train.describe().T


# ### Checking for missing values in the training features:

# In[184]:


X_train.isnull().sum()


# In[186]:


y_train.isnull().sum()


# ## Target variable analysis

# ### Unique Values

# In[190]:


y_train.unique()


# ### Value Counts

# In[193]:


y_train.value_counts()


# ### Basic Statistics

# In[196]:


print(f"Mean: {y_train.mean()}")
print(f"Median: {y_train.median()}")
print(f"Mode: {y_train.mode()}")
print(f"Standard Deviation: {y_train.std()}")
print(f"Variance: {y_train.var()}")


# ### Class Distribution

# In[199]:


print("Class Distribution of Target Variable y:")
print(y_train.value_counts(normalize=True) * 100)


# ### Class Distribution: Pie Chart

# In[202]:


y_counts = y_train.value_counts()

plt.figure(figsize=(8, 6))
plt.pie(y_counts, labels=y_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel", 2), startangle=90)
plt.title('Distribution of Target Variable y')
plt.show()


# ### Class Counts: Bar Plot

# In[205]:


plt.figure(figsize=(8, 6))
sns.countplot(x=y_train, palette="pastel", edgecolor='black')
plt.title("Bar Plot of Target Variable y")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()


# ### Imbalance Ratio

# In[208]:


class_0, class_1 = y_train.value_counts()

print(f"Imbalance Ratio: {round(class_0 / class_1, 2)}")


# ## Features variables analysis

# ### Age

# In[212]:


X_train['age'].unique()


# In[214]:


X_train['age'].value_counts()


# In[216]:


X_train['age'].value_counts(normalize=True)


# In[218]:


# plotting age
plt.figure(figsize=(10,4))
sns.distplot(x=X_train['age'])
plt.show()


# In[219]:


# defining age groups
age_bins = [18, 30, 50, 70, df['age'].max()]
age_labels = ['Young (18-30)', 'Middle-aged (31-50)', 'Senior (51-70)', 'Elderly (71+)']

# counting occurrences of each group
age_group_counts = pd.cut(X_train['age'], bins=age_bins, labels=age_labels).value_counts()

# plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(age_labels)), startangle=90)

plt.title('Age Group Distribution')
plt.show()


# ### Job

# In[223]:


X_train['job'].unique()


# In[225]:


X_train['job'].value_counts()


# In[227]:


X_train['job'].value_counts(normalize=True)


# In[229]:


# percentage distribution for each job
job_counts = X_train['job'].value_counts(normalize=True) * 100

# plotting the horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=job_counts.values, y=job_counts.index, palette="pastel")

# tittle & lables
plt.title('Job Distribution', fontsize=14)
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Job Category', fontsize=12)

plt.show()


# ### Marital

# In[232]:


X_train['marital'].unique()


# In[234]:


X_train['marital'].value_counts()


# In[236]:


X_train['marital'].value_counts(normalize=True)


# In[238]:


# counting occurrences of each group
marital_counts = X_train['marital'].value_counts()

# plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(marital_counts, labels=marital_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(marital_counts)), startangle=90)

plt.title('Marital Status Distribution')
plt.show()


# ### Education

# In[241]:


X_train['education'].unique()


# In[243]:


X_train['education'].value_counts()


# In[245]:


X_train['education'].value_counts(normalize=True)


# In[247]:


# counting occurrences of each group
education_counts = X_train['education'].value_counts()

# plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(education_counts)), startangle=90)

plt.title('Education Level Distribution')
plt.show()


# ### Contact

# In[250]:


X_train['contact'].unique()


# In[252]:


X_train['contact'].value_counts()


# In[254]:


X_train['contact'].value_counts(normalize=True)


# In[256]:


# counting occurrences of each group
contact_counts = X_train['contact'].value_counts()

# plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(contact_counts, labels=contact_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(contact_counts)), startangle=90)

plt.title('Contact Type Distribution')
plt.show()


# ### Poutcome

# In[259]:


X_train['poutcome'].unique()


# In[261]:


X_train['poutcome'].value_counts()


# In[263]:


X_train['poutcome'].value_counts(normalize=True)


# In[265]:


# counting occurrences of each group
contact_counts = X_train['poutcome'].value_counts()

# plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(contact_counts, labels=contact_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(contact_counts)), startangle=90)

plt.title('Poutcome Distribution')
plt.show()


# ### Default

# In[268]:


X_train['default'].unique()


# In[270]:


X_train['default'].value_counts()


# In[272]:


X_train['default'].value_counts(normalize=True)


# In[274]:


# plotting an histogram for the 'default' variable
plt.figure(figsize=(8, 6))
plt.hist(X_train['default'], bins=2, color='#A3C1E1', edgecolor='black', alpha=0.8)

# tittle & lables
plt.xticks([0, 1], labels=['No (0)', 'Yes (1)'])
plt.title("Histogram of Default Variable")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Frequency")

plt.show()


# ### Housing

# In[277]:


X_train['housing'].unique()


# In[279]:


X_train['housing'].value_counts()


# In[281]:


X_train['housing'].value_counts(normalize=True)


# In[283]:


# plotting an histogram for the 'housing' variable
plt.figure(figsize=(8, 6))
plt.hist(X_train['housing'], bins=2, color='#A3C1E1', edgecolor='black', alpha=0.8)

# tittle & lables
plt.xticks([0, 1], labels=['No (0)', 'Yes (1)'])
plt.title("Histogram of Housing Variable")
plt.xlabel("Housing (0 = No, 1 = Yes)")
plt.ylabel("Frequency")

plt.show()


# ### Loan

# In[286]:


X_train['loan'].unique()


# In[288]:


X_train['loan'].value_counts()


# In[290]:


X_train['loan'].value_counts(normalize=True)


# In[292]:


# plotting an histogram for the 'loan' variable
plt.figure(figsize=(8, 6))
plt.hist(X_train['loan'], bins=2, color='#A3C1E1', edgecolor='black', alpha=0.8)

# tittle & lables
plt.xticks([0, 1], labels=['No (0)', 'Yes (1)'])
plt.title("Histogram of Loan Variable")
plt.xlabel("Loan (0 = No, 1 = Yes)")
plt.ylabel("Frequency")

plt.show()


# ### Duration

# In[295]:


X_train['duration'].unique()


# In[297]:


X_train['duration'].value_counts()


# In[299]:


X_train['duration'].value_counts(normalize=True)


# In[301]:


# plotting an histogram for the 'duration' variable
plt.figure(figsize=(10, 6))
sns.histplot(X_train['duration'], bins=30, kde=False, color='#A3C1E1', edgecolor='black')

# tittle & lables
plt.title("Distribution of Call Duration")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")

plt.xlim(0, 2000)
plt.show()


# ### Previous

# In[304]:


X_train['pdays'].describe()


# In[306]:


# plotting an histogram for p-days
plt.figure(figsize=(10, 6))
sns.histplot(X_train['pdays'], bins=30, kde=False, color='#A3C1E1', edgecolor='black')

# tittle & lables
plt.title("Distribution of P-Days")
plt.xlabel("P-Days")
plt.ylabel("Frequency")

plt.xlim(-2, X_train['pdays'].max() + 10)
plt.show()


# ## Statistical Feature Analysis

# ### Chi-Square Test

# In[310]:


# list categorical variables
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Chi-Square test
for column in categorical_columns:
    crosstab = pd.crosstab(X_train[column], y_train)
    chi2, p, dof, expected = chi2_contingency(crosstab)
    print(f"Chi-Square Test for {column}: p-value = {p}")


# All categorical variables have very small p-values, meaning they have a strong statistical relationship with the target variable (`y`). None of these variables should be removed.

# ## Point-Biserial Correlation

# In[314]:


# list of numerical variables
numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns

# point-biserial correlation
for column in numerical_columns:
    correlation, p_value = pointbiserialr(X_train[column], y_train)
    print(f"Correlation between {column} and y: {correlation}, p-value: {p_value}")


# Most numerical variables show weak correlations with the target (`y`), but their p-values are significant. However:
# - **Default** and **Campaign** have very weak correlations, suggesting they may not contribute much to predictions and could be removed.
# - Other numerical variables, like **Duration** and **Pdays**, show stronger relationships and should be kept.

# ## Combined Views

# ### Pairplot (for numerical feature relationships)

# These pairplots visually show how all numerical features in the dataset relate to each other, while also highlighting the target variable (`y`). The colors represent the target classes (`0` and `1`), helping us spot patterns, relationships, and how well the features separate the two classes.

# In[320]:


# combine X_train and y_train into a single DataFrame for plotting
data_train = X_train.copy()
data_train['y'] = y_train

# select only numeric columns for pairplot
data_numeric = data_train.select_dtypes(include=["int", "float"])

# create pairplot with 'y' as the hue
sns.pairplot(data=data_numeric, kind='reg', hue='y')
plt.show()


# - **Class Separation:** Features like `duration` show clear differences between target classes (`0` and `1`), making them important for predictions.
# - **Correlations:** Some features, like `balance` and `duration`, show relationships that could be useful for modeling.
# - **Imbalance:** The orange (`1`) class is much smaller than the blue (`0`) class, confirming the dataset imbalance.
# - **Outliers:** Features like `previous` and `pdays` have extreme values that might affect the model.
# - **Feature Insights:** Features like `housing` and `loan` are grouped, while others like `age` are spread out.

# ### Histograms (for numerical features)

# These histograms show the distribution of numeric features, highlighting their range, patterns, and potential outliers. The smooth curves help visualize data density.

# In[324]:


# select only numeric features
numeric_features = X_train.select_dtypes(include=["int", "float"]).columns

# create a grid of histograms
fig, axs = plt.subplots(3, 3, figsize=(18, 15)) 
axs = axs.flatten()

# plotting each feature in a separate subplot
for i, feature in enumerate(numeric_features):
    sns.histplot(X_train[feature], kde=True, color='cornflowerblue', ax=axs[i])
    axs[i].set_title(f'Distribution of {feature}', fontsize=14)
    axs[i].set_xlabel(feature, fontsize=12)
    axs[i].set_ylabel('Frequency', fontsize=12)

# to remove empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# - **Age**: Right-skewed, with most individuals concentrated between 30 and 50 years.
# - **Default, Housing, Loan**: Binary variables with clear splits between 0 and 1.
# - **Balance, Duration, Campaign, Pdays, Previous**: Highly skewed distributions, with most values concentrated near zero, indicating outliers or rare cases for higher values.

# ### Box-plots

# In[327]:


# list of features
features = ['balance', 'age', 'duration', 'campaign']

# loop through each feature to create a boxplot
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='y', y=feature, data=data_train, palette="pastel")
    plt.title(f"Boxplot of {feature} by Target (y)")
    plt.xlabel("Target (y)")
    plt.ylabel(feature)
    plt.show()


# The boxplots show that `duration` is higher for `y=1`, meaning longer calls lead to better outcomes. `Campaign` has similar medians for both classes, with no clear pattern in outliers. `Balance` is slightly higher for `y=1`, but most values are low for both. `Age` overlaps a lot, showing little difference between outcomes. `Duration` stands out as the most useful feature.
# 

# ### Correlation plot

# In[330]:


# for the purpose of plotting the correlation heatmap
# for all features, lets encode categorical features

data_train = X_train.copy()
data_train['y'] = y_train

categorical_columns = X_train.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data_train[col] = le.fit_transform(data_train[col])
    label_encoders[col] = le

plt.figure(figsize=(20,15))
sns.heatmap(data_train.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.show()


# The heatmap shows that `duration` is the most correlated with the target (`y`), while `pdays` and `previous` are highly correlated with each other. Most other features show weak or no correlation, indicating small linear relationships.

# ## Final Decision
# 
# Based on the visualizations and statistical tests, we have decided to remove **Default** and **Campaign** as they show weak relationships with the target variable (`y`) and are unlikely to contribute meaningfully to the model's predictions.
# 

# In[333]:


X_train= X_train.drop(columns=['default','campaign'])
X_test= X_test.drop(columns=['default','campaign'])


# In[334]:


X_train.head()


# In[335]:


X_test.head()


# In[336]:


print(X_train.shape, X_test.shape)


# In[337]:


print(y_train.shape, y_test.shape)


# In[338]:


X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Datasets successfully exported as CSV files.")

