<br>
<!--
## <p align="center"> <img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="50"> https://github.com/Mindful-AI-Assistants/credit-card-prediction
### <p align="center">   Credit Card Defaults Prediction
-->

# <p align="center">  💳 Credit Card Defaults [Prediction]()
### <p align="center"> 📉 This project predicts credit card default risk using [data analysis and machine learning]().


<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/e02d364f-4f88-4068-a745-1ac2f58eda7e"/>

<br><br>


This repository aims to develop a predictive model for assessing credit card default risks, encompassing data analysis, feature engineering, and machine learning for accurate predictions.

Credit card default prediction involves using analytical approaches, such as data analysis techniques and statistical methods, to forecast the likelihood of an individual failing to repay their outstanding debt. This process typically includes:

<br>

1. [**Incorporating Alternative Variables**](): Adding geographical, behavioral, and consumption data to traditional factors like income, assets, and payment history enhances customer profiling.

2. [**Individual Credit Scoring**](): Evaluating credit scores on an individual basis for improved risk assessment.

3. [**Behavioral Profile Analysis**](): Assessing customer behavior to forecast potential defaults.

This strategy enables financial institutions to refine their credit granting processes and manage risk more efficiently.

<br>

## **Table of Contents**

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Theoretical Framework](#theoretical-framework)
4. [Dataset Description](#dataset-description)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Generated Graphs](#generated-graphs)
9. [Conclusion](#conclusion)
10. [How to Run](#how-to-run)
11. [Data Analysis Report](#data-analysis-report)
12. [Contribute](#contribute)
13. [GitHub Repository](#github-repository)
14. [Contact](#contact)
15. [References](#references)
16. [License](#license)

<br>

## [**Executive Summary**]()

This project aims to predict credit card defaults using a **Logistic Regression** model. Our primary focus is identifying significant factors such as payment history, educational level, and customer age, which influence the likelihood of default. The results of this project will help financial institutions make better decisions regarding risk management.



## [**Introduction**]()

Predicting credit card defaults is crucial for financial institutions. It allows them to better manage risks and prevent financial losses by identifying customers who are likely to default on their payments. This study uses a dataset of credit card customers and applies **Logistic Regression**, a common technique for binary classification, to predict default risk.



## [**Theoretical Framework**]()

- **Definition of Default**
- 
  Default occurs when a customer fails to meet their financial obligations within the specified timeframe. For financial institutions, this represents a significant risk, as recovering the money owed can be difficult and costly.

- **Credit Analysis and Predictive Modeling**  
  Credit analysis evaluates a customer’s financial profile, while predictive modeling anticipates future behavior, such as the likelihood of default, based on historical data.

- **Logistic Regression**  
  Logistic Regression is a statistical method used for binary classification problems (such as default or non-default). It calculates the probability of a customer defaulting based on specific features.



## [**Dataset Description**]()

### 👉🏻 Click here to get the [dataset](https://github.com/Mindful-AI-Assistants/credit-card-prediction/blob/3c2b535affd8448b5f925bec3d0346fa7d1722b9/Dataset/default%20of%20credit%20card%20clients.xls)

The dataset contains information on credit card customers, with variables such as:

- **LIMIT_BAL**: Total credit amount granted.
- **EDUCATION**: Education level of customers.
- **MARRIAGE**: Marital status (married, single, others).
- **AGE**: Age of the customer.
- **PAY_0 to PAY_6**: Payment status of the previous months.
- **BILL_AMT1 to BILL_AMT6**: Credit card bill amounts for the past six months.
- **default payment next month**: Indicator of default in the following month (1 = yes, 0 = no).


## [**Exploratory Data Analysis**]()

### Several visualizations were created to identify patterns in the data. Key insights include:

- **Education**: Lower education levels are associated with higher default rates.
- **Marital Status**: Single customers tend to default more than married customers.
- **Age**: Younger customers have higher default rates.
- **Credit Limit**: Lower credit limits are linked to higher default rates.
- **Payment History**: Customers with a history of delayed payments are more likely to default.
  

## [The following Python code loads, processes, and analyzes the data.]()

```python
copy code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
```

## [Load Dataset]()

```python
copy code

path = r'/path/to/dataset.xls'
defaults = pd.read_excel(path, engine="xlrd")
```

## [Preprocess Dataset]()

```python
copy code

defaults.columns = [col for col in defaults.iloc[0, :]]
defaults.drop(columns=["ID", "SEX"], inplace=True)
defaults.drop(index=0, inplace=True)
defaults.index = list(range(30000))
```

## [Adjust variables for consistency]()

```python
copy code

defaults["EDUCATION"] = defaults["EDUCATION"].apply(lambda x: 5 if x == 6 or x == 0 else x)
defaults["MARRIAGE"] = defaults["MARRIAGE"].apply(lambda x: 3 if x == 0 else x)
```

## [**Methodology**]()

### **Data Preparation**

- **Cleaning and Preparation**: We removed irrelevant columns (e.g., `ID`) and sensitive variables (e.g., `SEX`).
  
- **Transformations**: Adjustments were made to ensure data consistency in `EDUCATION` and `MARRIAGE`.
  

## [**Model Development**]()

The data was split into training (80%) and testing (20%) sets. The model was trained using **Logistic Regression**, which is ideal for binary classification tasks such as predicting defaults.

```python
copy code

# Split data into training and testing sets
X = defaults.drop(columns=["MARRIAGE", "default payment next month"], axis=1)
y = defaults["default payment next month"]
```

## [Standardize features]()

```python
copy code

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

## [Train-test split]()

```python
copy code

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

## [Train logistic regression model]()

```python
coph code

model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train, y_train)
```

## [**Model Evaluation**]()

We evaluated the model’s performance using several metrics, including **accuracy**, a **confusion matrix**, and a **classification report**.

```python
copy code

# Model evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

## [Accuracy]()

```python
copy code

train_accuracy = round(accuracy_score(y_train, y_train_pred) * 100, 2)
test_accuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)
```

## [Confusion Matrix]()

```python
copy code

matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(matrix, annot=True, fmt='d', cmap='viridis')
```


## [Display evaluation metrics]()

```python
copy code

print(f"Training Accuracy: {train_accuracy}%")
print(f"Test Accuracy: {test_accuracy}%")
print(classification_report(y_test, y_test_pred))
```

## [**Results**]()

The ***Logistic Regression*** model achieved an accuracy of approximately **80%**. The confusion matrix and classification report demonstrated that the model was able to differentiate between defaulters and non-defaulters with reasonable efficiency. 

## [**Generated Graphs**]()

### Here are the key visualizations, their corresponding code, and descriptions:

### 1. [**Default Distribution by Educational Level**]()

```python
copy code

# Plotting default rate by education level
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=defaults, x="EDUCATION", hue="default payment next month", palette="viridis", ax=ax)
ax.set_title("Default Distribution by Educational Level")
ax.set_xlabel("Education Level")
ax.set_ylabel("Count")
ax.set_xticklabels(["Graduate School", "University", "High School", "Others", "Unknown"])
plt.show()
```


**Description**: This graph shows the distribution of defaults across different education levels, indicating that individuals with lower education levels tend to have a higher likelihood of defaulting on their payments.

<br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/c04b1fa4-1a31-4b50-a9d2-87fd0fa3af69" />
   
<br><br>

### 2. [**Proportion of Defaulters and Non-Defaulters by Education**]()

```python
copy code

# Proportions of default vs non-default by education level using heatmap
aux = defaults.copy()
aux_education = aux.groupby("EDUCATION")["default payment next month"].value_counts(normalize=True).unstack()
plt.figure(figsize=(8, 6))
sns.heatmap(aux_education, annot=True, cmap="viridis", fmt=".2f")
plt.title("Proportion of Defaulters and Non-Defaulters by Education")
plt.show()
```

**Description**: This heatmap illustrates the proportions of defaulters and non-defaulters based on education levels. It reveals that higher education correlates with a lower probability of default.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/5b64ac02-eca3-47ee-80c0-fd3b7d17fdf8 "/>

<br><br>

### 3. [**Default Distribution by Marital Status**]()

```python
copy code

# Plotting default rate by marital status
plt.figure(figsize=(10, 6))
sns.countplot(data=defaults, x="MARRIAGE", hue="default payment next month", palette="viridis")
plt.title("Default Distribution by Marital Status")
plt.xticks(ticks=[0, 1, 2], labels=["Married", "Single", "Other"])
plt.show()
```

**Description**: This chart displays the distribution of defaults across different marital statuses, indicating that single individuals have a higher tendency to default compared to married individuals.


<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/ba6effbe-02d2-4e29-94df-60bf24aa70e8"/>

<br><br>

### 4. [**Proportion of Defaulters by Marital Status**]()

```python
copy code

# Proportions of default vs non-default by marital status using heatmap
aux_marriage = aux.groupby("MARRIAGE")["default payment next month"].value_counts(normalize=True).unstack()
plt.figure(figsize=(8, 6))
sns.heatmap(aux_marriage, annot=True, cmap="viridis", fmt=".2f")
plt.title("Proportion of Defaulters by Marital Status")
plt.show()
```

**Description**: The heatmap displays the proportions of defaulters and non-defaulters categorized by marital status, showing minimal variation among the different groups.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/9762e78a-e481-4ca1-a788-fcaa42d57cb0"/>

<br><br>

### 5. [**Default and Non-Default Rates by Age**]()

```python
copy code

# Plotting default rate by age
plt.figure(figsize=(17, 9))
sns.countplot(data=defaults, x="AGE", hue="default payment next month", palette="viridis")
plt.title("Default and Non-Default Rates by Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

**Description**: This graph indicates that the number of non-defaulters decreases more significantly with age, suggesting that older customers are less likely to default.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/48e8fe17-4362-4e94-849c-d40833ccc12c"/>

<br><br>

### 6. **Default and Non-Default Rates by Credit Limit**

```python
# Plotting default rate by credit limit quantiles
aux['LIMIT_BAL_quantile'] = pd.qcut(defaults['LIMIT_BAL'], q=4, labels=["Up to 50,000", "50,000 to 140,000", "140,000 to 240,000", "Above 240,000"])
plt.figure(figsize=(15, 8))
sns.countplot(data=aux, x="LIMIT_BAL_quantile", hue="default payment next month", palette="viridis")
plt.title("Default and Non-Default Rates by Credit Limit")
plt.show()
```


<br>

img

<br><br>















## [Contributors]() 

<br>

- [Fabiana 🚀 Campanari](https://github.com/FabianaCampanari)
- [Pedro Vyctor](https://github.com/ppvyctor)


<br><br>


  <p align="center"> <a href="#top">Back to Top</a>

#

##### <p align="center">Copyright 2024 Mindful-AI-Assistants. Code released under the  [MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)




