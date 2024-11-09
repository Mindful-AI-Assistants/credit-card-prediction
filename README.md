<br>
<!--
## <p align="center"> <img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="50"> https://github.com/Mindful-AI-Assistants/credit-card-prediction
### <p align="center">   Credit Card Defaults Prediction
-->

## <p align="center">  💳 Credit Card Defaults Prediction
### <p align="center"> 📉 The project predicts credit card default risk using data analysis and machine learning.


<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/12187c72-c579-41e8-99ec-d3fc806e2995"/>

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
  Default occurs when a customer fails to meet their financial obligations within the specified timeframe. For financial institutions, this represents a significant risk, as recovering the money owed can be difficult and costly.

- **Credit Analysis and Predictive Modeling**  
  Credit analysis evaluates a customer’s financial profile, while predictive modeling anticipates future behavior, such as the likelihood of default, based on historical data.

- **Logistic Regression**  
  Logistic Regression is a statistical method used for binary classification problems (such as default or non-default). It calculates the probability of a customer defaulting based on specific features.



## [**Dataset Description**]()
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
  

## Here is the code in Python used to load, process, and analyze the data.

### 

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

## Load Dataset

### 👉🏻 Click here to get the [dataset](https://github.com/Mindful-AI-Assistants/credit-card-prediction/blob/3c2b535affd8448b5f925bec3d0346fa7d1722b9/Dataset/default%20of%20credit%20card%20clients.xls)

```python
copy code

path = r'/path/to/dataset.xls'
defaults = pd.read_excel(path, engine="xlrd")
```

## Preprocess Dataset

```python
copy code

defaults.columns = [col for col in defaults.iloc[0, :]]
defaults.drop(columns=["ID", "SEX"], inplace=True)
defaults.drop(index=0, inplace=True)
defaults.index = list(range(30000))
```

























## [Contributors]() 

<br>

- [Fabiana 🚀 Campanari](https://github.com/FabianaCampanari)
- [Pedro Vyctor](https://github.com/ppvyctor)

#

##### <p align="center">Copyright 2024 Mindful-AI-Assistants. Code released under the  [MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)




