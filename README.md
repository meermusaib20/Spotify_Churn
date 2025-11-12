<!--
  Spotify Churn Analysis
  Author: Mir Musaib
  GitHub: github.com/meermusaib20/Spotify-Churn-Analysis
-->

# 🎵 Spotify User Churn Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Applied-green)
![Data Analytics](https://img.shields.io/badge/Data%20Analytics-Exploratory-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

### 📘 Overview

**Spotify Churn Analysis** is a **data-driven machine learning project** aimed at predicting user churn — identifying which users are likely to stop using Spotify’s services.  
By analyzing user activity data, subscription details, and engagement patterns, the project applies **classification algorithms** to uncover insights and improve retention strategies.

This project demonstrates the end-to-end data science workflow: **data cleaning → feature engineering → model training → evaluation → insights**.

---

### 🎯 Objectives

- Identify patterns that lead to user churn  
- Build predictive models to forecast churn behavior  
- Analyze key features influencing user retention  
- Provide actionable insights for data-driven business decisions  

---

### ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Data Preprocessing** | Feature scaling, encoding, outlier removal |
| **Modeling Techniques** | Logistic Regression, Decision Tree, Random Forest |
| **Environment** | Jupyter Notebook / Google Colab |
| **Version Control** | Git, GitHub |

---

### 🧩 Project Workflow

1. **Data Collection**  
   - Simulated or open-source dataset representing Spotify user activity and subscription data.

2. **Data Preprocessing**  
   - Handled missing values, encoded categorical features, normalized numeric columns.  
   - Removed outliers and irrelevant attributes.

3. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions, correlations, and feature importance using Matplotlib and Seaborn.  
   - Identified behavioral patterns between active and churned users.

4. **Feature Engineering**  
   - Created new variables such as *listening frequency*, *subscription duration*, *plan type*, etc.

5. **Model Building**  
   - Trained multiple ML models (Logistic Regression, Random Forest, Decision Tree).  
   - Compared accuracy, precision, recall, and F1-score to select the best model.

6. **Model Evaluation**  
   - Used a confusion matrix and ROC-AUC to assess performance.  
   - Interpreted feature importance for actionable insights.

---

### 📊 Sample Code Snippet

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions & evaluation
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
