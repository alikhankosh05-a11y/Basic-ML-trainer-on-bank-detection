# Bank Customer Churn Prediction

Machine Learning project for predicting customer churn in a bank.

---

## Project Overview

This project aims to predict whether a bank customer will **churn** (leave the bank) or stay using various machine learning models.  
The target variable is `Attrition_Flag`:

- **0** — Existing Customer  
- **1** — Attrited Customer (Churn)

---

## Dataset

- **File**: `BankChurners.csv`
- **Shape**: 10,127 rows × 23 columns (before preprocessing)
- Contains demographic, behavioral, and financial features of bank customers.

---

## Project Workflow

1. **Data Loading & Exploratory Analysis**
2. **Data Preprocessing**
   - Removed irrelevant columns (`CLIENTNUM`, Naive Bayes classifier columns)
   - Encoded categorical variables using One-Hot Encoding
   - Converted target variable to binary (0/1)
3. **Outlier Removal** using IQR method
4. **Train-Test Split** (stratified, 80/20)
5. **Feature Scaling** with `StandardScaler`
6. **Model Training & Evaluation**:
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
7. **Hyperparameter Tuning** for Random Forest using `GridSearchCV`

---

## Models & Results

| Model                        | Accuracy   | Notes                          |
|-----------------------------|------------|--------------------------------|
| Decision Tree               | 0.9344     | Baseline                       |
| K-Nearest Neighbors         | -          | -                              |
| Support Vector Machine      | -          | -                              |
| **Random Forest**           | **0.9561** | Best before tuning             |
| **Random Forest (Tuned)**   | **0.9576** | **Best Model**                 |

**Best Model**: Random Forest with tuned parameters (`max_depth=20`, `n_estimators=600`).

---

## Technologies Used

- **Python 3**
- **pandas, numpy**
- **matplotlib, seaborn** — Data Visualization
- **scikit-learn** — Modeling, Preprocessing & Evaluation
- Jupyter Notebook

---

## Project Structure
