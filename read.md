# 🏦 Loan Prediction Using Linear Regression

## 📌 Project Overview

This project predicts the **loan amount** a customer may be eligible for using **Linear Regression**.

The model uses customer information such as income, loan term, credit history, and other financial details to learn the relationship between the input features and the loan amount.

The project is implemented using **Python and Google Colab**.

---

## 🎯 Objective

* Analyze loan applicant data.
* Preprocess the dataset.
* Select relevant features.
* Train a Linear Regression model.
* Predict loan amounts.
* Evaluate model performance.

---

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Dataset

The dataset contains information about loan applicants.

Example features:

| Feature           | Description          |
| ----------------- | -------------------- |
| Gender            | Applicant gender     |
| Married           | Marital status       |
| Dependents        | Number of dependents |
| Education         | Education level      |
| Self_Employed     | Employment status    |
| ApplicantIncome   | Applicant income     |
| CoapplicantIncome | Co-applicant income  |
| LoanAmount        | Loan amount          |
| Loan_Amount_Term  | Loan repayment term  |
| Credit_History    | Credit history       |

### Target Variable

`LoanAmount`

The model predicts the loan amount based on the applicant's information.

---

## 🚀 Running the Project in Google Colab

### Step 1: Open Google Colab

Open:

https://colab.research.google.com/

Create a new Python notebook.

### Step 2: Upload Dataset

Upload your CSV dataset to Colab.

### Step 3: Install / Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 📥 Load Dataset

```python
df = pd.read_csv("loan_data.csv")

print(df.head())
print(df.shape)
print(df.info())
```

---

## 🔍 Data Preprocessing

Check missing values:

```python
print(df.isnull().sum())
```

Fill numerical missing values:

```python
df['ApplicantIncome'] = df['ApplicantIncome'].fillna(
    df['ApplicantIncome'].median()
)

df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(
    df['CoapplicantIncome'].median()
)

df['LoanAmount'] = df['LoanAmount'].fillna(
    df['LoanAmount'].median()
)

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].median()
)

df['Credit_History'] = df['Credit_History'].fillna(
    df['Credit_History'].mode()[0]
)
```

---

## 🔄 Convert Categorical Data

```python
df = pd.get_dummies(
    df,
    drop_first=True
)
```

---

## ✂️ Select Features and Target

```python
X = df.drop('LoanAmount', axis=1)
y = df['LoanAmount']
```

---

## 📊 Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

* **80%** → Training data
* **20%** → Testing data

---

## 🤖 Train Linear Regression Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

---

## 🔮 Make Predictions

```python
y_pred = model.predict(X_test)

print("Predicted Loan Amounts:")
print(y_pred[:10])
```

---

## 📈 Model Evaluation

### Mean Absolute Error

```python
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
```

### Mean Squared Error

```python
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
```

### R² Score

```python
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
```

---

## 📉 Actual vs Predicted

```python
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Loan Amount")
plt.ylabel("Predicted Loan Amount")
plt.title("Actual vs Predicted Loan Amount")

plt.show()
```

---

## 🧪 Example Prediction

```python
sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print("Predicted Loan Amount:", prediction[0])
```

---

## 📌 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Prediction
   ↓
Model Evaluation
```

---

## 📊 Expected Output

```text
MAE: <value>
MSE: <value>
R2 Score: <value>

Predicted Loan Amount: <value>
```

The exact values depend on the dataset used.

---

## 💡 Key Learning

This project demonstrates how **Linear Regression** can be used to predict a continuous financial value such as loan amount from applicant-related features.

---

## 🔮 Future Improvements

* Try Random Forest Regression.
* Try XGBoost.
* Perform feature selection.
* Use hyperparameter tuning.
* Deploy the model using Flask or Streamlit.
* Create a web interface for loan prediction.

---

## 👩‍💻 Author

**Sowdhanya R**

B.E. Computer Science and Engineering

---

## ⭐ Conclusion

The Loan Prediction project demonstrates the complete machine-learning workflow, from **data preprocessing to model training, prediction, and evaluation**, using Linear Regression in Google Colab.
