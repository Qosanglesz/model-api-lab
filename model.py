import pandas as pd
import numpy as np
import joblib

url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include].copy()  # Explicitly create a copy

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.items():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_.loc[:, col] = df_[col].fillna(0)  # Avoid SettingWithCopyWarning

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
