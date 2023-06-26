# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:10:27 2023

@author: shirt
"""

import numpy as np
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from madlan_data_prep import prepare_data

dataset = prepare_data("C:\\Users\\shirt\Desktop\\מטלה מסכמת\\output_all_students_Train_v10.csv")

correlation_matrix = dataset.corr()['price'].to_frame()
# שימוש במטריצת הקורלציה ליצירת מפת חום
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix , annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# חישוב מטריצת הקורלציה
corr_matrix = dataset.corr()
print(correlation_matrix)
# שימוש במטריצת הקורלציה ליצירת מפת חום
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



X = dataset.iloc[:, :-1]
y = dataset.loc[:, 'price'].values

rkf = RepeatedKFold(n_splits=10, random_state=42)
rkf.get_n_splits(X, y)
alpha=0.0001
l1_ratio=0.9
for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

#dataset.to_csv("dataset1.csv", index=False ,encoding = 'utf-8-sig')
num_cols = [col for col in X_train.columns if X_train[col].dtypes != 'O']
cat_cols = [col for col in X_train.columns if X_train[col].dtypes == 'O']

numerical_pipeline = Pipeline([('scaling', StandardScaler())])
categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

column_transformer = ColumnTransformer([
    ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)], remainder='drop')

pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio))])
pipe_preprocessing_model.fit(X_train, y_train)
y_pred = pipe_preprocessing_model.predict(X_test)

def score_model(y_test, y_pred, model_name):
    MSE = mse(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R_squared = r2_score(y_test, y_pred)
    print(f"Model: {model_name}, RMSE: {np.round(RMSE, 2)}")

score_model(y_test, y_pred, "ElasticNet")

joblib.dump(pipe_preprocessing_model, 'trained_model.pkl')