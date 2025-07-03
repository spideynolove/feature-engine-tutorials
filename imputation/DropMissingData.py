import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import DropMissingData
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
imputer = DropMissingData(variables=['Alley', 'MasVnrType', 'LotFrontage',
    'MasVnrArea'], missing_only=False)
imputer.fit(X_train)
imputer.variables_
X_train[imputer.variables].isna().sum()
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[imputer.variables].isna().sum()
X_train.shape
train_t.shape
tmp = imputer.return_na_data(X_train)
tmp.shape
1022 - 963
imputer = DropMissingData(variables=['Alley', 'MasVnrType', 'LotFrontage',
    'MasVnrArea'], missing_only=False, threshold=0.5)
imputer.fit(X_train)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[imputer.variables].isna().sum()
imputer = DropMissingData(missing_only=True)
imputer.fit(X_train)
imputer.variables_
X_train[imputer.variables_].isna().sum()
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[imputer.variables_].isna().sum()
train_t.shape
imputer = DropMissingData(missing_only=True, threshold=0.75)
imputer.fit(X_train)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t.shape
