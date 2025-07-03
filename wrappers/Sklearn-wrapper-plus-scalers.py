import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engine.wrappers import SklearnTransformerWrapper
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
cols = [var for var in X_train.columns if X_train[var].dtypes != 'O']
cols
scaler = SklearnTransformerWrapper(transformer=StandardScaler(), variables=cols
    )
scaler.fit(X_train.fillna(0))
X_train = scaler.transform(X_train.fillna(0))
X_test = scaler.transform(X_test.fillna(0))
scaler.transformer_.mean_
scaler.transformer_.scale_
X_train[cols].mean()
X_train[cols].std()
