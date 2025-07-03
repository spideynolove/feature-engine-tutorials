import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from feature_engine.wrappers import SklearnTransformerWrapper
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()
imputer = SklearnTransformerWrapper(transformer=SimpleImputer(strategy=
    'mean'), variables=['LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
imputer.transformer_.statistics_
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()
cols = [c for c in train_df.columns if train_df[c].dtypes == 'O' and 
    train_df[c].isnull().sum() > 0]
train_df[cols].head()
imputer = SklearnTransformerWrapper(transformer=SimpleImputer(strategy=
    'most_frequent'), variables=cols)
imputer.fit(X_train)
imputer.transformer_.statistics_
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
X_train[cols].isnull().mean()
X_test[cols].head()
