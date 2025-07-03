import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
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
X_train[cols].hist(bins=50, figsize=(15, 15))
plt.show()
variables = ['GrLivArea', 'GarageArea']
X_train[variables].isnull().mean()
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
discretizer = Pipeline([('imputer', SklearnTransformerWrapper(transformer=
    SimpleImputer(strategy='median'), variables=variables)), ('discretizer',
    SklearnTransformerWrapper(transformer=KBinsDiscretizer(n_bins=5,
    strategy='quantile', encode='ordinal'), variables=variables))])
discretizer.fit(X_train)
X_train = discretizer.transform(X_train)
X_test = discretizer.transform(X_test)
X_test['GrLivArea'].value_counts(normalize=True)
X_test['GarageArea'].value_counts(normalize=True)
X_test[variables].hist()
plt.show()
