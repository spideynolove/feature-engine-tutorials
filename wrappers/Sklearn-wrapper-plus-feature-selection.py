import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest, SelectFromModel
from sklearn.linear_model import Lasso
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
selector = SklearnTransformerWrapper(transformer=SelectKBest(f_regression,
    k=5), variables=cols)
selector.fit(X_train.fillna(0), y_train)
selector.transformer_.get_support(indices=True)
X_train.columns[selector.transformer_.get_support(indices=True)]
X_train_t = selector.transform(X_train.fillna(0))
X_test_t = selector.transform(X_test.fillna(0))
X_test_t.head()
lasso = Lasso(alpha=10000, random_state=0)
sfm = SelectFromModel(lasso, prefit=False)
selector = SklearnTransformerWrapper(transformer=sfm, variables=cols)
selector.fit(X_train.fillna(0), y_train)
selector.transformer_.get_support(indices=True)
len(selector.transformer_.get_support(indices=True))
len(cols)
X_train_t = selector.transform(X_train.fillna(0))
X_test_t = selector.transform(X_test.fillna(0))
X_test_t.head()
