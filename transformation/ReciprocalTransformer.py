import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.transformation import ReciprocalTransformer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
rt = ReciprocalTransformer(variables=['LotArea', 'GrLivArea'])
rt.fit(X_train)
rt.variables_
train_t = rt.transform(X_train)
test_t = rt.transform(X_test)
X_train['GrLivArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('GrLivArea')
train_t['GrLivArea'].hist(bins=50)
plt.title('Transformed variable')
plt.xlabel('GrLivArea')
X_train['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')
train_t['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')
train_orig = rt.inverse_transform(train_t)
test_orig = rt.inverse_transform(test_t)
train_orig['LotArea'].hist(bins=50)
train_orig['GrLivArea'].hist(bins=50)
variables = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea',
    'TotRmsAbvGrd', 'SalePrice']
train_df = pd.read_csv('../data/house-prices/train.csv', usecols=['Id'] +
    variables)
test_df = pd.read_csv('../data/house-prices/test.csv', usecols=['Id'] +
    variables[:-1])
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)
arbitrary_imputer.fit(X_train)
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)
rt = ReciprocalTransformer()
rt.fit(train_t)
rt.variables_
train_t['GrLivArea'].hist(bins=50)
train_t['LotArea'].hist(bins=50)
train_t = rt.transform(train_t)
test_t = rt.transform(test_t)
train_t['GrLivArea'].hist(bins=50)
train_t['LotArea'].hist(bins=50)
train_orig = rt.inverse_transform(train_t)
test_orig = rt.inverse_transform(test_t)
train_orig['LotArea'].hist(bins=50)
train_orig['GrLivArea'].hist(bins=50)
