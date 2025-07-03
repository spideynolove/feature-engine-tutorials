import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.transformation import LogTransformer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train['LotArea'].hist(bins=50)
X_train['GrLivArea'].hist(bins=50)
lt = LogTransformer(variables=['LotArea', 'GrLivArea'], base='e')
lt.fit(X_train)
lt.variables_
train_t = lt.transform(X_train)
test_t = lt.transform(X_test)
train_t['LotArea'].hist(bins=50)
train_t['GrLivArea'].hist(bins=50)
train_orig = lt.inverse_transform(train_t)
test_orig = lt.inverse_transform(test_t)
train_orig['LotArea'].hist(bins=50)
train_orig['GrLivArea'].hist(bins=50)
variables = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea',
    'TotRmsAbvGrd', 'SalePrice']
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)
arbitrary_imputer.fit(X_train)
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)
numeric_columns = train_t.select_dtypes(include=['int64', 'float64']).columns
train_numeric = train_t[numeric_columns].copy()
train_numeric
meaningful_zeros = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'PoolArea']
area_columns = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
train_shifted = train_numeric.copy()
for col in area_columns:
    train_shifted[col] = train_numeric[col] + 1
variables_to_transform = [col for col in train_numeric.columns if col not in
    meaningful_zeros]
lt = LogTransformer(base='10', variables=variables_to_transform)
lt.fit(train_shifted)
lt.variables_
train_t['GrLivArea'].hist(bins=50)
plt.title('GrLivArea')
train_t['LotArea'].hist(bins=50)
plt.title('LotArea')
train_t.columns
train_shifted = train_t.copy()
test_shifted = test_t.copy()
for col in area_columns:
    if col in train_t.columns:
        train_shifted[col] = train_t[col] + 1
    if col in test_t.columns:
        test_shifted[col] = test_t[col] + 1
variables_to_transform = [col for col in train_numeric.columns if col not in
    meaningful_zeros]
lt = LogTransformer(base='10', variables=variables_to_transform)
lt.fit(train_shifted)
train_transformed = lt.transform(train_shifted)
test_transformed = lt.transform(test_shifted)
train_t[variables_to_transform] = train_transformed[variables_to_transform]
test_t[variables_to_transform] = test_transformed[variables_to_transform]
train_t['GrLivArea'].hist(bins=50)
plt.title('GrLivArea')
train_t['LotArea'].hist(bins=50)
plt.title('LotArea')
train_orig = lt.inverse_transform(train_t)
test_orig = lt.inverse_transform(test_t)
train_orig['LotArea'].hist(bins=50)
train_orig['GrLivArea'].hist(bins=50)
