import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.transformation import BoxCoxTransformer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
bct = BoxCoxTransformer(variables=['LotArea', 'GrLivArea'])
bct.fit(X_train)
bct.lambda_dict_
train_t = bct.transform(X_train)
test_t = bct.transform(X_test)
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
numeric_columns
train_numeric = train_t[numeric_columns].copy()
for column in numeric_columns:
    min_val = train_numeric[column].min()
    if min_val <= 0:
        print(f'{column}: minimum value = {min_val}')
        shift = abs(min_val) + 1
        train_numeric[column] = train_numeric[column] + shift
train_numeric.describe()
for col in train_numeric.columns:
    q75 = train_numeric[col].quantile(0.75)
    q25 = train_numeric[col].quantile(0.25)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    if train_numeric[col].max() > upper_bound:
        print(f'\n{col}:')
        print(f'Max value: {train_numeric[col].max()}')
        print(f'Upper bound: {upper_bound}')
"""

from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer




problematic_cols = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
good_cols = [col for col in train_numeric.columns if col not in problematic_cols]
train_good = train_numeric[good_cols]

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
train_winsorized_good = winsor.fit_transform(train_good)

train_winsorized = pd.concat([train_winsorized_good, train_numeric[problematic_cols]], axis=1)


yjt = YeoJohnsonTransformer()
train_transformed = yjt.fit_transform(train_winsorized)

print("
Skewness before transformation:")
print(train_numeric.skew())
print("
Skewness after transformation:")
print(train_transformed.skew())

"""
bct = BoxCoxTransformer()
bct.fit(train_numeric)
bct.variables_
from feature_engine.transformation import YeoJohnsonTransformer
test_numeric = test_t[numeric_columns].copy()
yjt = YeoJohnsonTransformer()
yjt.fit(train_numeric)
train_t[numeric_columns] = yjt.transform(train_numeric)
test_t[numeric_columns] = yjt.transform(test_numeric)
bct.lambda_dict_
