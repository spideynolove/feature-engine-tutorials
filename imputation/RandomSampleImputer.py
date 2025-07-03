import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import RandomSampleImputer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
imputer = RandomSampleImputer(variables=['Alley', 'MasVnrType',
    'LotFrontage', 'MasVnrArea'], random_state=10, seed='general')
imputer.fit(X_train)
imputer.X_.head()
X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
imputer = RandomSampleImputer(random_state=['MSSubClass', 'YrSold'], seed=
    'observation', seeding_method='add', variables=None)
imputer.fit(X_train)
imputer.X_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
test_t.isnull().sum()
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
