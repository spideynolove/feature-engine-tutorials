import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import EndTailImputer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()
imputer = EndTailImputer(imputation_method='gaussian', tail='right', fold=3,
    variables=['LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t['LotFrontage'].isnull().sum()
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
imputer = EndTailImputer(imputation_method='iqr', tail='left', fold=3,
    variables=['LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[['LotFrontage', 'MasVnrArea']].isnull().sum()
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
imputer = EndTailImputer(imputation_method='max', fold=3, variables=[
    'LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
imputer.imputer_dict_
X_train[imputer.variables_].max()
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[['LotFrontage', 'MasVnrArea']].isnull().sum()
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
imputer = EndTailImputer()
imputer.imputation_method
imputer.tail
imputer.fold
imputer.fit(X_train)
imputer.variables_
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
[v for v in train_t.columns if train_t[v].dtypes != 'O' and train_t[v].
    isnull().sum() > 1]
