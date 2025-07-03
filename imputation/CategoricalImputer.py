import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import CategoricalImputer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['Alley', 'MasVnrType']].isnull().mean()
X_train['MasVnrType'].value_counts().plot.bar()
plt.ylabel('Number of observations')
plt.title('MasVnrType')
imputer = CategoricalImputer(imputation_method='missing', variables=[
    'Alley', 'MasVnrType'])
imputer.fit(X_train)
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
test_t['MasVnrType'].value_counts().plot.bar()
plt.ylabel('Number of observations')
plt.title('Imputed MasVnrType')
test_t['Alley'].value_counts().plot.bar()
plt.ylabel('Number of observations')
plt.title('Imputed Alley')
imputer = CategoricalImputer(variables='MasVnrType', fill_value=
    'this_is_missing')
train_t = imputer.fit_transform(X_train)
test_t = imputer.transform(X_test)
imputer.imputer_dict_
test_t['MasVnrType'].value_counts().plot.bar()
plt.ylabel('Number of observations')
plt.title('Imputed MasVnrType')
imputer = CategoricalImputer(imputation_method='frequent', variables=[
    'Alley', 'MasVnrType'])
imputer.fit(X_train)
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
X_train['MasVnrType'].value_counts()
train_t['MasVnrType'].value_counts()
imputer = CategoricalImputer(imputation_method='frequent')
imputer.fit(X_train)
imputer.imputer_dict_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
[v for v in train_t.columns if train_t[v].dtypes == 'O' and train_t[v].
    isnull().sum() > 1]
imputer.get_feature_names_out()
