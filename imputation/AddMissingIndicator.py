import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()
imputer = AddMissingIndicator(variables=['Alley', 'MasVnrType',
    'LotFrontage', 'MasVnrArea'])
imputer.fit(X_train)
imputer.variables_
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].head(
    )
train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].mean(
    )
X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].dtypes
pipe = Pipeline([('indicators', AddMissingIndicator(variables=['Alley',
    'MasVnrType', 'LotFrontage', 'MasVnrArea'])), ('imputer_num',
    MeanMedianImputer(imputation_method='median', variables=['LotFrontage',
    'MasVnrArea'])), ('imputer_cat', CategoricalImputer(imputation_method=
    'frequent', variables=['Alley', 'MasVnrType']))])
pipe.fit(X_train)
pipe.named_steps['indicators'].variables_
pipe.named_steps['imputer_num'].imputer_dict_
pipe.named_steps['imputer_cat'].imputer_dict_
train_t = pipe.transform(X_train)
test_t = pipe.transform(X_test)
vars_ = ['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea', 'Alley_na',
    'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']
train_t[vars_].head()
train_t[vars_].isnull().sum()
imputer = AddMissingIndicator(variables=None, missing_only=True)
imputer.fit(X_train)
imputer.variables
imputer.variables_
len(imputer.variables_)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
X_train.shape, train_t.shape
train_t.head()
imputer = AddMissingIndicator(variables=None, missing_only=False)
imputer.fit(X_train)
len(imputer.variables_)
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
X_train.shape, train_t.shape
pipe = Pipeline([('indicators', AddMissingIndicator(missing_only=True)), (
    'imputer_num', MeanMedianImputer(imputation_method='median')), (
    'imputer_cat', CategoricalImputer(imputation_method='frequent'))])
pipe.fit(X_train)
pipe.named_steps['indicators'].variables_
pipe.named_steps['imputer_num'].imputer_dict_
pipe.named_steps['imputer_cat'].imputer_dict_
train_t = pipe.transform(X_train)
test_t = pipe.transform(X_test)
train_t.isnull().sum()
[v for v in train_t.columns if train_t[v].isnull().sum() > 1]
