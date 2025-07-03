import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual']
rare_label_enc = RareLabelEncoder(n_categories=2, variables=cols)
X_train = rare_label_enc.fit_transform(X_train.fillna('Missing'))
X_test = rare_label_enc.transform(X_test.fillna('Missing'))
encoder = SklearnTransformerWrapper(transformer=OrdinalEncoder(), variables
    =cols)
encoder.fit(X_train)
encoder.transformer_.categories_
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
X_train[cols].isnull().mean()
X_test[cols].head()
