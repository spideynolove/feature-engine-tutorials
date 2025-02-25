# Generated from: Sklearn-wrapper-plus-Categorical-Encoding.ipynb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder

# Read the separate files
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')

# Separate features and target in training data
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']

# For test data, you might not have the target variable
# Note: test data might not have SalePrice column
X_test = test_df.drop(['Id'], axis=1)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# ## OrdinalEncoder


cols = [
    'Alley',
    'MasVnrType',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Electrical',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
]


# let's remove rare labels to avoid errors when encoding

rare_label_enc = RareLabelEncoder(n_categories=2, variables=cols)

X_train = rare_label_enc.fit_transform(X_train.fillna('Missing'))
X_test = rare_label_enc.transform(X_test.fillna('Missing'))


# now let's replace categories by integers

encoder = SklearnTransformerWrapper(
    transformer=OrdinalEncoder(),
    variables=cols,
)

encoder.fit(X_train)


# we can navigate to the parameters of the sklearn transformer
# like this:

encoder.transformer_.categories_


# encode categories

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

X_train[cols].isnull().mean()


X_test[cols].head()
