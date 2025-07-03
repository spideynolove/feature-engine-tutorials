import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import GeometricWidthDiscretiser
data = pd.read_csv('../data/housing.csv')
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
disc = GeometricWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)
disc.binner_dict_
fig, ax = plt.subplots(1, 2)
X_train['LotArea'].hist(ax=ax[0], bins=10)
train_t['LotArea'].hist(ax=ax[1], bins=10)
