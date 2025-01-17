# Generated from: GeometricWidthDiscretiser.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import GeometricWidthDiscretiser


# Load dataset
data = pd.read_csv('houseprice.csv')

# Separate into train and test sets
X_train, X_test, y_train, y_test =  train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'], test_size=0.3, random_state=0)


# set up the discretisation transformer
disc = GeometricWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])

# fit the transformer
disc.fit(X_train)


# transform the data
train_t= disc.transform(X_train)
test_t= disc.transform(X_test)


disc.binner_dict_


fig, ax = plt.subplots(1, 2)
X_train['LotArea'].hist(ax=ax[0], bins=10);
train_t['LotArea'].hist(ax=ax[1], bins=10);

