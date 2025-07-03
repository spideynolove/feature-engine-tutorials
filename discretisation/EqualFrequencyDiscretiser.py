import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser
plt.rcParams['figure.figsize'] = [15, 5]
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['LotArea', 'GrLivArea']].hist(bins=50)
plt.show()
"""
Parameters
----------

q : int, default=10
    Desired number of equal frequency intervals / bins. In other words the
    number of quantiles in which the variables should be divided.

variables : list
    The list of numerical variables that will be discretised. If None, the
    EqualFrequencyDiscretiser() will select all numerical variables.

return_object : bool, default=False
    Whether the numbers in the discrete variable should be returned as
    numeric or as object. The decision is made by the user based on
    whether they would like to proceed the engineering of the variable as
    if it was numerical or categorical.

return_boundaries: bool, default=False
    whether the output should be the interval boundaries. If True, it returns
    the interval boundaries. If False, it returns integers.
"""
efd = EqualFrequencyDiscretiser(q=10, variables=['LotArea', 'GrLivArea'])
efd.fit(X_train)
efd.binner_dict_
train_t = efd.transform(X_train)
test_t = efd.transform(X_test)
train_t['GrLivArea'].unique()
train_t['LotArea'].unique()
tmp = pd.concat([X_train[['LotArea', 'GrLivArea']], train_t[['LotArea',
    'GrLivArea']]], axis=1)
tmp.columns = ['LotArea', 'GrLivArea', 'LotArea_binned', 'GrLivArea_binned']
tmp.head()
plt.subplot(1, 2, 1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')
plt.subplot(1, 2, 2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')
plt.show()
efd = EqualFrequencyDiscretiser(q=10, variables=['LotArea', 'GrLivArea'],
    return_boundaries=True)
efd.fit(X_train)
train_t = efd.transform(X_train)
test_t = efd.transform(X_test)
np.sort(np.ravel(train_t['GrLivArea'].unique()))
np.sort(np.ravel(test_t['GrLivArea'].unique()))
