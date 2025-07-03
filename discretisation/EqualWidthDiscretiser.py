import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualWidthDiscretiser
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

bins : int, default=10
    Desired number of equal width intervals / bins.

variables : list
    The list of numerical variables to transform. If None, the
    discretiser will automatically select all numerical type variables.

return_object : bool, default=False
    Whether the numbers in the discrete variable should be returned as
    numeric or as object. The decision should be made by the user based on
    whether they would like to proceed the engineering of the variable as
    if it was numerical or categorical.

return_boundaries: bool, default=False
    whether the output should be the interval boundaries. If True, it returns
    the interval boundaries. If False, it returns integers.
"""
ewd = EqualWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])
ewd.fit(X_train)
ewd.binner_dict_
train_t = ewd.transform(X_train)
test_t = ewd.transform(X_test)
train_t['GrLivArea'].unique()
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
ewd = EqualWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'],
    return_boundaries=True)
ewd.fit(X_train)
train_t = ewd.transform(X_train)
test_t = ewd.transform(X_test)
np.sort(np.ravel(train_t['GrLivArea'].unique()))
np.sort(np.ravel(test_t['GrLivArea'].unique()))
val = np.sort(np.ravel(train_t['GrLivArea'].unique()))
val
import re


def extract_upper_bound(interval_str):
    match = re.search('([0-9.]+)\\]$', interval_str)
    if match:
        return float(match.group(1))
    return None


upper_bounds = [extract_upper_bound(x) for x in val if extract_upper_bound(
    x) is not None]
upper_bounds.sort()
differences = np.diff(upper_bounds)
print(differences)


def extract_bounds(interval_str):
    numbers = re.findall('[-+]?\\d*\\.\\d+|\\d+', interval_str)
    if len(numbers) == 2:
        return float(numbers[0]), float(numbers[1])
    return None


bounds = [extract_bounds(x) for x in val if extract_bounds(x) is not None]
bounds.sort(key=lambda x: x[1])
interval_sizes = [(bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]
print(interval_sizes)
