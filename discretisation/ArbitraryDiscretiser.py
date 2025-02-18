# # ArbitraryDiscretiser
# The ArbitraryDiscretiser() divides continuous numerical variables into contiguous intervals are arbitrarily entered by the user.
# The user needs to enter a dictionary with variable names as keys, and a list of the limits of the intervals as values. For example {'var1': [0, 10, 100, 1000], 'var2': [5, 10, 15, 20]}.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.discretisation import ArbitraryDiscretiser
plt.rcParams["figure.figsize"] = [15,5]

# Read the separate files
train_df = pd.read_csv('house-prices/train.csv')
test_df = pd.read_csv('house-prices/test.csv')

# Separate features and target in training data
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']

# For test data, you might not have the target variable
X_test = test_df.drop(['Id'], axis=1)  # Note: test data might not have SalePrice column

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

# we will discretise two continuous variables
X_train[["LotArea", 'GrLivArea']].hist(bins=50)
plt.show()


# The ArbitraryDiscretiser() works only with numerical variables. The discretiser will
# check if the dictionary entered by the user contains variables present in the
# training set, and if these variables are cast as numerical, before doing any
# transformation.
#
# Then it transforms the variables, that is, it sorts the values into the intervals,
# transform.

'''
Parameters
----------

binning_dict : dict
    The dictionary with the variable : interval limits pairs, provided by the user.
    A valid dictionary looks like this:

     binning_dict = {'var1':[0, 10, 100, 1000], 'var2':[5, 10, 15, 20]}.

return_object : bool, default=False
    Whether the numbers in the discrete variable should be returned as
    numeric or as object. The decision is made by the user based on
    whether they would like to proceed the engineering of the variable as
    if it was numerical or categorical.

return_boundaries: bool, default=False
    whether the output should be the interval boundaries. If True, it returns
    the interval boundaries. If False, it returns integers.
'''

atd = ArbitraryDiscretiser(binning_dict={"LotArea":[-np.inf,4000,8000,12000,16000,20000,np.inf],
                                        "GrLivArea":[-np.inf,500,1000,1500,2000,2500,np.inf]})

atd.fit(X_train)

# binner_dict contains the boundaries of the different bins
atd.binner_dict_

train_t = atd.transform(X_train)
test_t = atd.transform(X_test)

# the below are the bins into which the observations were sorted
print(train_t['GrLivArea'].unique())
print(train_t['LotArea'].unique())

# here I put side by side the original variable and the transformed variable
tmp = pd.concat([X_train[["LotArea", 'GrLivArea']], train_t[["LotArea", 'GrLivArea']]], axis=1)
tmp.columns = ["LotArea", 'GrLivArea',"LotArea_binned", 'GrLivArea_binned']
tmp.head()


plt.subplot(1,2,1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per bin')
plt.subplot(1,2,2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per bin')

plt.show()


# ### Now return interval boundaries instead
atd = ArbitraryDiscretiser(binning_dict={"LotArea": [-np.inf, 4000, 8000, 12000, 16000, 20000, np.inf],
                                         "GrLivArea": [-np.inf, 500, 1000, 1500, 2000, 2500, np.inf]},
                           # to return the boundary limits
                           return_boundaries=True)

atd.fit(X_train)


train_t = atd.transform(X_train)
test_t = atd.transform(X_test)


# the numbers are the different bins into which the observations
# were sorted
np.sort(np.ravel(train_t['GrLivArea'].unique()))

np.sort(np.ravel(test_t['GrLivArea'].unique()))

# bar plot to show the intervals returned by the transformer
test_t.LotArea.value_counts(sort=False).plot.bar(figsize=(6,4))
plt.ylabel('Number of houses')
plt.title('Number of houses per interval')
plt.show()

