# Generated from: EqualWidthDiscretiser.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # EqualWidthDiscretiser
#
# The EqualWidthDiscretiser() divides continuous numerical variables into
# intervals of the same width, that is, equidistant intervals. Note that the
# proportion of observations per interval may vary.
#
# The number of intervals
# in which the variable should be divided must be indicated by the user.
#
# **Note**
#
# For this demonstration, we use the Ames House Prices dataset produced by Professor Dean De Cock:
#
# Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing
# Data as an End of Semester Regression Project, Journal of Statistics Education, Vol.19, No. 3
#
# http://jse.amstat.org/v19n3/decock.pdf
#
# https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627
#
# The version of the dataset used in this notebook can be obtained from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.discretisation import EqualWidthDiscretiser

plt.rcParams["figure.figsize"] = [15,5]


data = pd.read_csv('housing.csv')
data.head()


# let's separate into training and testing set
X = data.drop(["Id", "SalePrice"], axis=1)
y = data.SalePrice

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# we will discretise two continuous variables

X_train[["LotArea", 'GrLivArea']].hist(bins=50)
plt.show()


# The EqualWidthDiscretiser() works only with numerical variables.
# A list of variables can be passed as argument. Alternatively, the discretiser
# will automatically select all numerical variables.
#
# The EqualWidthDiscretiser() first finds the boundaries for the intervals for
# each variable, fit.
#
# Then, it transforms the variables, that is, sorts the values into the intervals,
# transform.


'''
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
'''

ewd = EqualWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])

ewd.fit(X_train)


# binner_dict contains the boundaries of the different bins
ewd.binner_dict_


train_t = ewd.transform(X_train)
test_t = ewd.transform(X_test)


# the below are the bins into which the observations were sorted
train_t['GrLivArea'].unique()


# here I put side by side the original variable and the transformed variable
tmp = pd.concat([X_train[["LotArea", 'GrLivArea']],
                 train_t[["LotArea", 'GrLivArea']]], axis=1)

tmp.columns = ["LotArea", 'GrLivArea', "LotArea_binned", 'GrLivArea_binned']

tmp.head()


# Note that the bins are not equally distributed
plt.subplot(1, 2, 1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')

plt.subplot(1, 2, 2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')

plt.show()


# ### Now return interval boundaries instead


ewd = EqualWidthDiscretiser(
    bins=10, variables=['LotArea', 'GrLivArea'], return_boundaries=True)

ewd.fit(X_train)


train_t = ewd.transform(X_train)
test_t = ewd.transform(X_test)


# the numbers are the different bins into which the observations
# were sorted
np.sort(np.ravel(train_t['GrLivArea'].unique()))


np.sort(np.ravel(test_t['GrLivArea'].unique()))


#the intervals are more or less of the same length
val = np.sort(np.ravel(train_t['GrLivArea'].unique()))
print(list(x.right-x.left for x in val)[1:-1])

