# Generated from: EqualFrequencyDiscretiser.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # EqualFrequencyDiscretiser
#
# The EqualFrequencyDiscretiser() divides continuous numerical variables
# into contiguous equal frequency intervals, that is, intervals that contain
# approximately the same proportion of observations.
#
# The interval limits are determined by the quantiles. The number of intervals,
# i.e., the number of quantiles in which the variable should be divided is
# determined by the user.
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

from feature_engine.discretisation import EqualFrequencyDiscretiser

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


# we will use two continuous variables for transformation

X_train[["LotArea", 'GrLivArea']].hist(bins=50)

plt.show()


# The EqualFrequencyDiscretiser() works only with numerical variables.
# A list of variables can be passed as argument. Alternatively, the discretiser
# will automatically select and transform all numerical variables.
#
# The EqualFrequencyDiscretiser() first finds the boundaries for the intervals or
# quantiles for each variable, fit.
#
# Then it transforms the variables, that is, it sorts the values into the intervals,
# transform.


'''
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
'''

efd = EqualFrequencyDiscretiser(q=10, variables=['LotArea', 'GrLivArea'])

efd.fit(X_train)


# binner_dict contains the boundaries of the different bins
efd.binner_dict_


train_t = efd.transform(X_train)
test_t = efd.transform(X_test)


# the numbers are the different bins into which the observations
# were sorted
train_t['GrLivArea'].unique()


# the numbers are the different bins into which the observations
# were sorted
train_t['LotArea'].unique()


# here I put side by side the original variable and the transformed variable
tmp = pd.concat([X_train[["LotArea", 'GrLivArea']], train_t[["LotArea", 'GrLivArea']]], axis=1)
tmp.columns = ["LotArea", 'GrLivArea',"LotArea_binned", 'GrLivArea_binned']
tmp.head()


# in  equal frequency discretisation, we obtain the same amount of observations
# in each one of the bins.
plt.subplot(1,2,1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')

plt.subplot(1,2,2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of observations per interval')

plt.show()


# ### Return interval limits instead


# Now, let's return bin boundaries instead

efd = EqualFrequencyDiscretiser(
    q=10, variables=['LotArea', 'GrLivArea'], return_boundaries=True)

efd.fit(X_train)


train_t = efd.transform(X_train)
test_t = efd.transform(X_test)


# the numbers are the different bins into which the observations
# were sorted
np.sort(np.ravel(train_t['GrLivArea'].unique()))


np.sort(np.ravel(test_t['GrLivArea'].unique()))

