# Generated from: Winsorizer.ipynb

# # Winsorizer
# Winzorizer finds maximum and minimum values following a Gaussian or skewed distribution as indicated. It can also cap the right, left or both ends of the distribution.
#
# The Winsorizer() caps maximum and / or minimum values of a variable.
#
# The Winsorizer() works only with numerical variables. A list of variables can
# be indicated. Alternatively, the Winsorizer() will select all numerical
# variables in the train set.
#
# The Winsorizer() first calculates the capping values at the end of the
# distribution. The values are determined using:
#
# - a Gaussian approximation,
# - the inter-quantile range proximity rule (IQR)
# - percentiles.
#

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.outliers import Winsorizer


# Load titanic dataset from file

def load_titanic(filepath='../data/titanic.csv'):
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['embarked'].fillna('C', inplace=True)
    data['fare'] = data['fare'].astype('float')
    data['fare'].fillna(data['fare'].median(), inplace=True)
    data['age'] = data['age'].astype('float')
    data['age'].fillna(data['age'].median(), inplace=True)
    data.drop(['name', 'ticket'], axis=1, inplace=True)
    return data

# To plot histogram of given numerical feature
def plot_hist(data, col):
    plt.figure(figsize=(8, 5))
    plt.hist(data[col], bins=30)
    plt.title("Distribution of " + col)
    return plt.show()


# Loading titanic dataset
data = load_titanic()
data.sample(5)


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data.drop('survived', axis=1),
                                                    data['survived'],
                                                    test_size=0.3,
                                                    random_state=0)

print("train data:", X_train.shape)
print("test data:", X_test.shape)


# let's find out the maximum Age and maximum Fare in the titanic

print("Max age:", data.age.max())
print("Max fare:", data.fare.max())


# Histogram of age feature before capping outliers
plot_hist(data, 'age')


# Histogram of fare feature before capping outliers
plot_hist(data, 'fare')


# ### Capping : Gaussian
#
# Gaussian limits:
# + right tail: mean + 3* std
# + left tail: mean - 3* std


'''Parameters
----------
capping_method : str, default=gaussian

    Desired capping method. Can take 'gaussian', 'iqr' or 'quantiles'.

tail : str, default=right

    Whether to cap outliers on the right, left or both tails of the distribution.
    Can take 'left', 'right' or 'both'.

fold: int or float, default=3

    How far out to to place the capping values. The number that will multiply
    the std or IQR to calculate the capping values. Recommended values, 2
    or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity
    rule.

variables: list, default=None
  
missing_values: string, default='raise'

    Indicates if missing values should be ignored or raised.
'''
# capping at right tail using gaussian capping method
capper = Winsorizer(
    capping_method='gaussian', tail='right', fold=3, variables=['age', 'fare'])

# fitting winsorizer object to training data
capper.fit(X_train)


# here we can find the maximum caps allowed
capper.right_tail_caps_


# this dictionary is empty, because we selected only right tail
capper.left_tail_caps_


# # Histogram of age feature after capping outliers
plot_hist(capper.transform(X_train), 'age')


# transforming the training and testing data
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

# let's check the new maximum Age and maximum Fare in the titanic
train_t.age.max(), train_t.fare.max()


# ### Gaussian approximation capping, both tails


# Capping the outliers at both tails using gaussian capping method

winsor = Winsorizer(capping_method='gaussian',
                    tail='both', fold=2, variables='fare')
winsor.fit(X_train)


print("Minimum caps :", winsor.left_tail_caps_)

print("Maximum caps :", winsor.right_tail_caps_)


# Histogram of fare feature after capping outliers
plot_hist(winsor.transform(X_train), 'fare')


# transforming the training and testing data
train_t = winsor.transform(X_train)
test_t = winsor.transform(X_test)

print("Max fare:", train_t.fare.max())
print("Min fare:", train_t.fare.min())


# ### Inter Quartile Range, both tails
# **IQR limits:**
#
# - right tail: 75th quantile + 3* IQR
# - left tail:  25th quantile - 3* IQR
#
# where IQR is the inter-quartile range: 75th quantile - 25th quantile.


# capping at both tails using iqr capping method
winsor = Winsorizer(capping_method='iqr', tail='both',
                    variables=['age', 'fare'])

winsor.fit(X_train)


winsor.left_tail_caps_


winsor.right_tail_caps_


# transforming the training and testing data

train_t = winsor.transform(X_train)
test_t = winsor.transform(X_test)

print("Max fare:", train_t.fare.max())
print("Min fare", train_t.fare.min())


# ### percentiles or quantiles:
#
# - right tail: 98th percentile
# - left tail:  2nd percentile


# capping at both tails using quantiles capping method
winsor = Winsorizer(capping_method='quantiles', tail='both',
                    fold=0.02, variables=['age', 'fare'])

winsor.fit(X_train)


print("Minimum caps :", winsor.left_tail_caps_)

print("Maximum caps :", winsor.right_tail_caps_)


# transforming the training and testing data
train_t = winsor.transform(X_train)
test_t = winsor.transform(X_test)

print("Max age:", train_t.age.max())
print("Min age", train_t.age.min())


# Histogram of age feature after capping outliers
plot_hist(train_t, 'age')

