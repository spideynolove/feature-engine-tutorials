# Generated from: OutlierTrimmer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # OutlierTrimmer
# The OutlierTrimmer() removes observations with outliers from the dataset.
#
# It works only with numerical variables. A list of variables can be indicated.
# Alternatively, the OutlierTrimmer() will select all numerical variables.
#
# The OutlierTrimmer() first calculates the maximum and /or minimum values
# beyond which a value will be considered an outlier, and thus removed.
#
# Limits are determined using:
#
# - a Gaussian approximation
# - the inter-quantile range proximity rule
# - percentiles.
#
# ### Example:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.outliers import OutlierTrimmer


# Load titanic dataset from OpenML

def load_titanic():
    data = pd.read_csv(
        'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
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

print("train data shape before removing outliers:", X_train.shape)
print("test data shape before removing outliers:", X_test.shape)


# let's find out the maximum Age and maximum Fare in the titanic

print("Max age:", data.age.max())
print("Max fare:", data.fare.max())

print("Min age:", data.age.min())
print("Min fare:", data.fare.min())


# Histogram of age feature before capping outliers
plot_hist(data, 'age')


# Histogram of fare feature before capping outliers
plot_hist(data, 'fare')


# ### Outlier trimming using Gaussian limits:
# The transformer will find the maximum and / or minimum values to
#     trim the variables using the Gaussian approximation.
#
#
# - right tail: mean + 3* std
# - left tail: mean - 3* std


'''Parameters
----------

capping_method : str, default=gaussian
    Desired capping method. Can take 'gaussian', 'iqr' or 'quantiles'.
    
tail : str, default=right
    Whether to cap outliers on the right, left or both tails of the distribution.
    Can take 'left', 'right' or 'both'.

fold: int or float, default=3
    How far out to to place the capping values. The number that will multiply
    the std or IQR to calculate the capping values.

variables : list, default=None

missing_values: string, default='raise'
    Indicates if missing values should be ignored or raised.'''

# removing outliers based on right tail of age and fare columns using gaussian capping method
trimmer = OutlierTrimmer(
    capping_method='gaussian', tail='right', fold=3, variables=['age', 'fare'])

# fitting trimmer object to training data
trimmer.fit(X_train)


# here we can find the maximum caps allowed
trimmer.right_tail_caps_


# this dictionary is empty, because we selected only right tail
trimmer.left_tail_caps_


# transforming the training and testing data
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)

# let's check the new maximum Age and maximum Fare in the titanic
print("Max age:", train_t.age.max())
print("Max fare:", train_t.fare.max())


print("train data shape after removing outliers:", train_t.shape)
print(f"{X_train.shape[0] - train_t.shape[0]} observations are removed\n")

print("test data shape after removing outliers:", test_t.shape)
print(f"{X_test.shape[0] - test_t.shape[0]} observations are removed")


# ### Gaussian approximation trimming, both tails


# Trimming the outliers at both tails using gaussian  method
trimmer = OutlierTrimmer(
    capping_method='gaussian', tail='both', fold=2, variables=['fare', 'age'])
trimmer.fit(X_train)


print("Minimum caps :", trimmer.left_tail_caps_)

print("Maximum caps :", trimmer.right_tail_caps_)


# transforming the training and testing data
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)

print("train data shape after removing outliers:", train_t.shape)
print(f"{X_train.shape[0] - train_t.shape[0]} observations are removed\n")

print("test data shape after removing outliers:", test_t.shape)
print(f"{X_test.shape[0] - test_t.shape[0]} observations are removed")


# ### Inter Quartile Range, both tails
# The transformer will find the boundaries using the IQR proximity rule.
# **IQR limits:**
#
# - right tail: 75th quantile + 3* IQR
# - left tail:  25th quantile - 3* IQR
#
# where IQR is the inter-quartile range: 75th quantile - 25th quantile.


# trimming at both tails using iqr capping method
trimmer = OutlierTrimmer(
    capping_method='iqr', tail='both', variables=['age', 'fare'])

trimmer.fit(X_train)


print("Minimum caps :", trimmer.left_tail_caps_)

print("Maximum caps :", trimmer.right_tail_caps_)


# transforming the training and testing data
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)

print("train data shape after removing outliers:", train_t.shape)
print(f"{X_train.shape[0] - train_t.shape[0]} observations are removed\n")

print("test data shape after removing outliers:", test_t.shape)
print(f"{X_test.shape[0] - test_t.shape[0]} observations are removed")


# ### percentiles or quantiles:
# The limits are given by the percentiles.
# - right tail: 98th percentile
# - left tail:  2nd percentile


# trimming at both tails using quantiles capping method
trimmer = OutlierTrimmer(capping_method='quantiles',
                         tail='both', fold=0.02, variables=['age', 'fare'])

trimmer.fit(X_train)


print("Minimum caps :", trimmer.left_tail_caps_)

print("Maximum caps :", trimmer.right_tail_caps_)


# transforming the training and testing data
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)

print("train data shape after removing outliers:", train_t.shape)
print(f"{X_train.shape[0] - train_t.shape[0]} observations are removed\n")

print("test data shape after removing outliers:", test_t.shape)
print(f"{X_test.shape[0] - test_t.shape[0]} observations are removed")


# Histogram of age feature after removing outliers
plot_hist(train_t, 'age')


# Histogram of fare feature after removing outliers
plot_hist(train_t, 'fare')

