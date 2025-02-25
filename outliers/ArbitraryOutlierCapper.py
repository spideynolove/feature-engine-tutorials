# Generated from: ArbitraryOutlierCapper.ipynb

# # ArbitraryOutlierCapper
# The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
# at an arbitrary value indicated by the user.
#
# The user must provide the maximum or minimum values that will be used <br>
# to cap each variable in a dictionary {feature : capping_value}


# ### Example


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.outliers import ArbitraryOutlierCapper


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


data = load_titanic()
data.sample(5)


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data.drop('survived', axis=1),
                                                    data['survived'],
                                                    test_size=0.3,
                                                    random_state=0)

print("train data:", X_train.shape)
print("test data:", X_test.shape)


# Histogram of age feature before capping outliers

plot_hist(data, 'age')


# Histogram of fare feature before capping outliers

plot_hist(data, 'fare')


# let's find out the maximum&minimum Age and maximum Fare in the titanic
print("Max age:", data.age.max())
print("Max fare:", data.fare.max())

print("Min age:", data.age.min())
print("Min fare:", data.fare.min())


# ### Maximum capping


'''Parameters
----------
max_capping_dict : dictionary, default=None
    Dictionary containing the user specified capping values for the right tail of
    the distribution of each variable (maximum values).

min_capping_dict : dictionary, default=None
    Dictionary containing user specified capping values for the eft tail of the
    distribution of each variable (minimum values).

missing_values : string, default='raise'
    Indicates if missing values should be ignored or raised. If
    `missing_values='raise'` the transformer will return an error if the
    training or the datasets to transform contain missing values.
'''

# capping of age and fare features at right tail
capper = ArbitraryOutlierCapper(
    max_capping_dict={'age': 50, 'fare': 150}, min_capping_dict=None)

capper.fit(X_train)


# here we can find the maximum caps allowed
print("Maximum caps:", capper.right_tail_caps_)


# this dictionary is empty, because we selected only right tail
capper.left_tail_caps_


# transforming train and test data
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

#check max age and max fare after capping
print("Max age after capping:", train_t.age.max())
print("Max fare after capping:", train_t.fare.max())


# ### Minimum capping


# capping outliers at left tail
capper = ArbitraryOutlierCapper(
    max_capping_dict=None, min_capping_dict={'age': 10, 'fare': 100})

capper.fit(X_train)


# this dictionary is empty, because we selected only right tail
capper.right_tail_caps_


# here we can find the minimum caps allowed
capper.left_tail_caps_


# transforming train and test set
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

# After capping
print("Min age:", train_t.age.min())
print("Min fare:", train_t.fare.min())


# ### Both ends capping


# capping outliers at both tails
capper = ArbitraryOutlierCapper(
    min_capping_dict={'age': 5, 'fare': 5},
    max_capping_dict={'age': 60, 'fare': 150})
capper.fit(X_train)


# here we can find the maximum caps allowed
capper.right_tail_caps_


# here we can find the minimum caps allowed
capper.left_tail_caps_


# transforming train and test data
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

# After capping outliers
print("Max age:", train_t.age.max())
print("Max fare:", train_t.fare.max())

print("Min age:", train_t.age.min())
print("Min fare:", train_t.fare.min())


# Histogram of age feature after capping outliers
plot_hist(train_t, 'age')


# Histogram of fare feature after capping outliers
plot_hist(train_t, 'fare')

