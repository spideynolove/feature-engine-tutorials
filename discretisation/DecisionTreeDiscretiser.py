# Generated from: DecisionTreeDiscretiser.ipynb

# # DecisionTreeDiscretiser
#
# The DecisionTreeDiscretiser() divides continuous numerical variables into discrete, finite, values estimated by a decision tree.
#
# The methods is inspired by the following article from the winners of the KDD 2009 competition:
# http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf
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

from feature_engine.discretisation import DecisionTreeDiscretiser

plt.rcParams["figure.figsize"] = [15,5]


# Read the separate files
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')

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


# The DecisionTreeDiscretiser() works only with numerical variables.
# A list of variables can be passed as an argument. Alternatively, the
# discretiser will automatically select and transform all numerical variables.
#
# The DecisionTreeDiscretiser() first trains a decision tree for each variable,
# fit.
#
# The DecisionTreeDiscretiser() then transforms the variables, that is,
# makes predictions based on the variable values, using the trained decision
# tree, transform.


'''
Parameters
----------

cv : int, default=3
    Desired number of cross-validation fold to be used to fit the decision
    tree.

scoring: str, default='neg_mean_squared_error'
    Desired metric to optimise the performance for the tree. Comes from
    sklearn metrics. See DecisionTreeRegressor or DecisionTreeClassifier
    model evaluation documentation for more options:
    https://scikit-learn.org/stable/modules/model_evaluation.html

variables : list
    The list of numerical variables that will be transformed. If None, the
    discretiser will automatically select all numerical type variables.

regression : boolean, default=True
    Indicates whether the discretiser should train a regression or a classification
    decision tree.

param_grid : dictionary, default=None
    The list of parameters over which the decision tree should be optimised
    during the grid search. The param_grid can contain any of the permitted
    parameters for Scikit-learn's DecisionTreeRegressor() or
    DecisionTreeClassifier().

    If None, then param_grid = {'max_depth': [1, 2, 3, 4]}

random_state : int, default=None
    The random_state to initialise the training of the decision tree. It is one
    of the parameters of the Scikit-learn's DecisionTreeRegressor() or
    DecisionTreeClassifier(). For reproducibility it is recommended to set
    the random_state to an integer.
'''

treeDisc = DecisionTreeDiscretiser(cv=3,
                                   scoring='neg_mean_squared_error',
                                   variables=['LotArea', 'GrLivArea'],
                                   regression=True,
                                   random_state=29)

# the DecisionTreeDiscretiser needs the target for fitting
treeDisc.fit(X_train, y_train)


# the binner_dict_ contains the best decision tree for each variable
treeDisc.binner_dict_


train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)


# the below account for the best obtained bins, aka, the tree predictions

train_t['GrLivArea'].unique()


# the below account for the best obtained bins, aka, the tree predictions

train_t['LotArea'].unique()


# here I put side by side the original variable and the transformed variable

tmp = pd.concat([X_train[["LotArea", 'GrLivArea']],
                 train_t[["LotArea", 'GrLivArea']]], axis=1)

tmp.columns = ["LotArea", 'GrLivArea', "LotArea_binned", 'GrLivArea_binned']

tmp.head()


# in  equal frequency discretisation, we obtain the same amount of observations
# in each one of the bins.

plt.subplot(1,2,1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')

plt.subplot(1,2,2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.ylabel('Number of houses')

plt.show()


# ## DecisionTreeDiscretiser with binary classification


# Load titanic dataset from file

def load_titanic(filepath='titanic.csv'):
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    return data


data = load_titanic("../data/titanic-2/Titanic-Dataset.csv")
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data.drop(['survived'], axis=1),
                                                    data['survived'],
                                                    test_size=0.3, 
                                                    random_state=0)

print(X_train.shape)
print(X_test.shape)


#this discretiser transforms the numerical variables
X_train[['fare', 'age']].dtypes


treeDisc = DecisionTreeDiscretiser(cv=3,
                                   scoring='roc_auc',
                                   variables=['fare', 'age'],
                                   regression=False,
                                   param_grid={'max_depth': [1, 2]},
                                   random_state=29,
                                   )

treeDisc.fit(X_train, y_train)


treeDisc.binner_dict_


train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)


# the below account for the best obtained bins
# in this case, the tree has found that dividing the data in 6 bins is enough
train_t['age'].unique()


# the below account for the best obtained bins
# in this case, the tree has found that dividing the data in 8 bins is enough
train_t['fare'].unique()


# here I put side by side the original variable and the transformed variable

tmp = pd.concat([X_train[["fare", 'age']], train_t[["fare", 'age']]], axis=1)

tmp.columns = ["fare", 'age', "fare_binned", 'age_binned']

tmp.head()


plt.subplot(1,2,1)
tmp.groupby('fare_binned')['fare'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')

plt.subplot(1,2,2)
tmp.groupby('age_binned')['age'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')

plt.show()


# The DecisionTreeDiscretiser() returns values which show
# a monotonic relationship with target

pd.concat([test_t, y_test], axis=1).groupby(
    'age')['survived'].mean().plot(figsize=(6, 4))

plt.ylabel("Mean of target")
plt.title("Relationship between fare and target")
plt.show()


# The DecisionTreeDiscretiser() returns values which show
# a monotonic relationship with target

pd.concat([test_t, y_test], axis=1).groupby(
    'fare')['survived'].mean().plot(figsize=(6, 4))

plt.ylabel("Mean of target")
plt.title("Relationship between fare and target")
plt.show()


# ## DecisionTreeDiscretiser for Multi-class classification


# Load iris dataset from sklearn
from sklearn.datasets import load_iris

data = pd.DataFrame(load_iris().data, 
                    columns=load_iris().feature_names).join(
    pd.Series(load_iris().target, name='type'))

data.head()


data.type.unique() # 3 - class classification


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data.drop('type', axis=1),
                                                    data['type'],
                                                    test_size=0.3,
                                                    random_state=0)

print(X_train.shape)
print(X_test.shape)


#selected two numerical variables
X_train[['sepal length (cm)', 'sepal width (cm)']].dtypes


treeDisc = DecisionTreeDiscretiser(cv=3,
                                   scoring='accuracy',
                                   variables=[
                                       'sepal length (cm)', 'sepal width (cm)'],
                                   regression=False,
                                   random_state=29,
                                   )

treeDisc.fit(X_train, y_train)


treeDisc.binner_dict_


train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)


# here I put side by side the original variable and the transformed variable
tmp = pd.concat([X_train[['sepal length (cm)', 'sepal width (cm)']],
                 train_t[['sepal length (cm)', 'sepal width (cm)']]], axis=1)

tmp.columns = ['sepal length (cm)', 'sepal width (cm)',
               'sepalLen_binned', 'sepalWid_binned']

tmp.head()


plt.subplot(1, 2, 1)
tmp.groupby('sepalLen_binned')['sepal length (cm)'].count().plot.bar()
plt.ylabel('Number of species')
plt.title('Number of observations per discrete value')

plt.subplot(1, 2, 2)
tmp.groupby('sepalWid_binned')['sepal width (cm)'].count().plot.bar()
plt.ylabel('Number of species')
plt.title('Number of observations per discrete value')

plt.show()

