# Generated from: DecisionTreeEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # DecisionTreeEncoder
#
# The DecisionTreeEncoder() encodes categorical variables with predictions of a decision tree model.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import DecisionTreeEncoder


# Load titanic dataset from OpenML

def load_titanic(filepath='titanic.csv'):
    # data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    # data.drop(labels=['boat', 'body', 'home.dest', 'name', 'ticket'], axis=1, inplace=True)
    return data


# data = load_titanic("../data/titanic.csv")
data = load_titanic("../data/titanic-2/Titanic-Dataset.csv")
data.head()


X = data.drop(['survived', 'name', 'ticket'], axis=1)
y = data.survived


# we will encode the below variables, they have no missing values
X[['cabin', 'pclass', 'embarked']].isnull().sum()


''' Make sure that the variables are type (object).
if not, cast it as object , otherwise the transformer will either send an error (if we pass it as argument) 
or not pick it up (if we leave variables=None). '''

X[['cabin', 'pclass', 'embarked']].dtypes


# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# The categorical variable will be first encoded into integers with the
# OrdinalEncoder(). The integers can be assigned arbitrarily to the
# categories or following the mean value of the target in each category.
#
# Then a decision tree will be fit using the resulting numerical variable to predict
# the target  variable. Finally, the original categorical variable values will be
# replaced by the predictions of the decision tree.


'''
Parameters
    ----------

    encoding_method: str, default='arbitrary'
        The categorical encoding method that will be used to encode the original
        categories to numerical values.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    regression : boolean, default=True
        Indicates whether the encoder should train a regression or a classification
        decision tree.

    param_grid : dictionary, default=None
        The list of parameters over which the decision tree should be optimised
        during the grid search. The param_grid can contain any of the permitted
        parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().

        If None, then param_grid = {'max_depth': [1, 2, 3, 4]}.

    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.
'''


tree_enc = DecisionTreeEncoder(encoding_method='arbitrary',
                               cv=3,
                               scoring = 'roc_auc',
                               param_grid = {'max_depth': [1, 2, 3, 4]},
                               regression = False,
                               variables=['cabin', 'pclass', 'embarked']
                              )

tree_enc.fit(X_train,y_train) # to fit you need to pass the target y


tree_enc.encoder_


# transform and visualise the data

train_t = tree_enc.transform(X_train)
test_t = tree_enc.transform(X_test)

test_t.sample(5)


# ### Automatically select the variables
#
# This encoder will select all categorical variables to encode, when no variables are specified when calling the encoder.


tree_enc = DecisionTreeEncoder(encoding_method='arbitrary',
                               cv=3,
                               scoring = 'roc_auc',
                               param_grid = {'max_depth': [1, 2, 3, 4]},
                               regression = False,
                              )

tree_enc.fit(X_train,y_train) # to fit you need to pass the target y


tree_enc.encoder_


# transform and visualise the data

train_t = tree_enc.transform(X_train)
test_t = tree_enc.transform(X_test)

test_t.sample(5)

