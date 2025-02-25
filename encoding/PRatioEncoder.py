# Generated from: PRatioEncoder.ipynb

# # PRatioEncoder
#
# The PRatioEncoder() replaces categories by the ratio of the probability of the
# target = 1 and the probability of the target = 0.<br>
#
# The target probability ratio is given by: p(1) / p(0).
#
# The log of the target probability ratio is: np.log( p(1) / p(0) )
# #### It only works for binary classification.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import PRatioEncoder

from feature_engine.encoding import RareLabelEncoder #to reduce cardinality


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


## Rare value encoder first to reduce the cardinality
# see RareLabelEncoder jupyter notebook for more details on this encoder
rare_encoder = RareLabelEncoder(tol=0.03,
                                n_categories=2, 
                                variables=['cabin', 'pclass', 'embarked'])

rare_encoder.fit(X_train)

# transform
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)


# The PRatioEncoder() replaces categories by the ratio of the probability of the
# target = 1 and the probability of the target = 0.
#
# The target probability ratio is given by: p(1) / p(0)
#
# The log of the target probability ratio is: np.log( p(1) / p(0) )
#
# Note: This categorical encoding is exclusive for binary classification.
#
# For example in the variable colour, if the mean of the target = 1 for blue
# is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
# 0.8 / 0.2 = 4 if ratio is selected, or log(0.8/0.2) = 1.386 if log_ratio
# is selected.
#
# Note: the division by 0 is not defined and the log(0) is not defined.
# Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
# log_ratio, in any of the variables, the encoder will return an error.
#
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed as
# argument, the encoder will find and encode all categorical variables
# (object type).


# ### Ratio


'''
Parameters
----------

encoding_method : str, default=woe
    Desired method of encoding.

    'ratio' : probability ratio

    'log_ratio' : log probability ratio

variables : list, default=None
    The list of categorical variables that will be encoded. If None, the
    encoder will find and select all object type variables.
'''
Ratio_enc = PRatioEncoder(encoding_method='ratio',
                           variables=['cabin', 'pclass', 'embarked'])

# to fit you need to pass the target y
Ratio_enc.fit(train_t, y_train)


Ratio_enc.encoder_dict_


# transform and visualise the data

train_t = Ratio_enc.transform(train_t)
test_t = Ratio_enc.transform(test_t)

test_t.sample(5)


# ### log ratio


train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)

logRatio_enc = PRatioEncoder(encoding_method='log_ratio',
                           variables=['cabin', 'pclass', 'embarked'])

# to fit you need to pass the target y
logRatio_enc.fit(train_t, y_train)


logRatio_enc.encoder_dict_


# transform and visualise the data

train_t = logRatio_enc.transform(train_t)
test_t = logRatio_enc.transform(test_t)

test_t.sample(5)


''' The PRatioEncoder(encoding_method='ratio' or 'log_ratio') has the characteristic that return monotonic
 variables, that is, encoded variables which values increase as the target increases'''

# let's explore the monotonic relationship
plt.figure(figsize=(7,5))
pd.concat([test_t,y_test], axis=1).groupby("pclass")["survived"].mean().plot()
#plt.xticks([0,1,2])
plt.yticks(np.arange(0,1.1,0.1))
plt.title("Relationship between pclass and target")
plt.xlabel("Pclass")
plt.ylabel("Mean of target")
plt.show()


# ### Automatically select the variables
#
# This encoder will select all categorical variables to encode, when no variables are specified when calling the encoder.


train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)

logRatio_enc = PRatioEncoder(encoding_method='log_ratio')

# to fit you need to pass the target y
logRatio_enc.fit(train_t, y_train)


# transform and visualise the data

train_t = logRatio_enc.transform(train_t)
test_t = logRatio_enc.transform(test_t)

test_t.sample(5)

