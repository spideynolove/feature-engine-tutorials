# Generated from: WoEEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# ## WoEEncoder (weight of evidence)
#
# This encoder replaces the labels by the weight of evidence 
# #### It only works for binary classification.
#
# The weight of evidence is given by: log( p(1) / p(0) )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import WoEEncoder

from feature_engine.encoding import RareLabelEncoder #to reduce cardinality


# Load titanic dataset from OpenML

def load_titanic():
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float')
    data['fare'] = data['fare'].astype('float')
    data['embarked'].fillna('C', inplace=True)
    data.drop(labels=['boat', 'body', 'home.dest'], axis=1, inplace=True)
    return data


data = load_titanic()
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


# The WoERatioEncoder() replaces categories by the weight of evidence
# or by the ratio between the probability of the target = 1 and the probability
# of the  target = 0.
#
# The weight of evidence is given by: log(P(X=x<sub>j</sub>|Y = 1)/P(X=x<sub>j</sub>|Y=0))
#
#
# Note: This categorical encoding is exclusive for binary classification.
#
# For example in the variable colour, if the mean of the target = 1 for blue
# is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
# np.log(0.8/0.2) = 1.386
# #### Note: 
# The division by 0 is not defined and the log(0) is not defined.
# Thus, if p(0) = 0 or p(1) = 0 for
# woe , in any of the variables, the encoder will return an error.
#
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed as 
# argument, the encoder will find and encode all categorical variables
# (object type).<br>
#
# For details on the calculation of the weight of evidence visit:<br>
# https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html


# ### Weight of evidence


woe_enc = WoEEncoder(variables=['cabin', 'pclass', 'embarked'])

# to fit you need to pass the target y
woe_enc.fit(train_t, y_train)


woe_enc.encoder_dict_


# transform and visualise the data

train_t = woe_enc.transform(train_t)
test_t = woe_enc.transform(test_t)

test_t.sample(5)


''' The WoEEncoder has the characteristic that return monotonic
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


ratio_enc = WoEEncoder()

# to fit we need to pass the target y
ratio_enc.fit(train_t, y_train)


# transform and visualise the data

train_t = ratio_enc.transform(train_t)
test_t = ratio_enc.transform(test_t)

test_t.head()

