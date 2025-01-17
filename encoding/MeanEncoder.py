# Generated from: MeanEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # MeanEncoder
#
# The MeanEncoder() replaces the labels of the variables by the mean value of the target for that label. <br>For example, in the variable colour, if the mean value of the binary target is 0.5 for the label blue, then blue is replaced by 0.5


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder


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


# The MeanEncoder() replaces categories by the mean value of the
# target for each category.<br><br>
# For example in the variable colour, if the mean of the target for blue, red
# and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
# and grey by 0.1.<br><br>
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed as 
# argument, the encoder will find and encode all categorical variables
# (object type).


# we will transform 3 variables
'''
Parameters
----------  
variables : list, default=None
    The list of categorical variables that will be encoded. If None, the 
    encoder will find and select all object type variables.
'''

mean_enc = MeanEncoder(variables=['cabin', 'pclass', 'embarked'])

# Note: the MeanCategoricalEncoder needs the target to fit
mean_enc.fit(X_train, y_train)


# see the dictionary with the mappings per variable

mean_enc.encoder_dict_


# we can see the transformed variables in the head view

train_t = mean_enc.transform(X_train)
test_t = mean_enc.transform(X_test)

test_t.head()


''' The MeanEncoder has the characteristic that return monotonic
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


mean_enc = MeanEncoder()

mean_enc.fit(X_train, y_train)


mean_enc.variables


# we can see the transformed variables in the head view

train_t = mean_enc.transform(X_train)
test_t = mean_enc.transform(X_test)

test_t.head()

