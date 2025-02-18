# Generated from: CountFrequencyEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # CountFrequencyEncoder
# <p>The CountFrequencyEncoder() replaces categories by the count of
# observations per category or by the percentage of observations per category.<br>
# For example in the variable colour, if 10 observations are blue, blue will
# be replaced by 10. Alternatively, if 10% of the observations are blue, blue
# will be replaced by 0.1.</p>


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import CountFrequencyEncoder


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


# The CountFrequencyEncoder(), replaces the categories by the count or frequency of the observations in the train set for that category. 
#
# If we select "count" in the encoding_method, then for the variable colour, if there are 10 observations in the train set that show colour blue, blue will be replaced by 10.<br><br> Alternatively, if we select "frequency" in the encoding_method, if 10% of the observations in the train set show blue colour, then blue will be replaced by 0.1.


# ### Frequency
#
# Labels are replaced by the percentage of the observations that show that label in the train set.


'''
Parameters
----------

encoding_method : str, default='count' 
                Desired method of encoding.

        'count': number of observations per category
        
        'frequency': percentage of observations per category

variables : list
          The list of categorical variables that will be encoded. If None, the 
          encoder will find and transform all object type variables.
'''
count_encoder = CountFrequencyEncoder(encoding_method='frequency',
                                      variables=['cabin', 'pclass', 'embarked'])

count_encoder.fit(X_train)


# we can explore the encoder_dict_ to find out the category replacements.
count_encoder.encoder_dict_


# transform the data: see the change in the head view
train_t = count_encoder.transform(X_train)
test_t = count_encoder.transform(X_test)
test_t.head()


test_t['pclass'].value_counts().plot.bar()
plt.show()


test_orig = count_encoder.inverse_transform(test_t)
test_orig.head()


# ### Count
#
# Labels are replaced by the number of the observations that show that label in the train set.


# this time we encode only 1 variable

count_enc = CountFrequencyEncoder(encoding_method='count',
                                                variables='cabin')

count_enc.fit(X_train)


# we can find the mappings in the encoder_dict_ attribute.

count_enc.encoder_dict_


# transform the data: see the change in the head view for Cabin

train_t = count_enc.transform(X_train)
test_t = count_enc.transform(X_test)

test_t.head()


# ### Select categorical variables automatically
#
# If we don't indicate which variables we want to encode, the encoder will find all categorical variables


# this time we ommit the argument for variable
count_enc = CountFrequencyEncoder(encoding_method = 'count')

count_enc.fit(X_train)


# we can see that the encoder selected automatically all the categorical variables

count_enc.variables


# transform the data: see the change in the head view

train_t = count_enc.transform(X_train)
test_t = count_enc.transform(X_test)

test_t.head()


# ### Note
# if there are labels in the test set that were not present in the train set, the transformer will introduce NaN, and raise a warning.

