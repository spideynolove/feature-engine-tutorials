# Generated from: EqualWidthDiscretiser_plus_OrdinalEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # EqualWidthDiscretiser + OrdinalEncoder
#
#
# This is very useful for linear models, because by using discretisation + a monotonic encoding, we create monotonic variables with the target, from those that before were not originally. And this tends to help improve the performance of the linear model. 


# ## EqualWidthDiscretiser
#
# The EqualWidthDiscretiser() divides continuous numerical variables into
# intervals of the same width, that is, equidistant intervals. Note that the
# proportion of observations per interval may vary.
#
# The number of intervals
# in which the variable should be divided must be indicated by the user.
#
# <b>Note:</b> Check out the EqualWidthDiscretiser notebook to learn more about this transformer.


# ## OrdinalEncoder
# The OrdinalEncoder() will replace the variable labels by digits, from 1 to the number of different labels. 
#
# If we select "arbitrary", then the encoder will assign numbers as the labels appear in the variable (first come first served).
#
# If we select "ordered", the encoder will assign numbers following the mean of the target value for that label. So labels for which the mean of the target is higher will get the number 1, and those where the mean of the target is smallest will get the number n.
#
# <b>Note:</b> Check out the OrdinalEncoder notebook to know more about this transformer.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.encoding import OrdinalEncoder

plt.rcParams["figure.figsize"] = [15,5]


# Load titanic dataset from OpenML

def load_titanic():
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    data.drop(labels=['boat', 'body', 'home.dest', 'name', 'ticket'], axis=1, inplace=True)
    return data


data = load_titanic()
data.head()


# let's separate into training and testing set
X = data.drop(['survived'], axis=1)
y = data.survived

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# we will use two continuous variables for the transformations

X_train[["age", 'fare']].hist(bins=30)
plt.show()


# set up the discretiser
ewd = EqualWidthDiscretiser(
    bins=5,
    variables=['age', 'fare'],
    # important: return values as categorical
    return_object=True)

# set up the encoder
oe = OrdinalEncoder(variables=['age', 'fare'])

# pipeline
transformer = Pipeline(steps=[('EqualWidthDiscretiser', ewd),
                              ('OrdinalEncoder', oe),
                              ])

transformer.fit(X_train, y_train)


transformer.named_steps['EqualWidthDiscretiser'].binner_dict_


transformer.named_steps['OrdinalEncoder'].encoder_dict_


train_t = transformer.transform(X_train)
test_t = transformer.transform(X_test)

test_t.head()


# let's explore the monotonic relationship
plt.figure(figsize=(7,5))
pd.concat([test_t,y_test], axis=1).groupby("fare")["survived"].mean().plot()
plt.title("Relationship between fare and target")
plt.xlabel("fare")
plt.ylabel("Mean of target")
plt.show()


# Note how the bins are monotonically ordered with the target.

