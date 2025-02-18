# Generated from: GeometricWidthDiscretiser_plus_MeanEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # GeometricWidthDiscretiser + MeanEncoder
#
# This is very useful for linear models, because by using discretisation + a monotonic encoding, we create monotonic variables with the target, from those that before were not originally. And this tends to help improve the performance of the linear model. 


# ## GeometricWidthDiscretiser
#
# The GeometricWidthDiscretiser() divides continuous numerical variables into
# intervals of increasing width with equal increments. Note that the
# proportion of observations per interval may vary.
#
# The size of the interval will follow geometric progression.


# ## MeanEncoder
#
# This encoder replaces the labels by the target mean.
#
# <b>Note:</b> Check out the MeanEncoder notebook to learn more about this transformer.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import GeometricWidthDiscretiser
from feature_engine.encoding import MeanEncoder

plt.rcParams["figure.figsize"] = [15,5]


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


# let's separate into training and testing set
X = data.drop(['survived'], axis=1)
y = data.survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("X_train :" ,X_train.shape)
print("X_test :" ,X_test.shape)


# we will use two continuous variables for the transformations
X_train[["age", 'fare']].hist(bins=30)
plt.show()


# set up the discretiser

efd = GeometricWidthDiscretiser(
    bins=5,
    variables=['age', 'fare'],
    # important: return values as categorical
    return_object=True)

# set up the encoder
woe = MeanEncoder(variables=['age', 'fare'])

# pipeline
transformer = Pipeline(
    steps=[
        ('GeometricWidthDiscretiser', efd),
        ('MeanEncoder', woe),
    ]
)

transformer.fit(X_train, y_train)


transformer.named_steps['GeometricWidthDiscretiser'].binner_dict_


transformer.named_steps['MeanEncoder'].encoder_dict_


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

