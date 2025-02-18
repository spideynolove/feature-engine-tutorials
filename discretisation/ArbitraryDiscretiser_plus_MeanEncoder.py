# # ArbitraryDiscretiser + MeanEncoder
# This is very useful for linear models, because by using discretisation + a monotonic encoding, we create monotonic variables with the target, from those that before were not originally. And this tends to help improve the performance of the linear model. 

# ## ArbitraryDiscretiser
# The ArbitraryDiscretiser() divides continuous numerical variables into contiguous intervals arbitrarily defined by the user.
# The user needs to enter a dictionary with variable names as keys, and a list of the limits of the intervals as values. For example {'var1': [0, 10, 100, 1000],'var2': [5, 10, 15, 20]}.
# <b>Note:</b> Check out the ArbitraryDiscretiser notebook to learn more about this transformer.

# ## MeanEncoder
# The MeanEncoder() replaces the labels of the variables by the mean value of the target for that label. <br>For example, in the variable colour, if the mean value of the binary target is 0.5 for the label blue, then blue is replaced by 0.5
# <b>Note:</b> Read MeanEncoder notebook to know more about this transformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.encoding import MeanEncoder

plt.rcParams["figure.figsize"] = [15,5]

# Load titanic dataset from OpenML
def load_titanic():
    # data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = pd.read_csv('titanic.csv')
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    # data.drop(labels=['boat', 'body', 'home.dest', 'name', 'ticket'], axis=1, inplace=True)
    return data


data = load_titanic()

# let's separate into training and testing set
X = data.drop(['survived'], axis=1)
y = data.survived

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# we will transform two continuous variables
X_train[["age", 'fare']].hist(bins=30)
plt.show()


# set up the discretiser
arb_disc = ArbitraryDiscretiser(
    binning_dict={'age': [0, 18, 30, 50, 100],
                  'fare': [-1, 20, 40, 60, 80, 600]},
    # returns values as categorical
    return_object=True)

# set up the mean encoder
mean_enc = MeanEncoder(variables=['age', 'fare'])

# set up the pipeline
transformer = Pipeline(steps=[('ArbitraryDiscretiser', arb_disc),
                              ('MeanEncoder', mean_enc),
                              ])
# train the pipeline
transformer.fit(X_train, y_train)

transformer.named_steps['ArbitraryDiscretiser'].binner_dict_

transformer.named_steps['MeanEncoder'].encoder_dict_

train_t = transformer.transform(X_train)
test_t = transformer.transform(X_test)

# let's explore the monotonic relationship
plt.figure(figsize=(7, 5))
pd.concat([test_t, y_test], axis=1).groupby("fare")["survived"].mean().plot()
plt.title("Relationship between fare and target")
plt.xlabel("fare")
plt.ylabel("Mean of target")
plt.show()

# We can observe an almost linear relationship between the variable "fare" after the transformation and the target.