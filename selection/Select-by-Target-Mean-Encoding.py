# Generated from: Select-by-Target-Mean-Encoding.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# ## Select with Target Mean as Performance Proxy
#
# **Method used in a KDD 2009 competition**
#
# This feature selection approach was used by data scientists at the University of Melbourne in the [KDD 2009](http://www.kdd.org/kdd-cup/view/kdd-cup-2009) data science competition. The task consisted in predicting churn based on a dataset with a huge number of features.
#
# The authors describe this procedure as an aggressive non-parametric feature selection procedure that is based in contemplating the relationship between the feature and the target.
#
#
# **The procedure consists in the following steps**:
#
# For each categorical variable:
#
#     1) Separate into train and test
#
#     2) Determine the mean value of the target within each label of the categorical variable using the train set
#
#     3) Use that mean target value per label as the prediction (using the test set) and calculate the roc-auc.
#
# For each numerical variable:
#
#     1) Separate into train and test
#
#     2) Divide the variable intervals
#
#     3) Calculate the mean target within each interval using the training set 
#
#     4) Use that mean target value / bin as the prediction (using the test set) and calculate the roc-auc
#
#
# The authors quote the following advantages of the method:
#
# - Speed: computing mean and quantiles is direct and efficient
# - Stability respect to scale: extreme values for continuous variables do not skew the predictions
# - Comparable between categorical and numerical variables
# - Accommodation of non-linearities
#
# **Important**
# The authors here use the roc-auc, but in principle, we could use any metric, including those valid for regression.
#
# The authors sort continuous variables into percentiles, but Feature-engine gives the option to sort into equal-frequency or equal-width intervals.
#
# **Reference**:
# [Predicting customer behaviour: The University of Melbourne's KDD Cup Report. Miller et al. JMLR Workshop and Conference Proceedings 7:45-55](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf)


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from feature_engine.selection import SelectByTargetMeanPerformance


# load the titanic dataset
data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

# remove unwanted variables
data.drop(labels = ['name','boat', 'ticket','body', 'home.dest'], axis=1, inplace=True)

# replace ? by Nan
data = data.replace('?', np.nan)

# missing values
data.dropna(subset=['embarked', 'fare'], inplace=True)

data['age'] = data['age'].astype('float')
data['age'] = data['age'].fillna(data['age'].mean())

data['fare'] = data['fare'].astype('float')

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return 'N' 
    
data['cabin'] = data['cabin'].apply(get_first_cabin)


data.head()


# Variable preprocessing:

# then I will narrow down the different cabins by selecting only the
# first letter, which represents the deck in which the cabin was located

# captures first letter of string (the letter of the cabin)
data['cabin'] = data['cabin'].str[0]

# now we will rename those cabin letters that appear only 1 or 2 in the
# dataset by N

# replace rare cabins by N
data['cabin'] = np.where(data['cabin'].isin(['T', 'G']), 'N', data['cabin'])

data['cabin'].unique()


data.dtypes


# number of passengers per value
data['parch'].value_counts()


# cap variable at 3, the rest of the values are
# shown by too few observations

data['parch'] = np.where(data['parch']>3,3,data['parch'])


data['sibsp'].value_counts()


# cap variable at 3, the rest of the values are
# shown by too few observations

data['sibsp'] = np.where(data['sibsp']>3,3,data['sibsp'])


# cast discrete variables as categorical

# feature-engine considers categorical variables all those of type
# object. So in order to work with numerical variables as if they
# were categorical, we  need to cast them as object

data[['pclass','sibsp','parch']] = data[['pclass','sibsp','parch']].astype('O')


# check absence of missing data

data.isnull().sum()


# **Important**
#
# In all feature selection procedures, it is good practice to select the features by examining only the training set. And this is to avoid overfit.


# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['survived'], axis=1),
    data['survived'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# feautre engine automates the selection for both
# categorical and numerical variables

sel = SelectByTargetMeanPerformance(
    variables=None, # automatically finds categorical and numerical variables
    scoring="roc_auc_score", # the metric to evaluate performance
    threshold=0.6, # the threshold for feature selection, 
    bins=3, # the number of intervals to discretise the numerical variables
    strategy="equal_frequency", # whether the intervals should be of equal size or equal number of observations
    cv=2,# cross validation
    random_state=1, #seed for reproducibility
)

sel.fit(X_train, y_train)


# after fitting, we can find the categorical variables
# using this attribute

sel.variables_categorical_


# and here we find the numerical variables

sel.variables_numerical_


# here the selector stores the roc-auc per feature

sel.feature_performance_


# and these are the features that will be dropped

sel.features_to_drop_


X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape


# That is all for this lecture, I hope you enjoyed it and see you in the next one!

