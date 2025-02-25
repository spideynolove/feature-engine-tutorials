# Generated from: Drop-High-PSI-Features.ipynb

# # Drop Features with High PSI Value
#
# The **DropHighPSIFeatures** selects features based on the Population Stability Index (PSI). The higher this value, the more unstable a feature. Unstable in this case means that there is a significant change in the distribution of the feature in the groups being compared.
#
# To determine the PSI of a feature, the DropHighPSIFeatures takes a dataframe and splits it in 2 based on a reference variable. This reference variable can be numerical, categorical or date. If the variable is numerical, the split ensures a certain proportion of observations in each sub-dataframe. If the variable is categorical, we can split the data based on the categories. And if the variable is a date, we can split the data based on dates.
#
# **In this notebook, we showcase many possible ways in which the DropHighPSIFeatures can be used to select features based on their PSI value.**
#
# ### Dataset
#
# We use the Credit Approval data set from the UCI Machine Learning Repository.
#
# To download the Credit Approval dataset from the UCI Machine Learning Repository visit [this website](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/) and click on crx.data to download data. Save crx.data to the parent folder to this notebook folder.
#
# **Citation:**
#
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
#
# # Data preparation
#
# We will edit some of the original variables and add some additional features to simulate different scenarios.


from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from feature_engine.selection import DropHighPSIFeatures


# ## Load the data


# load data
data = pd.read_csv('../data/credit+approval/crx.data', header=None)

# add variable names according to UCI Machine Learning
# Repo information
data.columns = ['A'+str(s) for s in range(1, 17)]

# replace ? by np.nan
data = data.replace('?', np.nan)

# re-cast some variables to the correct types
data['A2'] = data['A2'].astype('float')
data['A14'] = data['A14'].astype('float')

# encode target as binary
data['A16'] = data['A16'].map({'+': 1, '-': 0})

data.head()


# ## Edit and add features


# simulate customers from different portfolios.
data['A13'] = data['A13'].map(
    {'g': 'portfolio_1', 's': 'portfolio_2', 'p': 'portfolio_3'})
data['A13'].fillna('Unknown', inplace=True)

# simulate customers from different channels
data['A12'] = data['A12'].map({'f': 'wholesale', 't': 'retail'})
data['A12'].fillna('Missing', inplace=True)


# simulate customers from different age groups

data['A6'].fillna('Missing', inplace=True)

labels = {
    'w': '20-25',
    'q': '25-30',
    'm': '30-35',
    'r': '35-40',
    'cc': '40-45',
    'k': '45-50',
    'c': '50-55',
    'd': '55-60',
    'x': '60-65',
    'i': '65-70',
    'e': '70-75',
    'aa': '75-80',
    'ff': '85-90',
    'j': 'Unknown',
    'Missing': 'Missing',
}

data['A6'] = data['A6'].map(labels)


# add a datetime variable

data['date'] = pd.date_range(start='1/1/2018', periods=len(data))

data.head()


# ## Data Analysis
#
# We will plot the distributions of numerical and categorical variables.


# categorical variables

vars_cat = data.select_dtypes(include='O').columns.to_list()

vars_cat


for var in vars_cat:
    data[var].value_counts(normalize=True).plot.bar()
    plt.title(var)
    plt.ylabel('% observations')
    plt.show()


# numerical variables

vars_num = data.select_dtypes(exclude='O').columns.to_list()

vars_num.remove('A16')

vars_num.remove('date')

vars_num


for var in vars_num:
    data[var].hist(bins=50)
    plt.title(var)
    plt.ylabel('Number observations')
    plt.show()


# # PSI feature selection
#
# ## Split data based on proportions
#
# DropHighPSIFeatures splits the dataset in 2, a base dataset and a comparison dataset. The comparison dataset is compared against the base dataset to determine the PSI.
#
# We may want to divide the dataset just based on **proportion of observations**. We want to have, say, 60% of observations in the base dataset. We can use the **dataframe index** to guide the split.
#
# **NOTE** that for the split, the transformer orders the variable, here the index, and then smaller values of the variable will be in the base dataset, and bigger values of the variable will go to the test dataset. In other words, this is not a random split.


# First, we split the data into a train and a test set

X_train, X_test, y_train, y_test = train_test_split(
    data[vars_cat+vars_num],
    data['A16'],
    test_size=0.1,
    random_state=42,
)


# Now we set up the DropHighPSIFeatures
# to split based on fraction of observations

transformer = DropHighPSIFeatures(
    split_frac=0.6,  # the proportion of obs in the base dataset
    split_col=None,  # If None, it uses the index
    strategy='equal_frequency',  # whether to create the bins of equal frequency
    threshold=0.1,  # the PSI threshold to drop variables
    variables=vars_num,  # the variables to analyse
    missing_values='ignore',
)


# Now we fit the transformer to the train set.

# Here, the transformer will split the data,
# determine the PSI of each feature and identify
# those that will be removed.

transformer.fit(X_train)


# the value in the index that determines the separation
# into base and comparison datasets.

# Observations whose index value is smaller than
# the cut_off will be in the base dataset.
# The remaining ones in the test data.

transformer.cut_off_


# the PSI threshold above which variables
# will be removed.

# We can change this when we initialize the transformer

transformer.threshold


# During fit() the transformer determines the PSI
# values for each variable and stores it.

transformer.psi_values_


# The variables that will be dropped:
# those whose PSI is biggher than the threshold.

transformer.features_to_drop_


# To understand what the DropHighPSIFeatures is doing, let's split the train set manually, in the same what that the transformer is doing. Then, let's plot the distribution of the variables in each of the sub-dataframes.


# Let's plot the variables distribution
# in each of the dataset portions

# create series to flag if an observation belongs to
# the base or test dataframe.

# Note how we use the cut_off identified by the
# transformer:
tmp = X_train.index <= transformer.cut_off_

# plot
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - moderate PSI')


# We observe a difference in the cumulative distribution of A8 between dataframes.


# For comparison, let's plot a variable with low PSI

sns.ecdfplot(data=X_train, x='A2', hue=tmp)
plt.title('A2 - low PSI')


# We see that the cumulative distribution of A8 is different in both datasets and this is why it is flagged for removal. On the other hand, the cumulative distribution of A2 is not different in the sub-datasets.
#
# Now we can go ahead and drop the features from the train and test sets. We use the transform() method.


# print shape before dropping variables

X_train.shape, X_test.shape


X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

# print shape **after** dropping variables

X_train.shape, X_test.shape


# The datasets have now 2 variables less, those that had higher PSI values.
#
# ## Split data based on categorical values
#
# In the previous example, we sorted the observations based on a numerical variable, the index, and then we assigned the top 60% of the observations to the base dataframe.
#
# Now, we will sort the observations based on a categorical variable, and assign the top 50% to the base dataframe.
#
# **Note** when splitting based on categorical variables the proportions achieved after the split may not match exactly the one specified.
#
# ### When is this split useful?
#
# This way of splitting the data is useful when, for example, we have a variable with the customer's ID. The ID's normally increase in time, with smaller values corresponding to older customers and bigger ID values corresponding to newly acquired customers.
#
# **Our example**
#
# In our data, we have customers from different age groups. We want to know if the variable distribution in younger age groups differ from older age groups. This is a suitable case to split based on a categorical value without specifically specifying the cut_off.
#
# The transformer will sort the categories of the variable and then those with smaller category values will be in the base dataframe, and the remaining in the comparison dataset.


# First, we split the data into a train and a test set

X_train, X_test, y_train, y_test = train_test_split(
    data[vars_cat+vars_num],
    data['A16'],
    test_size=0.1,
    random_state=42,
)


# Now, we set up the transformer

# Note that if we do not specify which variables to analyse,
# the transformer will find the numerical variables automatically

transformer = DropHighPSIFeatures(
    split_frac=0.5,  # percentage of observations in base df
    split_col='A6',  # the categorical variable with the age groups
    strategy='equal_frequency',
    bins=8,  # the number of bins into which the observations should be sorted
    threshold=0.1,
    variables=None,  # When None, finds numerical variables automatically
    missing_values='ignore',
)


# Now we fit the transformer to the train set.

# Here, the transformer will split the data,
# determine the PSI of each feature and identify
# those that will be removed.

transformer.fit(X_train)


# The transformer identified the numerical variables

transformer.variables_


# the age group under which observations will be
# in the base df.

transformer.cut_off_


# The PSI values determined for each feature

transformer.psi_values_


# The variables that will be dropped.

transformer.features_to_drop_


# There is no significant shift in the distribution of the variables between younger and older customers. Thus, no variables will be dropped.
#
# To understand what the DropHighPSIFeatures is doing, let's split the train set manually, in the same what that the transformer is doing. Then, let's plot the distribution of the variables in each of the sub-dataframes.


# Let's plot the variables distribution
# in each of the dataset portions

# create series to flag if an observation belongs to
# the base or comparison dataframe.

# Note how we use the cut_off identified by the
# transformer
tmp = X_train['A6'] <= transformer.cut_off_

# plot
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - low PSI')


# Let's plot another variable with low PSI

sns.ecdfplot(data=X_train, x='A15', hue=tmp)
plt.title('A15 - low PSI')


# As we can see, the distributions of the variables in both dataframes is quite similar.
#
# Now, let's identify which observations were assigned to each sub-dataframe by the transformer.


# The observations belonging to these age groups
# were assigned to the base df.

X_train[tmp]['A6'].unique()


# The number of age groups in the base df

X_train[tmp]['A6'].nunique()


# Proportion of observations in the base df

len(X_train[tmp]['A6']) / len(X_train)


# Note that we aimed for 50% of observations in the base reference, but based on this categorical variable, the closer we could get is 41%.


# The observations belonging to these age groups
# were assigned to the comparison df.

X_train[~tmp]['A6'].unique()


# The number of age groups in the comparison df

X_train[~tmp]['A6'].nunique()


# Proportion of observations in the comparison df

len(X_train[~tmp]['A6']) / len(X_train)


# Note that we have more age groups in the comparison df, but these groups have fewer observations, so the proportion of observations in the base and test dfs is the closest possible to what we wanted: 50%.
#
# Now we can go ahead and drop the features from the train and test sets.
#
# In this case, we would be dropping None.


# print shape before dropping variables

X_train.shape, X_test.shape


X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

# print shape **after** dropping variables

X_train.shape, X_test.shape


# ## Split data based on distinct values
#
# In the previous example, we split the data using a categorical variable as guide, but ultimately, the split was done based on proportion of observations.
#
# In the extreme example where 50% of our customers belong to the age group 20-25 and the remaining 50% belong to older age groups, we would have only 1 age group in the base dataframe and all the remaining in the comparison dataframe if we split as we did in our previous example. This may result in a biased comparison.
#
# If we want to ensure that we have 50% of the possible age groups in each base and comparison dataframe, we can do so with the parameter `split_distinct`.


# First, we split the data into a train and a test set

X_train, X_test, y_train, y_test = train_test_split(
    data[vars_cat+vars_num],
    data['A16'],
    test_size=0.1,
    random_state=42,
)


transformer = DropHighPSIFeatures(
    split_frac=0.5,  # proportion of (unique) categories in the base df
    split_distinct=True,  # we split based on unique categories
    split_col='A6',  # the categorical variable guiding the split
    strategy='equal_frequency',
    bins=5,
    threshold=0.1,
    missing_values='ignore',
)


# Now we fit the transformer to the train set
# Here, the transformer will split the data,
# determine the PSI of each feature and identify
# those that will be removed.

transformer.fit(X_train)


# the age group under which, observations will be
# in the base df.

transformer.cut_off_


# Note that this cut_off is different from the one we obtained previously.


# The PSI values determined for each feature

transformer.psi_values_


# The variables that will be dropped.

transformer.features_to_drop_


# To understand what the DropHighPSIFeatures is doing, let's split the train set manually, in the same what that the transformer is doing. Then, let's plot the distribution of the variables in each of the sub-dataframes.


# Let's plot the variables distribution
# in each of the dataset portions

# create series to flag if an observation belongs to
# the base or comparison dataframe.

# Note how we use the cut_off identified by the
# transformer
tmp = X_train['A6'] <= transformer.cut_off_

# plot
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - high PSI')


# There is a mild difference in the variable distribution.


# For comparison, let's plot a variable with low PSI

sns.ecdfplot(data=X_train, x='A15', hue=tmp)
plt.title('A15 - low PSI')


# Now, let's identify which observations were assigned to each sub-dataframe by the transformer.


# The observations belonging to these age groups
# were assigned to the base df.

X_train[tmp]['A6'].unique()


# The number of age groups in the base df

X_train[tmp]['A6'].nunique()


# Proportion of observations in the base df

len(X_train[tmp]['A6']) / len(X_train)


# The observations belonging to these age groups
# were assigned to the comparison df.

X_train[~tmp]['A6'].unique()


# The number of age groups in the comparison df

X_train[~tmp]['A6'].nunique()


# Proportion of observations in the comparison df

len(X_train[~tmp]['A6']) / len(X_train)


# Now, we have a similar proportion of age groups in the base and comparison dfs. But the proportion of observations is different.
#
# Now we can go ahead and drop the features from the train and test sets.


# print shape before dropping variables

X_train.shape, X_test.shape


X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

# print shape **after** dropping variables

X_train.shape, X_test.shape


# ## Split based on specific categories
#
# In the previous example, the categories had an intrinsic order. What if, we want to split based on category values which do not have an intrinsic order?
#
# We can do so by specifying which category values should go to the base dataframe.
#
# This way of splitting the data is useful if we want to compare features across customers coming from different portfolios, or different sales channels.


# First, we split the data into a train and a test set

X_train, X_test, y_train, y_test = train_test_split(
    data[vars_cat+vars_num],
    data['A16'],
    test_size=0.1,
    random_state=42,
)


# Set up the transformer

transformer = DropHighPSIFeatures(
    # the categories that should be in the base df
    cut_off=['portfolio_2', 'portfolio_3'],
    split_col='A13',  # the categorical variable with the portfolios
    strategy='equal_width',  # the intervals are equidistant
    bins=5,  # the number of intervals into which to sort the numerical values
    threshold=0.1,
    variables=vars_num,
    missing_values='ignore',
)


# Now we fit the transformer to the train set
# Here, the transformer will split the data,
# determine the PSI of each feature and identify
# those that will be removed.

transformer.fit(X_train)


# We specified the cut_off, so we should see
# the portfolios here

transformer.cut_off_


# The transformer stores the PSI values of the variables

transformer.psi_values_


# The variables that will be dropped.

transformer.features_to_drop_


# It looks like all variables will be dropped.
#
# To understand what the DropHighPSIFeatures is doing, let's split the train set manually, in the same what that the transformer is doing. Then, let's plot the distribution of the variables in each of the sub-dataframes.


# Let's plot the variables distribution
# in each of the dataset portions

# create series to flag if an observation belongs to
# the base or comparison dataframe.

# Note how we use the cut_off identified by the
# transformer
tmp = X_train['A13'].isin(transformer.cut_off_)

sns.ecdfplot(data=X_train, x='A3', hue=tmp)
plt.title('A3 - high PSI')


# Let's plot another variable with high PSI

sns.ecdfplot(data=X_train, x='A11', hue=tmp)
plt.title('A11 - high PSI')


# Let's plot a variable with lower PSI

sns.ecdfplot(data=X_train, x='A2', hue=tmp)
plt.title('A2 - high PSI')


# Now we can go ahead and drop the features from the train and test sets.


# print shape before dropping variables

X_train.shape, X_test.shape


X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

# print shape **after** dropping variables

X_train.shape, X_test.shape


# ## Split based on Date
#
# If our data had a valid timestamp, we could want to compare the distributions before and after a time point.


# Let's find out which are the minimum
# and maximum dates in our dataset

data['date'].agg(['min', 'max'])


# Now, we split the data into a train and a test set

X_train, X_test, y_train, y_test = train_test_split(
    data[vars_cat+vars_num+['date']],
    data['A16'],
    test_size=0.1,
    random_state=42,
)


# And we specify a transformer to split based
# on dates

transformer = DropHighPSIFeatures(
    cut_off=pd.to_datetime('2018-12-14'),  # the cut_off date
    split_col='date',  # the date variable
    strategy='equal_frequency',
    bins=8,
    threshold=0.1,
    missing_values='ignore',
)


# Now we fit the transformer to the train set.

# Here, the transformer will split the data,
# determine the PSI of each feature and identify
# those that will be removed.

transformer.fit(X_train)


# We specified the cut_off, so we should see
# our value here

transformer.cut_off_


# The transformer stores the PSI values of the variables

transformer.psi_values_


# The variables that will be dropped.

transformer.features_to_drop_


# To understand what the DropHighPSIFeatures is doing, let's split the train set manually, in the same what that the transformer is doing. Then, let's plot the distribution of the variables in each of the sub-dataframes.


# Let's plot the variables distribution
# in each of the dataset portions

# create series to flag if an observation belongs to
# the base or comparison dataframe.

# Note how we use the cut_off identified by the
# transformer
tmp = X_train['date'] <= transformer.cut_off_

# plot
sns.ecdfplot(data=X_train, x='A3', hue=tmp)
plt.title('A3 - moderate PSI')


# For comparison, let's plot a variable with low PSI

sns.ecdfplot(data=X_train, x='A14', hue=tmp)


# Now we can go ahead and drop the features from the train and test sets.


# print shape before dropping variables

X_train.shape, X_test.shape


X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

# print shape **after** dropping variables

X_train.shape, X_test.shape


# That is all!
#
# I hope I gave you a good idea about how we can use this transformer to select features based on the Population Stability Index.
