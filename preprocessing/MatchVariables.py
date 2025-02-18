# Generated from: MatchVariables.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # MatchVariables
#
#
# MatchVariables() ensures that the columns in the test set are identical to those
# in the train set.
#
# If the test set contains additional columns, they are dropped. Alternatively, if the
# test set lacks columns that were present in the train set, they will be added with a
# value determined by the user, for example np.nan.


import numpy as np
import pandas as pd

from feature_engine.preprocessing import MatchVariables


# # Load titanic dataset from OpenML

# def load_titanic(filepath='../data/titanic.csv'):
#     # data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
#     data = pd.read_csv(filepath)
#     data = data.replace('?', np.nan)
#     data['cabin'] = data['cabin'].astype(str).str[0]
#     data['pclass'] = data['pclass'].astype('O')
#     data['age'] = data['age'].astype('float')
#     data['fare'] = data['fare'].astype('float')
#     data['embarked'].fillna('C', inplace=True)
#     data.drop(
#         # labels=['name', 'ticket', 'boat', 'body', 'home.dest'],
#         labels=['name', 'ticket'],
#         axis=1, inplace=True,
#     )
#     return data

# data = load_titanic()
# # data.head()
# # data.shape

# # separate the dataset into train and test

# train = data.iloc[0:1000, :]
# test = data.iloc[1000:, :]

# train.shape, test.shape


def load_titanic(train_path='../data/titanic-3/train.csv', test_path='../data/titanic-3/test.csv'):
    # Read both train and test datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Common preprocessing for both datasets
    def preprocess_df(df):
        df = df.replace('?', np.nan)
        df['cabin'] = df['cabin'].astype(str).str[0]
        df['pclass'] = df['pclass'].astype('O')
        df['age'] = df['age'].astype('float')
        df['fare'] = df['fare'].astype('float')
        df['embarked'].fillna('C', inplace=True)
        df.drop(
            labels=['name', 'ticket'],
            axis=1, inplace=True,
        )
        return df
    
    # Apply preprocessing to both datasets
    train = preprocess_df(train)
    test = preprocess_df(test)
    
    return train, test


train, test = load_titanic()
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# set up the transformer
match_cols = MatchVariables(missing_values="ignore")

# learn the variables in the train set
match_cols.fit(train)


# the transformer stores the input variables
# match_cols.input_features_
match_cols.feature_names_in_


# ## 1 - Some columns are missing in the test set


match_cols


# Let's drop some columns in the test set for the demo
test_t = test.drop(["sex", "age"], axis=1)


# test.columns
test_t.shape


test_t.head()


# the transformer adds the columns back
test_tt = match_cols.transform(test_t)

print()
test_tt.head()


# Note how the missing columns were added back to the transformed test set, with
# missing values, in the position (i.e., order) in which they were in the train set.
#
# Similarly, if the test set contained additional columns, those would be removed:


# ## Test set contains variables not present in train set


test_t.loc[:, "new_col1"] = 5
test_t.loc[:, "new_col2"] = "test"

test_t.head()


# set up the transformer with different
# fill value
match_cols = MatchVariables(
    fill_value=0, missing_values="ignore",
)

# learn the variables in the train set
match_cols.fit(train)


test_tt = match_cols.transform(test_t)

print()
test_tt.head()


# Note how the columns that were present in the test set but not in train set were dropped. And now, the missing variables were added back into the dataset with the value 0.

