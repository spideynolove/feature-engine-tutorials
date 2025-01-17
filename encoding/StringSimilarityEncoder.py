# Generated from: StringSimilarityEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Imports


import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import StringSimilarityEncoder


# # Load and preprocess data


# Helper function for loading and preprocessing data
def load_titanic() -> pd.DataFrame:
    translate_table = str.maketrans('' , '', string.punctuation)
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = data.replace('?', np.nan)
    data['home.dest'] = (
        data['home.dest']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
    )
    data['name'] = (
        data['name']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
    )
    data['ticket'] = (
        data['ticket']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
    )
    return data


# Load dataset
data = load_titanic()


# Separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['survived', 'sex', 'cabin', 'embarked'], axis=1),
    data['survived'],
    test_size=0.3,
    random_state=0
)


# # StringSimilarityEncoder


# set up the encoder
encoder = StringSimilarityEncoder(top_categories=2, variables=['name', 'home.dest', 'ticket'])


# fit the encoder
encoder.fit(X_train)


# lets see what categories we will be comparing to others
encoder.encoder_dict_


# transform the data
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)


# check output
train_t.head(5)


# check output
test_t.head(5)


# plot encoded column - ticket
# OHE could produce only 0, but SSE produces values in [0,1] range
fig, ax = plt.subplots(2, 1);
train_t.plot(kind='scatter', x='ticket_ca 2343', y='ticket_ca 2144', sharex=True, title='Ticket encoding in train', ax=ax[0]);
test_t.plot(kind='scatter', x='ticket_ca 2343', y='ticket_ca 2144', sharex=True, title='Ticket encoding in test', ax=ax[1]);


# defining encoder that ignores NaNs
encoder = StringSimilarityEncoder(
    top_categories=2,
    handle_missing='ignore',
    variables=['name', 'home.dest', 'ticket']
)


# refiting the encoder
encoder.fit(X_train)


# lets see what categories we will be comparing to others
# note - no empty strings with handle_missing='ignore'
encoder.encoder_dict_


# transform the data
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)


# check output
train_t.head(5)


# check output
test_t.head(5)


# plot encoded column - home.dest
fig, ax = plt.subplots(2, 1);
train_t.plot(
    kind='scatter',
    x='home.dest_new york ny',
    y='home.dest_london',
    sharex=True,
    title='Home destination encoding in train',
    ax=ax[0]
);
test_t.plot(
    kind='scatter',
    x='home.dest_new york ny',
    y='home.dest_london',
    sharex=True,
    title='Home destination encoding in test',
    ax=ax[1]
);


# # Note on dimensionality reduction


# These encoded columns could also be compressed further to reduce dimensions
# since they are not boolean, but real numbers
from sklearn.decomposition import PCA


# defining encoder for home destination
encoder = StringSimilarityEncoder(
    top_categories=None,
    handle_missing='impute',
    variables=['home.dest']
)


# refiting the encoder
encoder.fit(X_train)


# transform the data
train_t = encoder.transform(X_train)


# check the shape (should be pretty big)
train_t.shape


# take home.dest encoded columns
home_encoded = train_t.filter(like='home.dest')


# defining PCA for compression
pca = PCA(n_components=0.9)


# train PCA
pca.fit(home_encoded)


# transform train and test datasets
train_compressed = pca.transform(home_encoded)


# check compressed shape (should be way smaller)
train_compressed.shape

