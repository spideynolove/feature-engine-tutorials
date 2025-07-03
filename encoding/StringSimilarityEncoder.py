import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import StringSimilarityEncoder


def load_titanic(filepath='titanic.csv'):
    translate_table = str.maketrans('', '', string.punctuation)
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['name'] = data['name'].str.strip().str.translate(translate_table
        ).str.replace('  ', ' ').str.lower()
    data['ticket'] = data['ticket'].str.strip().str.translate(translate_table
        ).str.replace('  ', ' ').str.lower()
    return data


data = load_titanic('../data/titanic-2/Titanic-Dataset.csv')
data.head()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['survived',
    'sex', 'cabin', 'embarked'], axis=1), data['survived'], test_size=0.3,
    random_state=0)
encoder = StringSimilarityEncoder(top_categories=2, variables=['name',
    'ticket'])
encoder.fit(X_train)
encoder.encoder_dict_
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)
train_t.head(5)
test_t.head(5)
fig, ax = plt.subplots(2, 1)
train_t.plot(kind='scatter', x='ticket_ca 2343', y='ticket_347082', sharex=
    True, title='Ticket encoding in train', ax=ax[0])
test_t.plot(kind='scatter', x='ticket_ca 2343', y='ticket_347082', sharex=
    True, title='Ticket encoding in test', ax=ax[1])
encoder = StringSimilarityEncoder(top_categories=2, missing_values='ignore',
    variables=['name', 'ticket'])
encoder.fit(X_train)
encoder.encoder_dict_
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)
train_t.head(5)
test_t.head(5)
fig, ax = plt.subplots(2, 1)
train_t.plot(kind='scatter', x='home.dest_new york ny', y=
    'home.dest_london', sharex=True, title=
    'Home destination encoding in train', ax=ax[0])
test_t.plot(kind='scatter', x='home.dest_new york ny', y='home.dest_london',
    sharex=True, title='Home destination encoding in test', ax=ax[1])
from sklearn.decomposition import PCA
encoder = StringSimilarityEncoder(top_categories=None, handle_missing=
    'impute', variables=['home.dest'])
encoder.fit(X_train)
train_t = encoder.transform(X_train)
train_t.shape
home_encoded = train_t.filter(like='home.dest')
pca = PCA(n_components=0.9)
pca.fit(home_encoded)
train_compressed = pca.transform(home_encoded)
train_compressed.shape
