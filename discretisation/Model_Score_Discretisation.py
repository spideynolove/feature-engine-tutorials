import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X.head(3)
np.unique(y)
X.groupby(y).size()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6,
    random_state=50)
from sklearn.preprocessing import MinMaxScaler
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
features = X.columns.tolist()
lr_model = Pipeline(steps=[('scaler', SklearnTransformerWrapper(transformer
    =MinMaxScaler(), variables=features)), ('algorithm', LogisticRegression())]
    )
lr_model.fit(X_train, y_train)
y_proba_train = lr_model.predict_proba(X_train)[:, 1]
y_proba_test = lr_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score
print(f'Train ROCAUC: {roc_auc_score(y_train, y_proba_train):.4f}')
print(f'Test ROCAUC: {roc_auc_score(y_test, y_proba_test):.4f}')
predictions_df = pd.DataFrame({'model_prob': y_proba_test, 'target': y_test})
predictions_df.head()
from feature_engine.discretisation import EqualFrequencyDiscretiser
disc = EqualFrequencyDiscretiser(q=4, variables=['model_prob'],
    return_boundaries=True)
predictions_df_t = disc.fit_transform(predictions_df)
predictions_df_t.groupby('model_prob')['target'].mean().plot(kind='bar', rot=45
    )
from feature_engine.discretisation import DecisionTreeDiscretiser
disc = DecisionTreeDiscretiser(cv=3, scoring='roc_auc', variables=[
    'model_prob'], regression=False)
predictions_df_t = disc.fit_transform(predictions_df, y_test)
predictions_df_t.groupby('model_prob')['target'].mean().plot(kind='bar')
predictions_df_t['model_prob'].value_counts().sort_index()
import string
tree_predictions = np.sort(predictions_df_t['model_prob'].unique())
ratings_map = {tree_prediction: rating for rating, tree_prediction in zip(
    string.ascii_uppercase, tree_predictions)}
ratings_map
predictions_df_t['cluster'] = predictions_df_t['model_prob'].map(ratings_map)
predictions_df_t.head()
predictions_df_t.groupby('cluster')['target'].mean().plot(kind='bar', rot=0,
    title='Mean Target by Cluster')
predictions_df_t['model_probability'] = predictions_df['model_prob']
predictions_df_t.head()
predictions_df_t.groupby('cluster').agg(lower_boundary=('model_probability',
    'min'), upper_boundary=('model_probability', 'max')).round(3)
