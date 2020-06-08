# -*- coding: utf-8 -*-
"""
Created on Wed 2020-06-3 07:09

@author: Rolando Inglés
"""

# %% importing modules
import numpy as np
import pandas as pd


def handling_categorial_data(input_df):
    """
        One-Hot Encoding a Feature on a Pandas Dataframe: Examples
        http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
        IMPORTANT: By default, the get_dummies() does not do dummy encoding, but one-hot encoding.
    """

    proto_one_hot_df = pd.get_dummies(input_df['proto'], prefix='proto', dummy_na=True)
    state_one_hot_df = pd.get_dummies(input_df['state'], prefix='state', dummy_na=True)

    """
    Guide to Encoding Categorical Values in Python
    https://pbpython.com/categorical-encoding.html
    """
    service_dash_replacement_dict = {'service': {'-': 'not-used-srvc' }}
    input_df.replace(service_dash_replacement_dict,
                                        inplace=True)
    service_one_hot_df = pd.get_dummies(input_df['service'],
                                        prefix='service',
                                        dummy_na=True)
    
    return proto_one_hot_df, state_one_hot_df, service_one_hot_df

# %% loading features descriptions
features_df = pd.read_csv("NUSW-NB15_features.csv", encoding='ANSI')

column_names_srs = features_df['Name']

# %% Data Collection - loading training set data
"""
first three characters are wrong
"""
UNSW_NB15_1_training_set_csv_df = pd.read_csv("UNSW_NB15_1_training_set.csv",
                                       encoding='ANSI',
                                       header=None,
                                       names=column_names_srs)

UNSW_NB15_2_hold_out_set_csv_df = pd.read_csv("UNSW_NB15_2_hold_out_set.csv",
                                       encoding='ANSI',
                                       header=None,
                                       names=column_names_srs)

# %% Data Wrangling - 1. filtering features

selected_features_list = ['proto', 'state', 'sbytes', 'dbytes',
                 'sttl', 'dttl', 'sloss', 'dloss', 'service',
                 'Sload', 'Dload', 'Spkts', 'Dpkts',
                 'swin', 'dwin', 'Sintpkt', 'Dintpkt',
                 'tcprtt', 'synack', 'ackdat']

X_training_set_df = UNSW_NB15_1_training_set_csv_df.filter(selected_features_list)
X_hold_out_set_df = UNSW_NB15_2_hold_out_set_csv_df.filter(selected_features_list)


# %% Data Wrangling - 2. Encoding Categorical Values

# training data set - it will be X_train_df
proto_one_hot_df, state_one_hot_df, service_one_hot_df = handling_categorial_data(X_training_set_df)
concat_X_training_set_df = pd.concat([X_training_set_df,
                              proto_one_hot_df,
                              state_one_hot_df,
                              service_one_hot_df], axis=1)
# _ric_debugging_ concat_X_training_set_df.iloc[:10].to_csv("_X_training_set_df.csv")

# hold-out data set - it will be X_test_df
proto_one_hot_df, state_one_hot_df, service_one_hot_df = handling_categorial_data(X_hold_out_set_df)
concat_X_hold_out_set_df = pd.concat([X_hold_out_set_df,
                              proto_one_hot_df,
                              state_one_hot_df,
                              service_one_hot_df], axis=1)
# _ric_debugging_ concat_X_hold_out_set_df.iloc[0:10].to_csv("_X_hold_out_set_df.csv")

# %% just checking missing columns
ric_a = concat_X_training_set_df.columns.values
ric_b = concat_X_hold_out_set_df.columns.values
# which columns are on X_training but not on X_hold_out
print("feature(s) not present on hold-out set ", np.setdiff1d(ric_a, ric_b))

# which columns are on X_hold_out but not on X_training
print("feature(s) not present on training set", np.setdiff1d(ric_b, ric_a))

# %% Data Wrangling - 3. Filtering Features

X_train_df = concat_X_training_set_df.drop(columns = ['proto', 'state', 'service'])
y_train_df = UNSW_NB15_1_training_set_csv_df.filter(['Label'])

X_test_df = concat_X_hold_out_set_df.drop(columns = ['proto', 'state', 'service'])
y_test_df = UNSW_NB15_2_hold_out_set_csv_df.filter(['Label'])

# _ric_ X_train_test_columns = X_train_df.columns.values

X_train_np = X_train_df.to_numpy()
y_train_np = y_train_df['Label'].to_numpy()

X_test_np = X_test_df.to_numpy()
y_test_np = y_test_df.to_numpy()

X_feature_names_np = X_train_df.columns.values

# %% Machine Learning for Cybersecurity Cookbook
# Chapter 3 - A Tour of Machine Learning Classifiers Using scikit-learn
import ric_model_evaluation_utils as meu

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_np)
X_train_ss_np = sc.transform(X_train_np)
X_test_ss_np = sc.transform(X_test_np)

#   First steps with scikit-learn – training a Perceptron           <ppn>
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_ss_np, y_train_np)

y_ppn_pred = ppn.predict(X_test_ss_np)
print('Misclassified examples: %d' % (y_test_np != y_ppn_pred).sum())

from sklearn.metrics import accuracy_score
print('accuracy_score.Accuracy: %.3f' % accuracy_score(y_test_np, y_ppn_pred))
print('ppn.score.Accuracy: %.3f' % ppn.score(X_test_ss_np, y_test_np))

# Modeling class probabilities via logistic regression              <lr>
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1, 
                        solver='lbfgs', multi_class='ovr')
lr.fit(X_train_ss_np, y_train_np)
y_lr_pred = lr.predict(X_test_ss_np)
print('lr.score.Accuracy: %.3f' % lr.score(X_test_ss_np, y_test_np))

# Maximum margin classification with support vector machines        <svm>
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_train_ss_np, y_train_np)
print('svm.score.Accuracy: %.3f (kernel=rbf, random_state=1, gamma=0.10, C=10.0)' % svm.score(X_test_ss_np, y_test_np))

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_ss_np, y_train_np)
print('svm.score.Accuracy: %.3f kernel=rbf, random_state=1, gamma=0.2, C=1.0)' % svm.score(X_test_ss_np, y_test_np))

# Decision tree Learning                                            <dtc>
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
dtc.fit(X_train_ss_np, y_train_np)
print('dtc.score.Accuracy: %.3f criterion=gini, max_depth=4, random_state=1' % dtc.score(X_test_ss_np, y_test_np))

# dtc - Combining multiple decision trees via random forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
rfc.fit(X_train_ss_np, y_train_np)
print('rfc.score.Accuracy: %.3f criterion=gini, max_depth=4, random_state=1' % rfc.score(X_test_ss_np, y_test_np))

# K-nearest neighbors – a lazy learning algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_ss_np, y_train_np)
y_pred_knn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.3f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_knn,
classes=[0,1])

# %% Machine Learning for Cybersecurity Cookbook
# Chapter 11 - Working with Unlabeled Data – Clustering Analysis
#   K-means clustering using scikit-learn
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X_train_ss_np)

y_pred_kn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.3f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_kn,
classes=[0,1])

# A smarter way of placing the initial cluster centroids using k-means++
km_kmpp = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km_kmpp = km_kmpp.fit_predict(X_train_ss_np)

y_pred_kn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.3f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_kn,
classes=[0,1])

# =============================================================================
# # Applying agglomerative clustering via scikitlearn
# from sklearn.cluster import AgglomerativeClustering
# ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
# # _ric_stuck_ labels = ac.fit_predict(X_train_ss_np)
# labels = ac.fit_predict(X_train_ss_np[0:1000])
# print('Cluster labels: %s' % labels)
# 
# =============================================================================

# Locating regions of high density via DBSCAN

# %% Hands-On Machine Learning with Scikit-Learn and TensorFlow
# Chapter 10 - Introduction to Artificial Neural Networks with Keras
#   Implementing MLP with Keras
# source: https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

import tensorflow as tf
from tensorflow import keras



#Let's split the full training set into a validation set 
# and a (smaller) training set. 
# We also scale the pixel intensities down to the 0-1 range and 
# convert them to floats, by dividing by 255.

# In [13]:
X_train_np.shape

# In [15]:
# =============================================================================
# X_valid_for_mlp_nn, X_train_for_mlp_nn = X_train_np[:5000] / 255., X_train_np[5000:] / 255.
# y_valid_for_mlp_nn, y_train_for_mlp_nn = y_train_np[:5000], y_train_np[5000:]
# X_test_np = X_test_np / 255.
# =============================================================================

X_valid_for_mlp_nn = X_train_np[:20000]
y_valid_for_mlp_nn = y_train_np[:20000]

X_train_for_mlp_nn = X_train_np[20000:]
y_train_for_mlp_nn = y_train_np[20000:]

# In 23:
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[46]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# In 24:
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# In 25:
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[46]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# In 27:
model.summary()

# In 36:
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# =============================================================================
# This is equivalent to:
# 
# model.compile(loss=keras.losses.sparse_categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(),
#               metrics=[keras.metrics.sparse_categorical_accuracy])
# =============================================================================

# In 37:
history = model.fit(X_train_for_mlp_nn, y_train_for_mlp_nn, epochs=100,
                    validation_data=(X_valid_for_mlp_nn, y_valid_for_mlp_nn))


# In [41]:
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
save_fig("keras_learning_curves_plot")
plt.show()

# In [42]:
model.evaluate(X_test_np, y_test_np)


# In [43]:
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)




