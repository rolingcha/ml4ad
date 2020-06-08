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

# %% Python Machine Learning
#   Chapter 3 - A Tour of Machine Learning Classifiers Using scikit-learn
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
print('accuracy_score.Accuracy: %.4f' % accuracy_score(y_test_np, y_ppn_pred))
print('ppn.score.Accuracy: %.4f' % ppn.score(X_test_ss_np, y_test_np))

# Modeling class probabilities via logistic regression              <lr>
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1, 
                        solver='lbfgs', multi_class='ovr')
lr.fit(X_train_ss_np, y_train_np)
y_lr_pred = lr.predict(X_test_ss_np)
print('lr.score.Accuracy: %.4f' % lr.score(X_test_ss_np, y_test_np))

# Maximum margin classification with support vector machines        <svm>
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_train_ss_np, y_train_np)
print('svm.score.Accuracy: %.4f (kernel=rbf, random_state=1, gamma=0.10, C=10.0)' % svm.score(X_test_ss_np, y_test_np))

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_ss_np, y_train_np)
print('svm.score.Accuracy: %.4f kernel=rbf, random_state=1, gamma=0.2, C=1.0)' % svm.score(X_test_ss_np, y_test_np))

# Decision tree Learning                                            <dtc>
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
dtc.fit(X_train_ss_np, y_train_np)
print('dtc.score.Accuracy: %.4f criterion=gini, max_depth=4, random_state=1' % dtc.score(X_test_ss_np, y_test_np))

# dtc - Combining multiple decision trees via random forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
rfc.fit(X_train_ss_np, y_train_np)
print('rfc.score.Accuracy: %.4f criterion=gini, max_depth=4, random_state=1' % rfc.score(X_test_ss_np, y_test_np))

# K-nearest neighbors – a lazy learning algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_ss_np, y_train_np)
y_pred_knn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.4f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_knn,
classes=[0,1])

# %% Python Machine Learning
#   Chapter 11 - Working with Unlabeled Data – Clustering Analysis
#   K-means clustering using scikit-learn
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X_train_ss_np)

y_pred_kn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.4f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_kn,
classes=[0,1])

# A smarter way of placing the initial cluster centroids using k-means++
km_kmpp = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km_kmpp = km_kmpp.fit_predict(X_train_ss_np)

y_pred_kn = knn.predict(X_test_ss_np)
print('knn.score.Accuracy: %.4f n_neighbors=5, p=2, metric=minkowski' % knn.score(X_test_ss_np, y_test_np))
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

# %% Python Machine Learning
#   Chapter 12 - Implementing a Multilayer Artificial Neural Network from Scratch

# Implementing a multilayer perceptron
# autor: Sebastian Raschka - rasbt
# source:  https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/ch12/neuralnet.py
from neuralnet import NeuralNetMLP

mlp_nn = NeuralNetMLP(n_hidden=100, 
                  l2=0.01, 
                  epochs=200, 
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)

# training
mlp_nn.fit(X_train=X_train_np[:20000],
           y_train=y_train_np[:20000],
           X_valid=X_train_np[20000:],
           y_valid=y_train_np[20000:])

# cost
import matplotlib.pyplot as plt
plt.plot(range(mlp_nn.epochs), mlp_nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

# plotting accuracy
plt.plot(range(mlp_nn.epochs), mlp_nn.eval_['train_acc'],
         label='training')
plt.plot(range(mlp_nn.epochs), mlp_nn.eval_['valid_acc'],
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

# accurracy
y_test_pred = mlp_nn.predict(X_test_np)
acc = (np.sum(y_test_np == y_test_pred).astype(np.float) / X_test_np.shape[0])
print('Test accuracy: %.2f%%' % (acc * 100))

meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_test_pred,
classes=[0,1])

