# -*- coding: utf-8 -*-
"""
Created on Wed 2020-06-3 07:09

@author: Rolando Inglés
"""

# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pd_df_label_to_np_arr_int(input_labels_pandas_dataframe):
    
    #y_train_np_array = np.empty(len(y_train_df), dtype=int)
    ouput_int_np_array = np.zeros(len(input_labels_pandas_dataframe), dtype=int)
    
    #print(y_train_df[0:10])
    ftp_cnt = 0
    ssh_cnt = 0
    ok_cnt = 0
    
    for input_labels_pandas_dataframe_idx, input_labels_pandas_dataframe_row in input_labels_pandas_dataframe.iterrows():
        if input_labels_pandas_dataframe_row['Label'] == 'FTP-BruteForce':
            ftp_cnt = ftp_cnt + 1
            ouput_int_np_array[input_labels_pandas_dataframe_idx] = 1
        elif input_labels_pandas_dataframe_row['Label'] == 'SSH-Bruteforce':
            ssh_cnt = ssh_cnt + 1
            ouput_int_np_array[input_labels_pandas_dataframe_idx] = 2
        else:
            ok_cnt = ok_cnt + 1
        #print(input_labels_pandas_dataframe_row['Label'])
    print("pd_df_label_to_np_arr_int: \ntotal=", ftp_cnt + ssh_cnt + ok_cnt, 
          "ftp=", ftp_cnt, "ssh=", ssh_cnt, "ok=", ok_cnt)
    return ouput_int_np_array
    

#training_set_df = pd.read_csv("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")



training_set_df = pd.read_csv("IDS2018_20180214_training_set.csv")

print("Keys of iris_dataset: \n{}".format(training_set_df.keys()))

IDS2018_selected_columns = ['Dst Port',
                                     'Protocol',
                                     'Flow Duration',
                                     'Tot Fwd Pkts',
                                     'Tot Bwd Pkts',
                                     'TotLen Fwd Pkts',
                                     'TotLen Bwd Pkts',
                                     'Flow Byts/s',
                                     'Flow Pkts/s',
                                     'Fwd PSH Flags',
                                     'Bwd PSH Flags',
                                     'Fwd URG Flags',
                                     'Bwd URG Flags',
                                     'Fwd Header Len',
                                     'Bwd Header Len',
                                     'Fwd Pkts/s',
                                     'Bwd Pkts/s',
                                     'FIN Flag Cnt',
                                     'SYN Flag Cnt',
                                     'RST Flag Cnt',
                                     'PSH Flag Cnt',
                                     'ACK Flag Cnt',
                                     'URG Flag Cnt',
                                     'CWE Flag Count',
                                     'ECE Flag Cnt',
                                     'Down/Up Ratio',
                                     'Init Fwd Win Byts',
                                     'Init Bwd Win Byts',
                                     'Fwd Act Data Pkts',
                                     'Fwd Seg Size Min'
                                      ]

X_train_df = training_set_df.filter(items=IDS2018_selected_columns)
X_train_np_array = X_train_df.to_numpy()

y_train_df = training_set_df.filter(items=['Label'])
y_train_np_array = pd_df_label_to_np_arr_int(y_train_df)

# %% showing some data information
X_train_df.head()
X_train_df.info()
X_train_df['Protocol'].value_counts()
X_train_df['Dst Port'].value_counts()
X_train_df.describe()

# %% cleaning data

# cleaning NaN 
#Just for checking X_train_df.to_csv("X_train_df_to_csv.output.csv")
#np.isnan(X_train_np_array)
i_j_NaN_coords_tuple = np.where(np.isnan(X_train_np_array))
print(i_j_NaN_coords_tuple[0].shape)
X_train_np_array_no_nan = np.delete(X_train_np_array, i_j_NaN_coords_tuple[0], axis=0)
y_train_np_array_no_nan = np.delete(y_train_np_array, i_j_NaN_coords_tuple[0], axis=0)

# cleaning Infinite
i_j_Infinite_coords_tuple = np.where(np.isinf(X_train_np_array_no_nan))
print(i_j_Infinite_coords_tuple[0].shape)
rows_with_Infinity = np.unique(i_j_Infinite_coords_tuple[0])
print(len(rows_with_Infinity))
X_train_np_array_no_nan_inf = np.delete(X_train_np_array_no_nan, rows_with_Infinity, axis=0)
y_train_np_array_no_nan_inf = np.delete(y_train_np_array_no_nan, rows_with_Infinity, axis=0)


# %% hold-out = test data set 
hold_out_set_df = pd.read_csv("IDS2018_20180214_hold-out_set.csv")

X_hold_out_df = hold_out_set_df.filter(items=IDS2018_selected_columns)
X_hold_out_np_array = X_hold_out_df.to_numpy()

y_hold_out_df = hold_out_set_df.filter(items=['Label'])
y_hold_out_np_array = pd_df_label_to_np_arr_int(y_hold_out_df)

# cleaning NaN
hold_out_i_j_NaN_coords_tuple = np.where(np.isnan(X_hold_out_np_array))
print(hold_out_i_j_NaN_coords_tuple[0].shape)
X_hold_out_np_array_no_NaN = np.delete(X_hold_out_np_array, hold_out_i_j_NaN_coords_tuple[0], axis=0)
y_hold_out_np_array_no_NaN = np.delete(y_hold_out_np_array, hold_out_i_j_NaN_coords_tuple[0], axis=0)

# cleaning infinite
hold_out_i_j_Infinite_coords_tuple = np.where(np.isinf(X_hold_out_np_array_no_NaN))
print(hold_out_i_j_Infinite_coords_tuple[0].shape)
hold_out_rows_with_Infinity = np.unique(hold_out_i_j_Infinite_coords_tuple[0])
print(len(hold_out_rows_with_Infinity))
X_hold_out_np_array_no_nan_inf = np.delete(X_hold_out_np_array_no_NaN, hold_out_rows_with_Infinity, axis=0)
y_hold_out_np_array_no_nan_inf = np.delete(y_hold_out_np_array_no_NaN, hold_out_rows_with_Infinity, axis=0)

X_hold_out_pd_df_no_nan_inf = pd.DataFrame(X_hold_out_np_array_no_nan_inf, None, IDS2018_selected_columns)
y_hold_out_pd_df_no_nan_inf = pd.DataFrame(y_hold_out_np_array_no_nan_inf, None, ["Label"])


# %% pandas to numpy
X_train_np = X_train_np_array_no_nan_inf
y_train_np = y_train_np_array_no_nan_inf

X_test_np = X_hold_out_pd_df_no_nan_inf.to_numpy()
y_test_np = y_hold_out_pd_df_no_nan_inf['Label'].to_numpy()

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

y_pred_kn = km.predict(X_test_ss_np)
print('km.score.Accuracy: %.4f n_neighbors=5, p=2, metric=minkowski' % km.score(X_test_ss_np, y_test_np))
meu.display_model_performance_metrics(true_labels=y_test_np, predicted_labels=y_pred_kn,
classes=[0,1])

# A smarter way of placing the initial cluster centroids using k-means++
km_kmpp = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km_kmpp = km_kmpp.fit_predict(X_train_ss_np)

y_pred_kn = km.predict(X_test_ss_np)
print('km.score.Accuracy: %.4f n_neighbors=5, p=2, metric=minkowski' % km.score(X_test_ss_np, y_test_np))
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
import matplotlib.pyplot as plt

mlp_nn = NeuralNetMLP(n_hidden=100, 
                  l2=0.01, 
                  epochs=200, 
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)

# training
mlp_nn.fit(X_train=X_train_np[:13000],
           y_train=y_train_np[:13000],
           X_valid=X_train_np[13000:],
           y_valid=y_train_np[13000:])

# cost

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

