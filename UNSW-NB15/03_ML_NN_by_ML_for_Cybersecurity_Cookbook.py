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
                 'tcprtt', 'synack', 'ackdat', 'Label']

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
#y_train_df = UNSW_NB15_1_training_set_csv_df.filter(['Label'])

X_test_df = concat_X_hold_out_set_df.drop(columns = ['proto', 'state', 'service'])
#y_test_df = UNSW_NB15_2_hold_out_set_csv_df.filter(['Label'])

# _ric_ X_train_test_columns = X_train_df.columns.values

# X_train_np = X_train_df.to_numpy()
# y_train_np = y_train_df['Label'].to_numpy()

# X_test_np = X_test_df.to_numpy()
# y_test_np = y_test_df.to_numpy()

# X_feature_names_np = X_train_df.columns.values

# %% Machine Learning for Cybersecurity Cookbook
#   Chapter 6 - Automatic Intrusion Detection
#       Network behavior anomaly detection

# 6. Split the dataset into normal and abnormal observations:
X_normal_df = X_train_df[X_train_df["Label"] == 0]
X_abnormal_df = X_train_df[X_train_df["Label"] == 1]

y_normal = X_normal_df.pop("Label").values
X_normal = X_normal_df.values

y_anomaly = X_abnormal_df.pop("Label").values
X_anomaly = X_abnormal_df.values


# %%
# 7. Train-test split the dataset:
from sklearn.model_selection import train_test_split

X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=0.3, random_state=11)

X_anomaly_train, X_anomaly_test, y_anomaly_train, y_anomaly_test = train_test_split(X_anomaly, y_anomaly, test_size=0.3, random_state=11)

import numpy as np
X_train = np.concatenate((X_normal_train, X_anomaly_train))
y_train = np.concatenate((y_normal_train, y_anomaly_train))
X_test = np.concatenate((X_normal_test, X_anomaly_test))
y_test = np.concatenate((y_normal_test, y_anomaly_test))


# %%
# 8. Instantiate and train an isolation forest classifier:
from sklearn.ensemble import IsolationForest
IF = IsolationForest(contamination='auto')
IF.fit(X_train)

# %%
# 9. Score the classifier on normal and anomalous observations:
decisionScores_train_normal = IF.decision_function(X_normal_train)
decisionScores_train_anomaly = IF.decision_function(X_anomaly_train)

# %%
# 10. Plot the scores for the normal set:
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_normal, bins=50)

# 11. Similarly, plot the scores on the anomalous observations for a visual
# examination:

plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_anomaly, bins=50)

# 12. Select a cut-off so as to separate out the anomalies from the normal observations:
cutoff = 0
# 13. Examine this cut-off on the test set:
from collections import Counter
print(Counter(y_test))
print(Counter(y_test[cutoff > IF.decision_function(X_test)]))


# todo
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html

# https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e

