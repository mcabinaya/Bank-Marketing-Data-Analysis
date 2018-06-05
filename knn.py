#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:38:42 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from evaluate_classifier_performance import evaluate_classifier_performance


df_train = pd.read_csv('Data/bank-additional-preprocessed-mode-standardize-train.csv')
df_test = pd.read_csv('Data/bank-additional-preprocessed-mode-standardize-test.csv')

del df_train['Unnamed: 0']
del df_test['Unnamed: 0']

df_train_class = pd.DataFrame(df_train['y'])    
df_train_features = df_train.loc[:, df_train.columns != 'y']

df_test_class = pd.DataFrame(df_test['y'])
df_test_features = df_test.loc[:, df_test.columns != 'y']


### KNN Model
n_neighbors_list = range(3,15)

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

max_iterations = 20
for t in range(0,max_iterations):
    print "---Iteration: ",t
    AVG_ACC = np.zeros(shape=[len(n_neighbors_list)])
    STD_ACC = np.zeros(shape=[len(n_neighbors_list)])
    
    x_count = 0
    for k_val in n_neighbors_list:
        temp_accuracy_list = []
        
        for train_subset_index, cv_index in skf_model.split(df_train_features,df_train_class):
            df_train_features_subset = df_train_features.loc[train_subset_index]
            df_train_class_subset = df_train_class.loc[train_subset_index]
            df_train_features_cv = df_train_features.loc[cv_index]
            df_train_class_cv = df_train_class.loc[cv_index]
        
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(df_train_features_subset, df_train_class_subset)
            score_value = knn_model.score(df_train_features_cv, df_train_class_cv)
            temp_accuracy_list.append(score_value)
                
        AVG_ACC[x_count] = np.mean(temp_accuracy_list)
        STD_ACC[x_count] = np.std(temp_accuracy_list)
        x_count += 1
    
    if t==0:
        final_AVG_ACC = AVG_ACC
        final_STD_ACC = STD_ACC
    else:
        final_AVG_ACC = np.vstack([final_AVG_ACC, AVG_ACC])
        final_STD_ACC = np.vstack([final_STD_ACC, STD_ACC])
    
final_accuracy_mean_list = np.mean(final_AVG_ACC, axis=0)
final_k_index = np.argmax(final_accuracy_mean_list)

chosen_k= n_neighbors_list[final_k_index]
print "By Cross Validation - Chosen K for KNN: ",chosen_k

knn_model_final = KNeighborsClassifier(n_neighbors=chosen_k)
knn_model_final.fit(df_train_features, df_train_class)
                     
predicted_train = knn_model_final.predict(df_train_features)
predicted_test = knn_model_final.predict(df_test_features)

predicted_prob_train = np.array([])
predicted_prob_test = np.array([])

evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, 'n')
