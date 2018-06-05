#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:51:20 2018

@author: abinaya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
    

### Support Vector Machine Model

C_list = np.linspace(0.5, 2.2, 7)
gamma_list = np.linspace(0.01, 0.05, 5)

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

max_iterations = 5
for t in range(0,max_iterations):
    print "---Iteration: ",t
    AVG_ACC = np.zeros(shape=[len(C_list),len(gamma_list)])
    STD_ACC = np.zeros(shape=[len(C_list),len(gamma_list)])
    
    x_count = 0
    for c_value in C_list:
        
        y_count = 0
        for gamma_value in gamma_list:
            print c_value, gamma_value

            temp_accuracy_list = []
            for train_subset_index, cv_index in skf_model.split(df_train_features,df_train_class):
                df_train_features_subset = df_train_features.loc[train_subset_index]
                df_train_class_subset = df_train_class.loc[train_subset_index]
                df_train_features_cv = df_train_features.loc[cv_index]
                df_train_class_cv = df_train_class.loc[cv_index]
                
                svm_model = SVC(C = c_value, gamma = gamma_value, kernel = 'rbf', class_weight= 'balanced')
                svm_model.fit(df_train_features_subset, df_train_class_subset)
                score_value = svm_model.score(df_train_features_cv, df_train_class_cv)
                temp_accuracy_list.append(score_value)
            
            AVG_ACC[x_count,y_count] = np.mean(temp_accuracy_list)
            STD_ACC[x_count,y_count] = np.std(temp_accuracy_list)
            y_count += 1
            
        x_count += 1
    
    if t==0:
        final_AVG_ACC = AVG_ACC
        final_STD_ACC = STD_ACC
    else:
        final_AVG_ACC = np.dstack([final_AVG_ACC, AVG_ACC])
        final_STD_ACC = np.dstack([final_STD_ACC, STD_ACC])
             
final_accuracy_mean_list = np.mean(final_AVG_ACC, axis=2)
max_ind = np.unravel_index(np.argmax(final_accuracy_mean_list, axis=None), final_accuracy_mean_list.shape)

chosen_C = C_list[max_ind[0]]
chosen_gamma = gamma_list[max_ind[1]]

print "By Cross Validation - Chosen C for SVM: ",chosen_C
print "By Cross Validation - Chosen Gamma for SVM: ",chosen_gamma

svm_model_final = SVC(C = c_value, gamma = gamma_value, kernel = 'rbf', class_weight='balanced', probability=True)
#svm_model_final = SVC(C = 1.5, gamma = 0.05,kernel = 'rbf', class_weight='balanced',probability=True)

svm_model_final.fit(df_train_features, df_train_class)
                     
predicted_train = svm_model_final.predict(df_train_features)
predicted_test = svm_model_final.predict(df_test_features)

predicted_prob_train = np.array([])
predicted_prob_test = np.array([])

predicted_prob_train = svm_model_final.predict_proba(df_train_features)
predicted_prob_test  = svm_model_final.predict_proba(df_test_features)

evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, 'y')
