#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:18:37 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from evaluate_classifier_performance import evaluate_classifier_performance

df_train = pd.read_csv('Data/bank-additional-preprocessed-mode-normalize-train.csv')
df_test = pd.read_csv('Data/bank-additional-preprocessed-mode-normalize-test.csv')

del df_train['Unnamed: 0']
del df_test['Unnamed: 0']

df_train_class = pd.DataFrame(df_train['y'])    
df_train_features = df_train.loc[:, df_train.columns != 'y']

df_test_class = pd.DataFrame(df_test['y'])
df_test_features = df_test.loc[:, df_test.columns != 'y']
    

### Perceptron Classifier
alpha_list = np.linspace(0.00001, 1, 15)
penality_list = ['l1','l2','elasticnet']

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

max_iterations = 20
for t in range(0,max_iterations):
    print "---Iteration: ",t
    AVG_ACC = np.zeros(shape=[len(alpha_list),len(penality_list)])
    STD_ACC = np.zeros(shape=[len(alpha_list),len(penality_list)])
    
    x_count = 0
    for alpha_value in alpha_list:
        
        y_count = 0
        for penality in penality_list:
            
            temp_accuracy_list = []
            for train_subset_index, cv_index in skf_model.split(df_train_features,df_train_class):
                df_train_features_subset = df_train_features.loc[train_subset_index]
                df_train_class_subset = df_train_class.loc[train_subset_index]
                df_train_features_cv = df_train_features.loc[cv_index]
                df_train_class_cv = df_train_class.loc[cv_index]
                
                perceptron_model = Perceptron(penalty=penality, alpha=alpha_value, class_weight= 'balanced')
                perceptron_model.fit(df_train_features_subset, df_train_class_subset)
                score_value = perceptron_model.score(df_train_features_cv, df_train_class_cv)
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

chosen_alpha = alpha_list[max_ind[0]]
chosen_penalty = penality_list[max_ind[1]]
print "By Cross Validation - Chosen alpha for Perceptron: ",chosen_alpha
print "By Cross Validation - Chosen Penalty for Perceptron: ",chosen_penalty

perceptron_model_final = Perceptron(penalty=chosen_penalty, alpha=chosen_alpha, class_weight= 'balanced')
perceptron_model_final = CalibratedClassifierCV(base_estimator=perceptron_model_final,cv=10, method='isotonic')
perceptron_model_final.fit(df_train_features, df_train_class)
                     
predicted_train = perceptron_model_final.predict(df_train_features)
predicted_test = perceptron_model_final.predict(df_test_features)

predicted_prob_train = perceptron_model_final.predict_proba(df_train_features)
predicted_prob_test  = perceptron_model_final.predict_proba(df_test_features)

evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, 'y')
