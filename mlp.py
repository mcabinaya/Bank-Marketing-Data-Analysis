#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:52:18 2018

@author: abinaya
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
#from imblearn.over_sampling import SMOTE
from evaluate_classifier_performance import evaluate_classifier_performance

df_train = pd.read_csv('bank-additional-preprocessed-svm-standardize-train.csv')
df_test = pd.read_csv('bank-additional-preprocessed-svm-standardize-test.csv')

del df_train['Unnamed: 0']
del df_test['Unnamed: 0']


df_train_class = pd.DataFrame(df_train['y'])    
df_train_features = df_train.loc[:, df_train.columns != 'y']

df_test_class = pd.DataFrame(df_test['y'])
df_test_features = df_test.loc[:, df_test.columns != 'y']

### Multi layer perceptrons

mlp_classifier = MLPClassifier(hidden_layer_sizes=[100,100], batch_size=50,)
mlp_classifier.fit(df_train_features, df_train_class)
                     
predicted_train = mlp_classifier.predict(df_train_features)
predicted_test = mlp_classifier.predict(df_test_features)

predicted_prob_train = mlp_classifier.predict_proba(df_train_features)
predicted_prob_test = mlp_classifier.predict_proba(df_test_features)

evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, 'y')
