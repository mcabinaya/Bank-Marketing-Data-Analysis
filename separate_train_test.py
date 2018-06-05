#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:16:27 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing_mode_impute import preprocessing_features, preprocessing_class

def separate_train_test(df):
    df_class = pd.DataFrame(df['y'])
    df_features = df.loc[:, df.columns != 'y']
    
    df_features_train, df_features_test, df_class_train,  df_class_test = train_test_split(df_features, df_class)
    
    df_train = pd.concat([df_features_train, df_class_train], axis=1)
    df_test = pd.concat([df_features_test, df_class_test], axis=1)

    return df_train, df_test

df = pd.read_csv('Data/bank-additional.csv')
df_train, df_test = separate_train_test(df)
df_train, df_test = preprocessing_features(df_train, df_test,"Standardize")
df_train, df_test = preprocessing_class(df_train, df_test)

df_train.to_csv('Data/bank-additional-preprocessed-mode-standardize-train.csv')
df_test.to_csv('Data/bank-additional-preprocessed-mode-standardize-test.csv')
