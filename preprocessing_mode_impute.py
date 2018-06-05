#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:20:07 2018

@author: abinaya
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def preprocessing_features(df_train, df_test, process_continuous):
    to_delete_features = ['default','pdays']
    continuous_features = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_ordered_features = ['education', 'housing', 'loan', 'contact','month', 'day_of_week','poutcome']
    categorical_unordered_features = ['job', 'marital']
    
    unknown_present_features = ['job','marital','education','housing','loan']
    
    ### Delete Features
    for feat in to_delete_features:
        print "\n--------- deleting feature --------- ",feat
        del df_train[feat]
        del df_test[feat]
    
    ### Fill unknowns in the features with feature-mode
    for feat in unknown_present_features:
        print "\n--------- Replacing unknowns in feature --------- ",feat
        feature_value_counts = df_train[feat].value_counts()
        print "Replaced with: ",feature_value_counts.idxmax()
        df_train.loc[df_train[feat] == "unknown",feat] = feature_value_counts.idxmax()
        df_test.loc[df_test[feat] == "unknown",feat] = feature_value_counts.idxmax()
    
    
    ### Label Categorical Ordered Features
    label_dict = {'education':{'illiterate':0, 'basic.4y':4, 'basic.6y':6, 'basic.9y':9, 'high.school':11, 'professional.course':13, 'university.degree':14},
                  'housing':{'no':0,'yes':1},
                  'loan':{'no':0,'yes':1},
                  'contact':{'telephone':0,'cellular':1},
                  'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12},
                  'day_of_week':{'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7},
                  'poutcome':{'nonexistent':0,'failure':1,'success':2}}
    
    for feat in categorical_ordered_features:
        print "\n--------- Labelling feature --------- ",feat
        df_train = df_train.replace({feat:label_dict[feat]})
        df_test = df_test.replace({feat:label_dict[feat]})
        print "Labelled as: ",label_dict[feat]

        
    ### One hot encoding Categorical Un-ordered Features  
    for feat in categorical_unordered_features:
        print "\n--------- One Hot Encoding feature --------- ",feat
        label_encoder = LabelEncoder()
        label_encoder.fit(df_train[feat])
        df_train[feat] = label_encoder.transform(df_train[feat])
        df_test[feat] = label_encoder.transform(df_test[feat])

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(df_train[categorical_unordered_features])
    one_hot_encoded_array_train = one_hot_encoder.transform(df_train[categorical_unordered_features])
    one_hot_encoded_df_train = pd.DataFrame(one_hot_encoded_array_train, index=df_train.index)
    one_hot_encoded_array_test = one_hot_encoder.transform(df_test[categorical_unordered_features])
    one_hot_encoded_df_test = pd.DataFrame(one_hot_encoded_array_test, index=df_test.index)
        
    df_train = pd.concat([df_train,one_hot_encoded_df_train], axis=1) #concatenate old columns with new one hot encoded columns
    df_test = pd.concat([df_test,one_hot_encoded_df_test], axis=1) #concatenate old columns with new one hot encoded columns

    df_train = df_train.drop(categorical_unordered_features, axis=1) #Delete columns which were one hot encoded
    df_test = df_test.drop(categorical_unordered_features, axis=1) #Delete columns which were one hot encoded

    ### Normalization or Standardization of Continuous Features
    if process_continuous == "Standardize":
        print "\n--------- Standardizing Continuous Features (Mean=0, Standard Deviation=1) --------- "
        standardization = StandardScaler()
        standardization.fit(df_train[continuous_features])
        df_train[continuous_features] = standardization.transform(df_train[continuous_features])
        df_test[continuous_features] = standardization.transform(df_test[continuous_features])

    elif process_continuous == "Normalize":
        print "\n--------- Normalizing Continuous Features (Min=0, Max=1) --------- "
        min_max_scaling = MinMaxScaler()
        min_max_scaling.fit(df_train[continuous_features])
        df_train[continuous_features] = min_max_scaling.transform(df_train[continuous_features])
        df_test[continuous_features] = min_max_scaling.transform(df_test[continuous_features])

    ### Return pre-processed df
    return df_train, df_test

def preprocessing_class(df_train, df_test):
    "\n--------- Labelling Class Information --------- "
    class_col = 'y'
    label_dict = {class_col:{'no':0,'yes':1}}
    df_train = df_train.replace({class_col:label_dict[class_col]})
    df_test = df_test.replace({class_col:label_dict[class_col]})
    return df_train, df_test
    
'''                           
###
df = pd.read_csv('bank-additional.csv')

df_class = pd.DataFrame(df['y'])
df_class = preprocessing_class(df_class)

df_features = df.loc[:, df.columns != 'y']
df_features = preprocessing_features(df_features,"Normalize")

df_preprocessed = pd.concat([df_features, df_class], axis=1)
df_preprocessed.to_csv('bank-additional-preprocessed-mode-normalize.csv')
'''