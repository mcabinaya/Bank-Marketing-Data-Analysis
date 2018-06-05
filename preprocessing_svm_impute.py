#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:25:04 2018

@author: abinaya
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def preprocessing_features(df_train, df_test, process_continuous):
    
    to_delete_features = ['default','pdays']
    continuous_features = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_ordered_features = ['education', 'housing', 'loan', 'contact','month', 'day_of_week','poutcome']
    categorical_unordered_features = ['job', 'marital']
    
    unknown_present_features = ['job','marital','education','housing','loan']
    all_present_features = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'contact', 'month', 'day_of_week','poutcome']
    
    ### Delete Features
    for feat in to_delete_features:
        print "\n--------- deleting feature --------- ",feat
        del df_train[feat]
        del df_test[feat]
        
    ### Normalization or Standardization of Continuous Features
    if process_continuous == "Standardize":
        print "\n--------- Standardizing Continuous Features (Mean=0, Standard Deviation=1) --------- "
        standardization = StandardScaler()
        standardization.fit(df_train[continuous_features])
        print "Mean: ",standardization.mean_
        print "Variance: ",standardization.var_
        df_train[continuous_features] = standardization.transform(df_train[continuous_features])
        df_test[continuous_features] = standardization.transform(df_test[continuous_features])
    
    elif process_continuous == "Normalize":
        print "\n--------- Normalizing Continuous Features (Min=0, Max=1) --------- "
        min_max_scaling = MinMaxScaler()
        min_max_scaling.fit(df_train[continuous_features])
        print min_max_scaling.data_min_
        print min_max_scaling.data_max_
        df_train[continuous_features] = min_max_scaling.transform(df_train[continuous_features])
        df_test[continuous_features] = min_max_scaling.transform(df_test[continuous_features])
    
        
    ### Label Categorical Ordered Features -- Features used for Imputation (All Present)
    label_dict = {'education':{'illiterate':0, 'basic.4y':4, 'basic.6y':6, 'basic.9y':9, 'high.school':11, 'professional.course':13, 'university.degree':14},
                  'housing':{'no':0,'yes':1},
                  'loan':{'no':0,'yes':1},
                  'contact':{'telephone':0,'cellular':1},
                  'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12},
                  'day_of_week':{'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7},
                  'poutcome':{'nonexistent':0,'failure':1,'success':2}}
    
    for feat in categorical_ordered_features:
        if feat not in unknown_present_features:
            print "\n--------- Labelling feature Before Imputation --------- ",feat
            df_train = df_train.replace({feat:label_dict[feat]})
            df_test = df_test.replace({feat:label_dict[feat]})
            print "Labelled as: ",label_dict[feat]
         
    ### Imputation using SVM
    df_train_impute = df_train.loc[:,df_train.columns.isin(all_present_features)]
    df_test_impute = df_test.loc[:,df_test.columns.isin(all_present_features)]
    
    for feat in unknown_present_features:
        print "\nFilling Unkowns for Feature: ",feat
        train_impute = df_train[feat]
        test_impute = df_test[feat]
        
        train_impute_no_unknowns = train_impute[train_impute != 'unknown']
        train_impute_unknowns = train_impute[train_impute == 'unknown']
        test_impute_unknowns = test_impute[test_impute == 'unknown']
        
        df_train_impute_train_features = df_train_impute.loc[train_impute_no_unknowns.index]
        df_train_impute_test_features = df_train_impute.loc[train_impute_unknowns.index]
        df_test_impute_test_features = df_test_impute.loc[test_impute_unknowns.index]
        
        svm_model = SVC()
        svm_model.fit(df_train_impute_train_features, train_impute_no_unknowns) 
        df_train.loc[df_train_impute_test_features.index, feat] = svm_model.predict(df_train_impute_test_features)
        print "Train Filled with: ",df_train.loc[df_train_impute_test_features.index, feat].value_counts()
        df_test.loc[df_test_impute_test_features.index, feat] = svm_model.predict(df_test_impute_test_features)
        print "Test Filled with: ", df_test.loc[df_test_impute_test_features.index, feat].value_counts()
    
    ### Label Categorical Ordered Features -- Features Imputated (Unkowns were Present)
    for feat in categorical_ordered_features:
        if feat in unknown_present_features:
            print "\n--------- Labelling feature After  Imputation --------- ",feat
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
    
    ### Return pre-processed df
    return df_train, df_test

def preprocessing_class(df_train, df_test):
    "\n--------- Labelling Class Information --------- "
    class_col = 'y'
    label_dict = {class_col:{'no':0,'yes':1}}
    df_train = df_train.replace({class_col:label_dict[class_col]})
    df_test = df_test.replace({class_col:label_dict[class_col]})
    return df_train, df_test
  
    
    
