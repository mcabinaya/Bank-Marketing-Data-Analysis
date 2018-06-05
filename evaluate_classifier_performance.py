#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:29:33 2018

@author: abinaya
"""
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

def evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, df_test_class, predicted_test, predicted_prob_test, roc_y_n):
    ### Confusion Matrix
    confusion_matrix_train = confusion_matrix(df_train_class, predicted_train)
    confusion_matrix_test = confusion_matrix(df_test_class, predicted_test)
    print "\nTraining Confusion Matrix:\n ", confusion_matrix_train
    print "\nTesting Confusion Matrix:\n ", confusion_matrix_test
 
    ### Accuracy score
    score_train = accuracy_score(df_train_class, predicted_train)
    score_test = accuracy_score(df_test_class, predicted_test)
    print "\nTraining Accuracy Score: ", score_train
    print "\nTesting Accuracy Score: ", score_test
       
    ### Precision, Recall  
    precision_train = precision_score(df_train_class, predicted_train)
    precision_test = precision_score(df_test_class, predicted_test)
    print "\nTraining Precision: ", precision_train
    print "\nTesting Precision: ", precision_test
    
    recall_train = recall_score(df_train_class, predicted_train)
    recall_test = recall_score(df_test_class, predicted_test)
    print "\nTraining Recall: ", recall_train
    print "\nTesting Recall: ", recall_test
    
    ### Classification Report
    print "\nTrain Classification Report: \n",classification_report(df_train_class, predicted_train)
    print "\nTest Classification Report: \n",classification_report(df_test_class, predicted_test)

    ### F1 Score
    f1score_train = f1_score(df_train_class, predicted_train)#, average='weighted')
    f1score_test = f1_score(df_test_class, predicted_test)#, average='weighted')
    print "\nTraining F1score: ", f1score_train
    print "\nTesting F1score: ", f1score_test
    
    f1score_train = f1_score(df_train_class, predicted_train, average='weighted')
    f1score_test = f1_score(df_test_class, predicted_test, average='weighted')
    print "\nTraining Weigted F1score: ", f1score_train
    print "\nTesting Weighted F1score: ", f1score_test
    
    
    ### ROC-AUC
    if roc_y_n == 'y':
        fpr, tpr, threshold = roc_curve(df_train_class, predicted_prob_train[:,1])
        roc_auc_train = auc(fpr, tpr)
        print "\nTraining AUC for ROC: ",roc_auc_train
        plt.figure()
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_train)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc = 'lower right')
        plt.title('Training - Receiver Operating Characteristic')
        
        fpr, tpr, threshold = roc_curve(df_test_class, predicted_prob_test[:,1])
        roc_auc_test = auc(fpr, tpr)
        print "\nTesting AUC for ROC: ",roc_auc_test
        plt.figure()
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_test)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc = 'lower right')
        plt.title('Testing - Receiver Operating Characteristic')
