#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:15:49 2018

@author: abinaya
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.close("all")

class understanding_data:
    
    def __init__(self, raw_df):
        self.raw_df = raw_df
        self.raw_df_grouped = raw_df.groupby("y")
        self.class_name_no = "no"
        self.class_name_yes = "yes"
        self.raw_df_grouped_no = self.raw_df_grouped.get_group(self.class_name_no)
        self.raw_df_grouped_yes = self.raw_df_grouped.get_group(self.class_name_yes)

    def plot_histogram_continuous(self, feature_name, bin_size):
        plt.figure()
        plt.hist(self.raw_df_grouped_no[feature_name], bins=bin_size, label=self.class_name_no)
        plt.hist(self.raw_df_grouped_yes[feature_name], bins=bin_size, label=self.class_name_yes)
        plt.legend()
        plt.title("Feature Histogram - "+feature_name)
        plt.xlabel("Feature values")
        plt.ylabel("Count")

    def plot_histogram_categorical(self, feature_name):
        feature_df = pd.DataFrame()
        feature_df["no"] = self.raw_df_grouped_no[feature_name].value_counts()
        feature_df["yes"] = self.raw_df_grouped_yes[feature_name].value_counts()
        
        feature_df.plot(kind='bar')
        plt.title("Feature Histogram - "+feature_name)
        plt.ylabel("Count")
        plt.xlabel("Feature unique values")
        plt.tight_layout()
    
    

### Read csv and get grouped df based on class
raw_df = pd.read_csv('Data/bank-additional.csv')

understanding_data_obj = understanding_data(raw_df)

### Feature 1 - AGE
understanding_data_obj.plot_histogram_continuous("age", 50)

### Feature 2 - JOB
understanding_data_obj.plot_histogram_categorical("job")

### Feature 3 - MARITAL
understanding_data_obj.plot_histogram_categorical("marital")

### Feature 4 - EDUCATION
understanding_data_obj.plot_histogram_categorical("education")

### Feature 5 - DEFAULT
understanding_data_obj.plot_histogram_categorical("default")

### Feature 6 - HOUSING
understanding_data_obj.plot_histogram_categorical("housing")

### Feature 7 - LOAN
understanding_data_obj.plot_histogram_categorical("loan")

### Feature 8 - CONTACT
understanding_data_obj.plot_histogram_categorical("contact")

### Feature 9 - MONTH
understanding_data_obj.plot_histogram_categorical("month")

### Feature 10 - DAY OF WEEK
understanding_data_obj.plot_histogram_categorical("day_of_week")

### Feature 11 - CAMPAIGN
understanding_data_obj.plot_histogram_continuous("campaign", 30)

### Feature 12 - PDAYS
understanding_data_obj.plot_histogram_continuous("pdays", 30)

### Feature 13 - PREVIOUS
understanding_data_obj.plot_histogram_categorical("previous")

### Feature 14 - POUTCOME
understanding_data_obj.plot_histogram_categorical("poutcome")

### Feature 15 - emp.var.rate
understanding_data_obj.plot_histogram_continuous("emp.var.rate", 50)

### Feature 16 - cons.price.idx
understanding_data_obj.plot_histogram_continuous("cons.price.idx", 50)

### Feature 17 - cons.conf.idx
understanding_data_obj.plot_histogram_continuous("cons.conf.idx", 50)

### Feature 18 - euribor3m
understanding_data_obj.plot_histogram_continuous("euribor3m", 50)

### Feature 19 - nr.employed
understanding_data_obj.plot_histogram_continuous("nr.employed", 50)
