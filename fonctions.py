# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:22:53 2019

@author: Arthur Delaitre
"""
import pandas as pd
import numpy as np
import matplotlib as plt

def missing_values_table(df):
    """Creates a recap of missing values per columns"""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns

def delete_row(list_index,X,y=None):
    if y is None:
        return X.drop(list_index)
    else:
        return X.drop(list_index),y.drop(list_index)

def f2_loss(preds, dtrain):
    """Score adapted to a minimization of the F2_score for xgboost"""
    labels = dtrain.get_label()
    diff = (preds - labels).astype("float")
    grad = np.where(labels==1,diff*4,diff)
    hess = np.where(labels==1,np.ones(preds.shape)*4,np.ones(preds.shape))
    return grad, hess

def correlated_cols(df,threshold=0.5):
    """Creates a list of pairs of features that are correlated according to a threshold"""
    corr = df.corr()
    thresh = 0.5
    list_corr_couples = []
    corr_coefs= []
    for col in corr.columns:
        table_1 = corr[col]>thresh
        table_2 = corr[col]<-thresh
        table = corr[table_1 | table_2][col]
        indexes = np.array(table.index)
        for index in indexes:
            if index < col:
                list_corr_couples.append(np.array([col,index]))
                corr_coefs.append(corr[col][index])
    return list_corr_couples,corr_coefs

def plot_correlated_pairs(df,threshold = 0.5,saving_path = None):
    """Plots the pairwise graphs of correlated features according to a threshold"""
    list_corr_couples,corr_coefs = correlated_cols(df,threshold)
    n_plot = len(list_corr_couples)
    plt.figure(figsize=(20,n_plot*5))
    i=0
    for tuple_xy in list_corr_couples:
        i+=1
        col_x,col_y = tuple_xy
        x = df[[col_x]].values
        y = df[[col_y]].values
        plt.subplot(n_plot//2+n_plot%2,2,i)
        plt.scatter(x,y)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title("Coef de corrélation : "+str(corr_coefs[i-1]))
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()
