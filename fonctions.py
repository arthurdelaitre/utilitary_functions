# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:22:53 2019

@author: Arthur Delaitre
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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

def distinct_families(df):
    """Gives the different second level headers of a dataframe"""
    families = []
    for col in df.columns:
        if col[1] not in families:
            families.append(col[1])
    return np.array(families)

def select_from_family(df,family):
    """Selects columns matching the second level header given"""
    cols = []
    for col in df.columns:
        if col[1] == family:
            cols.append(col)
    return df[cols]

def create_polynomial_rows(df,order):
    """Creates a DataFrame with new columns containing polynomial values of the other columns with respect to the order"""
    columns = df.columns
    new_columns = list(columns)
    i=0
    for col in columns:
        i+=1
        col_name = None
        if hasattr(col,'__len__'):
            col_name = [str(col[0])+"**"+str(order),col[1]]
        else:
            col_name = str(col)+"**"+str(order)
        new_columns.append(col_name)
        df[str(i)] = df[col]**order
    df.columns=pd.MultiIndex.from_arrays(np.transpose(np.array(new_columns)),names=['col','family'])
    return df

def reconstruct_with_model(df,cols_to_predict,model,cols_to_ignore=None,return_score=False):
    """Reconstructs the nans of cols_to_predict by training a model on the dataframe containing the necessary informations."""
    df_nonan = df.loc[list(df[cols_to_predict].dropna(axis=0).index)]
    df_nan = df.drop(index=df[cols_to_predict].dropna(axis=0).index)
    
    df_nan_X = df_nan.drop(cols_to_predict,axis=1)
    if cols_to_ignore is not None:
        df_nonan_wocols = df_nonan.drop(cols_to_ignore,axis=1)
        df_nan_wocols = df_nan_X.drop(cols_to_ignore,axis=1)
    else:
        df_nonan_wocols = df_nonan
        df_nan_wocols = df_nan_X
    df_nonan_wonan = df_nonan_wocols.dropna(axis=0)
    df_nonan_y = df_nonan_wonan[cols_to_predict]
    df_nonan_X = df_nonan_wonan.drop(cols_to_predict,axis=1)
    
    
    #Eval the model
    X_train, X_test, y_train, y_test = train_test_split(df_nonan_X, df_nonan_y, test_size=0.20)
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    print("Score obtained on a test set : "+str(score))
    
    #Train the model on full dataset
    model.fit(df_nonan_X,df_nonan_y)
    df_pred_y=pd.DataFrame(model.predict(df_nan_wocols),index=df_nan_X.index)
    df_pred_y.columns=pd.MultiIndex.from_arrays(np.transpose(np.array(list(cols_to_predict))),names=['col','family'])
    df_nan = pd.concat([df_nan_X,df_pred_y],axis=1)
    df_final = pd.concat([df_nonan,df_nan])
    
    if return_score:
        return df_final.sort_index(),score
    return df_final.sort_index()

def reconstruct_line(df,lines,model):
    """Reconstructs some lines of df using a model trained on all df-lines"""
    df_train = df.loc[df.index.difference(pd.Index(lines))]
    new_lines=[]
    for line in lines:
        line_data = df.loc[[line]]
        if line_data.isnull().any().any():
            df_2 = pd.concat([df_train,line_data])
            df_2 =reconstruct_with_model(df_2,df_2.columns[np.where(line_data.isnull())[1]],model)
            new_lines.append(df_2.loc[[line]])
    new_lines.append(df_train)
    df_final = pd.concat(new_lines)
    return df_final.sort_index()

def reconstruct_line_with_nonan(df,lines,model):
    """Reconstructs sone lines of df using a model trained on clean data of the df (Samples with no nans)"""
    df_1 = df.dropna(axis=0)
    df_2 = df.loc[lines]
    df_full = pd.concat([df_1,df_2])
    reconstruct = reconstruct_line(df_full,lines,model)
    df_final=pd.concat([reconstruct,df.loc[(df.index.difference(df_1.index))&(df.index.difference(df_2.index))]])
    return df_final.sort_index()

def plot_importance(importances,labels,sort=True,n_plot=None):
    if sort:
        arr = np.transpose(np.array([importances,labels]))
        sorted_arr = sorted(arr,key=lambda arr : arr[0])
        new_arr = np.transpose(sorted_arr)
        importances = new_arr[0]
        labels = new_arr[1]
    if n_plot is None:
        n_plot = len(importances)
    plt.figure(figsize=(10,n_plot/2))
    plt.barh(range(n_plot),importances)
    plt.yticks(range(n_plot),labels)
    
def plot_kde_corrected(df_X,df_X_submission,sorted_cols=None,kde_coef = 0.01,save=False):
    a=1
    """Plot the corrected distributions of the two dataframes. The correction is col = col/(abs(col)+1)**0.5 ."""
    if sorted_cols in None:
        sorted_cols=range(len(df_X.columns))
    for i in sorted_cols:

        plt.figure(figsize=(20,5))
        (df_X_submission[df_X_submission.columns[i]]/(np.abs(df_X_submission[df_X_submission.columns[i]]).add(1)**0.5)).plot.kde(kde_coef,xlim=(-3,3),label="test")
        (df_X[df_X.columns[i]]/(np.abs(df_X[df_X.columns[i]]).add(1)**0.5)).plot.kde(kde_coef,xlim=(-3,3),label="train")
        plt.title(labels[0][i]+' - Rang d\'importance : '+str(a))
        plt.legend()
        if save:
            plt.savefig('distribs/distributions_'+str(a)+'_var_'+str(i)+'.pdf')
        a+=1

def plot_central_kde(df_X,df_X_submission,sorted_cols=None,kde_coef = 0.01,save=False):
    """Plot the distributions of the two dataframes, centered and in the limits -3;+3 ."""
    a=1
    if sorted_cols is None:
        sorted_cols=range(len(df_X.columns))
    for i in sorted_cols:

        plt.figure(figsize=(20,5))
        col = df_X_submission.columns[i]
        df_X_submission[col][np.abs(df_X_submission[col])<3].plot.kde(kde_coef,xlim=(-3,3),label="test")
        col = df_X.columns[i]
        df_X[col][np.abs(df_X[col])<3].plot.kde(kde_coef,xlim=(-3,3),label="train")
        plt.title(labels[0][i]+' - Rang d\'importance : '+str(a))
        plt.legend()
        if save:
            plt.savefig('distribs/distributions_central_'+str(a)+'_var_'+str(i)+'.pdf')
        a+=1