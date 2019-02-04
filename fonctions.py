# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:22:53 2019

@author: Arthur Delaitre
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.integrate import simps
from sklearn.model_selection import cross_val_score
import tqdm
import xgboost as xgb
from sklearn.metrics import auc,accuracy_score,fbeta_score
from random import gauss

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
    
def plot_kde_corrected(df_X,df_X_submission,sorted_cols=None,kde_coef = 0.01,save=False,labels=None):
    """Plot the corrected distributions of the two dataframes. The correction is col = col/(abs(col)+1)**0.5 ."""
    a=1
    if labels is None:
        labels=sorted_cols
    if sorted_cols in None:
        sorted_cols=range(len(df_X.columns))
    for i in range(len(sorted_cols)):

        plt.figure(figsize=(20,5))
        (df_X_submission[df_X_submission.columns[i]]/(np.abs(df_X_submission[df_X_submission.columns[i]]).add(1)**0.5)).plot.kde(kde_coef,xlim=(-3,3),label="test")
        (df_X[df_X.columns[i]]/(np.abs(df_X[df_X.columns[i]]).add(1)**0.5)).plot.kde(kde_coef,xlim=(-3,3),label="train")
        plt.title(str(labels[i])+' - Rang d\'importance : '+str(a))
        plt.legend()
        if save:
            plt.savefig('distribs/distributions_'+str(a)+'_var_'+str(i)+'.pdf')
        a+=1

def plot_central_kde(df_X,df_X_submission,sorted_cols=None,kde_coef = 0.01,save=False,labels=None):
    """Plot the distributions of the two dataframes, centered and in the limits -3;+3 ."""
    a=1
    
    if sorted_cols is None:
        sorted_cols=df_X.columns
    if labels is None:
        labels=sorted_cols
    for i in range(len(sorted_cols)):

        plt.figure(figsize=(20,5))
        col = sorted_cols[i]
        df_X_submission[col][np.abs(df_X_submission[col])<3].plot.kde(kde_coef,xlim=(-3,3),label="test")
        col = sorted_cols[i]
        df_X[col][np.abs(df_X[col])<3].plot.kde(kde_coef,xlim=(-3,3),label="train")
        plt.title(str(labels[i])+' - Rang d\'importance : '+str(a))
        plt.legend()
        if save:
            plt.savefig('distribs/distributions_central_'+str(a)+'_var_'+str(i)+'.pdf')
        a+=1

def fix_distrib(first_distrib,second_distrib,limits=(-3,3),kde_coef=0.01):
    eval_points = np.arange(limits[0],limits[1],0.001)
    full = pd.concat([first_distrib,second_distrib],axis=1)
    plt.figure(figsize=(20,5))
    first_kde = first_distrib[np.abs(first_distrib)<3].plot.kde(kde_coef,ind=eval_points)
    second_kde = second_distrib[np.abs(second_distrib)<3].plot.kde(kde_coef,ind=eval_points)
    #plt.show()
    #kdes = full.plot.kde(0.01,ind=eval_points)
    line_1 = first_kde.lines[-1]
    line_2 = second_kde.lines[-1]
    data_1 = line_1.get_ydata()
    data_2 = line_2.get_ydata()
    
    area_sum = simps(data_2)+simps(data_1)
    
    conv = np.convolve(data_1,np.flip(data_2,axis=0),mode='same')
    area_conv = max(conv)
    ind_max = np.argmax(conv)
    first_distrib_copy = first_distrib.copy() - eval_points[ind_max]
    
    third_kde = first_distrib_copy[np.abs(first_distrib_copy)<3].plot.kde(kde_coef/2,ind=eval_points)
    line_3 = third_kde.lines[-1]
    data_3 = line_3.get_ydata()
    data_min = [min(data_3[k],data_2[k]) for k in range(len(data_2))]
    data_max = [max(data_3[k],data_2[k]) for k in range(len(data_2))]
    iou = simps(data_min)/simps(data_max)
    ratio = iou
    
    return first_distrib_copy,ratio

def distrib_ratio(first_distrib,second_distrib,limits=(-3,3),kde_coef=0.01):
    eval_points = np.arange(limits[0],limits[1],0.001)
    first_kde = first_distrib[np.abs(first_distrib)<3].plot.kde(kde_coef,ind=eval_points)
    second_kde = second_distrib[np.abs(second_distrib)<3].plot.kde(kde_coef,ind=eval_points)
    #plt.show()
    #kdes = full.plot.kde(0.01,ind=eval_points)
    line_1 = first_kde.lines[-1]
    line_2 = second_kde.lines[-1]
    data_1 = line_1.get_ydata()
    data_2 = line_2.get_ydata()

    data_min = [min(data_1[k],data_2[k]) for k in range(len(data_2))]
    data_max = [max(data_1[k],data_2[k]) for k in range(len(data_2))]
    iou = simps(data_min)/simps(data_max)
    ratio = iou
    
    return ratio

def transform_ratios(ratios):
    ratios = [(max(0,elt-min(ratios)))/(max(ratios)-min(ratios)) for elt in ratios]
    ratios = [elt**2 for elt in ratios]
    #print(ratios)
    return ratios

def compute_best_feature(improvements,ratios):
    score = [(max((improvements[k]*1000),1)**0.8-1)*ratios[k]/1000 for k in range(len(improvements))]
    ind = np.argmax(score)
    return ind,score[ind]

def compute_distrib_ratios(df_1,df_2,columns=None,transform_=True):
    if columns is None:
        columns=df_1.columns
    new_df_1 = df_1.copy()
    ratios = []
    for col in tqdm.tqdm(columns):
        
        new_col,ratio = fix_distrib(df_1[col],df_2[col])
        #print(col,ratio)
        ratios.append(ratio)
        new_df_1[col] = new_col
    
    if transform_:
        ratios = transform_ratios(ratios)
        
    return new_df_1,ratios

def compute_ratios(df_1,df_2,columns=None,transform_=True):
    if columns is None:
        columns=df_1.columns
    ratios = []
    for col in tqdm.tqdm(columns):
        
        _,ratio = distrib_ratio(df_1[col],df_2[col])
        #print(col,ratio)
        ratios.append(ratio)
    
    if transform_:
        ratios = transform_ratios(ratios)
        
    return ratios

def select_features(df_train,df_labels,df_2,base_columns,list_models,thresh = 0.01,ratios=True,decal=True,scoring='accuracy'):
    columns_selected = base_columns
    print("Calcul des ratios de distribution")
    new_df_train, distrib_ratios = compute_distrib_ratios(df_train,df_2)
    if not ratios:
        distrib_ratios = np.ones(len(distrib_ratios))
    if not decal:
        new_df_train = df_train
    boucle = True
    while boucle:
        print("Choix de la colonne à ajouter")
        improvements = improvements_made_by_features(list_models,new_df_train,df_labels,columns_selected,scoring=scoring)
        ind_best, score = compute_best_feature(improvements,distrib_ratios)
        if score > thresh :
            columns_selected.append(df_train.columns[ind_best])
            print("Colonne "+str(df_train.columns[ind_best])+" ajoutée")
        else :
            boucle = False
    return new_df_train,columns_selected

def select_and_modify(df_train,df_y,df_test,base_columns,columns_to_consider,list_models,ratios=True,decal=True,thresh=0.01,scoring='accuracy'):
    df_train_sel = df_train[columns_to_consider]
    df_test_sel = df_test[columns_to_consider]
    new_df_train,cols_selected = select_features(df_train_sel,df_y,df_test_sel,base_columns=base_columns,list_models=list_models,ratios=ratios,thresh=thresh,scoring=scoring,decal=decal)
    new_df_train = new_df_train[cols_selected]
    new_df_test = df_test_sel[cols_selected]
    return new_df_train,new_df_test

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
    


def cv_score(model,df_train,df_label,scoring):
    """Use the model to predict on the df_test."""
    return np.mean(cross_val_score(model,df_train,df_label,cv=5,scoring=scoring))

def diff_cv_score(model,df_train_first,df_train_second,df_label,scoring):
    """Return the improvement made with df_train_second compared to df_train_first"""
    score = cv_score(model,df_train_second,df_label,scoring) - cv_score(model,df_train_first,df_label,scoring)
    return score

def improvements_made_by_features(list_models,df_train,df_label,base_columns,scoring,use_tqdm=False,verbose=True):
    """Return the improvement in accuracy made by adding the features to the dataset."""
    improvements=[]
    base_scores = []
    for model in list_models:
        score = cv_score(model,df_train[base_columns],df_label,scoring)
        base_scores.append(score)
    base_score = np.mean(base_scores)
    if verbose:
        print("Score avant sélection d'une nouvelle variable : "+str(base_score))
    for col in tqdm.tqdm(df_train.columns):
        if col in base_columns:
            improvements.append(0)
        else:
            cv_scores = []
            new_cols = list(base_columns).copy()
            new_cols.append(col)
            for model in list_models:
                diff_score = cv_score(model,df_train[new_cols],df_label,scoring) - base_score
                cv_scores.append(diff_score)
            improvements.append(np.mean(cv_scores))
    return improvements

def score_and_pred_of_xgboost(df_train,df_y,params,df_test = None,df_test_y=None,obj=None,n_estimators = 10):
    dtrain = xgb.DMatrix(df_train, label=df_y)
    if obj is None:
        cv = xgb.cv(params,dtrain,nfold=4)
    else:
        cv = xgb.cv(params,dtrain,nfold=4,obj=obj)
    print(cv)
    x_train,x_test,y_train,y_test = train_test_split(df_train,df_y,test_size = 0.25)
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_test = xgb.DMatrix(x_test)
    if obj is None:
        bst_1 = xgb.train(params, d_train, n_estimators)
    else:
        bst_1 = xgb.train(params, d_train, n_estimators, obj=f2_loss)
    
    prediction_1 = bst_1.predict(d_test)
    
    plt.hist(prediction_1,bins=100)
    val_reelle_1 = y_test
    #print("Area under ROC curve : "+str(auc(val_reelle,prediction)))
    accs,fbeta_scores=[],[]
    test_vals = np.arange(-0.1,1.1,0.03)
    for threshold in test_vals:
        bin_pred = np.where(prediction_1>threshold,1,0)
        #print("Threshold : "+str(threshold))
        acc = accuracy_score(val_reelle_1,bin_pred)
        #print("Accuracy : "+str(acc))
        fbeta = fbeta_score(val_reelle_1,bin_pred,2)
        #print("F2 score : "+ str(fbeta))
        accs.append(acc)
        fbeta_scores.append(fbeta)
    plt.figure()
    plt.plot(test_vals,accs,label="Accuracy")
    plt.plot(test_vals,fbeta_scores,label = "f2_score")
    plt.title("Train - Max f2 : "+str(max(fbeta_scores)))
    plt.legend()
    plt.show()
    
    if df_test is not None:
        dtest = xgb.DMatrix(df_test)
        if obj is None:
            bst = xgb.train(params, dtrain, n_estimators)
        else:
            bst = xgb.train(params, dtrain, n_estimators, obj=f2_loss)
        prediction = bst.predict(dtest)
        print(bst.get_fscore())
        plt.hist(prediction,bins=100)
        if df_test_y is None:
            plt.show()
            return prediction
        else:
            val_reelle = df_test_y.values.reshape((len(df_test_y),))
            #print("Area under ROC curve : "+str(auc(val_reelle,prediction)))
            accs,fbeta_scores=[],[]
            test_vals = np.arange(-0.1,1.1,0.03)
            for threshold in test_vals:
                bin_pred = np.where(prediction>threshold,1,0)
                #print("Threshold : "+str(threshold))
                acc = accuracy_score(val_reelle,bin_pred)
                #print("Accuracy : "+str(acc))
                fbeta = fbeta_score(val_reelle,bin_pred,2)
                #print("F2 score : "+ str(fbeta))
                accs.append(acc)
                fbeta_scores.append(fbeta)
            plt.figure()
            plt.plot(test_vals,accs,label="Accuracy")
            plt.plot(test_vals,fbeta_scores,label = "f2_score")
            plt.title("Test - Max f2 : "+str(max(fbeta_scores)))
            plt.legend()
            plt.show()
            return prediction
        
def decalage_mediane(df_1,df_2):
    medians_df_1 = df_1.median().values
    medians_df_2 = df_2.median().values
    new_df_2 = df_2.copy()
    cols = df_1.columns
    for i in range(len(cols)):
        new_df_2[cols[i]] = df_2[cols[i]] - medians_df_2[i] + medians_df_1[i]
    
    return df_2

def normal_quantiles(df_1,df_2,cut=0.25):
    new_df_2 = decalage_mediane(df_1,df_2)
    
    medians_df_1 = df_1.median().values
    q25_df_1 = df_1.quantile(q=cut).values
    q75_df_1 = df_1.quantile(q=1-cut).values

    medians_new_df_2 = new_df_2.median().values
    q25_new_df_2 = new_df_2.quantile(q=cut).values
    q75_new_df_2 = new_df_2.quantile(q=1-cut).values
    
    cols = df_1.columns
    for i in range(len(cols)):
        col_for_sup = q75_df_1[i]-(q75_df_1[i]-medians_df_1[i])*(q75_new_df_2[i]-new_df_2[cols[i]])/(q75_new_df_2[i]-medians_new_df_2[i])
        col_for_inf = medians_df_1[i]-(q25_df_1[i]-medians_df_1[i])*(medians_new_df_2[i]-new_df_2[cols[i]])/(q25_new_df_2[i]-medians_new_df_2[i])
        new_df_2[cols[i]] = np.where(new_df_2[cols[i]]>medians_new_df_2[i],col_for_sup,col_for_inf)
    
    return new_df_2

def add_gaussian_noise(df,sigma):
    new_df = df.copy()
    n = len(df)
    cols = df.columns
    for col in cols:
        new_df[col] = df[col]+[gauss(0,sigma) for k in range(n)]
    return new_df