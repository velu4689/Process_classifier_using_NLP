# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:57:02 2018

@author: velmurugan.m
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold


DATA_LOCATION = r'D:\ATT\newData\p1\hourly_oem_db_features.csv'

def pcaAnalysis(data, target):    
    column_names = list(data.columns.values)    
    #print type(target)
    
    #data_std = StandardScaler().fit_transform(data)    
    
    pca = PCA(n_components=2)
    data_r = pca.fit_transform(data)#.transform(data)
    
    princiData = pd.DataFrame(data = data_r, columns = ['Prin Compo 1', 'Prin Compo 2'])#, 'Prin Compo 3', 'Prin Compo 4'])
    
    finalDf = pd.concat([ princiData, target ], axis = 1)    
        
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
    
    fig = plt.figure(figsize = (8,8))
    
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = [True,False]
    
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf.alert == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Prin Compo 1'], finalDf.loc[indicesToKeep, 'Prin Compo 2'], c = color, s = 50)
    
    #plt.figure()
    #colors = ['navy', 'orange']
    #lw = 2    
    #for color, i, target_name in zip(colors, [0, 1], target_names):
    #    plt.scatter(data_r[target == i, 0], data_r[target == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
        
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('PCA of ATT dataset')    
    #plt.show()
    
def ldaAnalysis(data, target):    
    lda = LinearDiscriminantAnalysis(n_components=2)       
    
    data_r2 = lda.fit(data, target).transform(data)
    
    princiData = pd.DataFrame(data = data_r2, columns = ['LDA Compo 1'])#, 'LDA Compo 2'])
    finalDf = pd.concat([ princiData, target ], axis = 1)    

    target_names = [True, False]
    colors = ['navy', 'orange']
    
    fig = plt.figure(figsize = (8,8))
    
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('LDA Component 1', fontsize = 15)
    ax.set_ylabel('LDA Component 2', fontsize = 15)
    ax.set_title('2 component LDA', fontsize = 20)
    
    targets = [True,False]
    
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf.alert == target
        ax.scatter(finalDf.loc[indicesToKeep, 'LDA Compo 1'], finalDf.loc[indicesToKeep, 'LDA Compo 2'], c = color, s = 50)
    
    #plt.figure()    
    #for color, i, target_name in zip(colors, [1, 0], target_names):
    #    print data_r2
    #    plt.scatter(data_r2[target == i, 1], data_r2[target == i, 0], alpha=.8, color=color, label=target_name)
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('LDA of ATT dataset')
    #plt.show()
    
def featureSelect(data, target):    

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV()
    
    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(data, target)
    n_features = sfm.transform(data).shape[1]
    
    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 2:
        sfm.threshold += 0.1
        X_transform = sfm.transform(data)
        n_features = X_transform.shape[1]

    
    # Plot the selected two features from X.
    plt.title(
        "Features selected from Boston using SelectFromModel with "
        "threshold %0.3f." % sfm.threshold)
    feature1 = X_transform[:, 0]
    feature2 = X_transform[:, 1]
    plt.plot(feature1, feature2, 'r.')
    plt.xlabel("Feature number 1")
    plt.ylabel("Feature number 2")
    plt.ylim([np.min(feature2), np.max(feature2)])
    plt.show()
    
def randomLassoFeatSelect(data, target):    
    column_names = list(data.columns.values)
    
    rlasso = RandomizedLasso(alpha=0.1)
    rlasso.fit(data, target)
 
    print "Features sorted by their score:"
    print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), column_names), reverse=True)

def recurseFeatEliminate(data, target):
    column_names = list(data.columns.values)
    
    #use linear regression as the model
    lr = LinearRegression()
    #rf = RandomForestRegressor()
    #rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=3)
    rfe.fit(data,target)
 
    #print "Features sorted by their rank: %s" % sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), column_names))
    
    sortedFeatSet = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), column_names))
    selected_features = [ x[1] for x in sortedFeatSet ][:15]    
    return selected_features

def randForestRegressor(data, target):    
    column_names = list(data.columns.values)
    rf = RandomForestRegressor()
    rf.fit(data, target)
    print "Features sorted by their score:"
    
    sortedFeatSet = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), column_names), reverse=True)
    selected_features = [ x[1] for x in sortedFeatSet ][15:]    
    return selected_features


def dataPreProcessingNB(data, target):
    column_names = list(data.columns.values)          
    scaled_features = {}
    
    for col in column_names:
        mean, std = data[col].mean(), data[col].std()
        scaled_features[col] = [mean, std]
        data.loc[:, col] = ( data[col] - mean)/std
                
    return data                

def NBClassifier(data, target):
    
    #pData = dataPreProcessingNB(data, target)
    
    train, test, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state = 10)
    gnb = GaussianNB()
    model = gnb.fit(train, train_labels)
    
    preds = gnb.predict(test)
    NBConfMatrix = confusion_matrix(test_labels, preds)    
    #print(preds)
    print(accuracy_score(test_labels, preds))
    print NBConfMatrix

def RandForClassifier(data, target):    
    train, test, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state = 10)
    #clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf = RandomForestClassifier(max_depth=10,n_estimators=10)
    clf.fit(train, train_labels)
    
    preds = clf.predict(test)
    RFConfMatrix = confusion_matrix(test_labels, preds)   
    print(accuracy_score(test_labels, preds))
    print RFConfMatrix

def featVarianceEval(data):
    iColNames = (data.columns.values)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    
    sel.fit_transform(data)
    oColNames = iColNames[ ~sel.get_support() ]        
    return oColNames
        
def dataProcessing():    
    rawData = pd.read_csv(DATA_LOCATION)   

    print rawData.columns.values[:6]    
    print rawData.columns.values[7:13]
    print rawData.columns.values[14:20]
    print rawData.columns.values[21:]
    
    data   = rawData.drop(['alert', 'collection_timestamp', 'target_guid'], axis=1)    
    #print data.isnull().sum()
    
    target = (rawData['alert'])#.astype(int)    
    #data   = rawData.loc[:, rawData.columns != ['alert', 'collection_timestamp']   ]            
    
    #pcaAnalysis(data, target)
    #ldaAnalysis(data, target)
    #featureSelect(data, target)      
    #randomLassoFeatSelect(data, target)
    #recurseFeatEliminate(data, target)
    #featVarianceEval(data)
        
    #selFeatureList = randForestRegressor(data, target)    
    #selFeatureList = featVarianceEval(data)    
    #data = data.drop( selFeatureList, axis=1 )
    
    # Gaussian Naive Bayes Classifier:
    #NBClassifier(data, target)   
    
    # Gaussian Naive Bayes Classifier:
    #RandForClassifier(data, target)
    
def main():
    dataProcessing()

if __name__ == '__main__':
    main()


