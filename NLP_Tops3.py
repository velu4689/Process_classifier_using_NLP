# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:20:12 2018

@author: velmurugan.m
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from nltk import word_tokenize
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DATA_LOCATION = 'D:\Winstream_data\TOPS\FAS_6.xlsx'


########################## Read Input Data & Format the same ###############

def readFormatData(DATA_LOCATION):
    ### Read Data from Excel Sheet ###
    orgDataFrame = pd.read_excel(DATA_LOCATION, sheetname='Analysed data').iloc[:2002]    
    
    ### Extract Data Frame for Response & Response Action ###
    respData = orgDataFrame[orgDataFrame.columns[5]]
    respActData = orgDataFrame['Manual Action taken'] 

    ### Form a new Data Frame with the required fields ###   
    orgDict = zip(respData, respActData)
    orgDataFrame = pd.DataFrame(orgDict, columns = ['Resp_Data','Resp_Action_Data'])
    
    ### Append Label Num IDs for the Target Class Labels ###
    orgDataFrame['label_num'] = orgDataFrame.Resp_Action_Data.map({
                                                                    'Check & work the order':1, 
                                                                    'Check the Billing/Dir/Order for active TN #':2,
                                                                    'Create the Record in 2nd Drop':3,
                                                                    'Disconnect & Create Record in the location':4,
                                                                    'Hold the order until response':5,
                                                                    'Proceed with Change Process ':6,
                                                                    'Cancel the order':7
                                                             })
    
    print orgDataFrame.Resp_Action_Data.value_counts()
    
    lookup_dict = {
                    'svc'   : 'service', 
                    'svr'   : 'service', 
                    'cus'   : 'customer', 
                    'cust'  : 'customer',                     
                    'actv'  : 'active',
                    '2nd'   : 'second',
                    'addtl' : 'additional',
                    'autod' : 'autodialer',
                    'ialer' : 'dialer',
                    'auto-dialer': 'autodialer',
                    'rdy'   : 'ready',   
                    'loc'   : 'location',
                    'dup'   : 'duplicate'                   
                  }
    
    
    respData  = orgDataFrame.Resp_Data         
    labelData = orgDataFrame.label_num
            
    newRespData = []
    
    for line in respData:
        tokens = word_tokenize( (str(line )).lower() ) 
        tokens = tokens[2:]
        
        newVal = map( lambda val: lookup_dict[val] if val in lookup_dict else val, tokens )    
        
        remDate = filter( lambda ThisWord: not re.match('^(?:(?:[0-9]{1,2}[:\/,]){1,2}[0-9]{1,4})$', ThisWord), newVal)
        
        remInt = filter( lambda ThisWord: not re.match('^(\d{1,10}|\d{12})$', ThisWord), remDate)  
        
        remSplCh = filter( lambda ThisWord: not re.match('[^ a-zA-Z0-9]', ThisWord), remInt)  
        
        remDat = filter( lambda ThisWord: not re.match("(u')", ThisWord), remSplCh)         
        
        newLine = " ".join(remDat)
    
        newRespData.append(newLine)    
    
    #newRespDataSeries = " ".join(newRespData)    
    newRespDataSeries = pd.Series( newRespData )
    
    return (newRespDataSeries, labelData)

########################## Apply NB-Sklearn ##########################
    
def NBClassSklearn(rData, lData):    
    respTrain, respTest, labTrain, labTest = train_test_split(rData, lData, random_state=1)
    
    #vect = CountVectorizer(stop_words='english')
    vect = TfidfVectorizer(min_df=1, max_df = 1.0, stop_words='english')    
    
    respTrainVec = vect.fit_transform(respTrain)    
    respTestVec = vect.transform(respTest)
    
    nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    nb.fit(respTrainVec, labTrain)    
    
    labPredClass = nb.predict(respTestVec)    
    
    metrics.confusion_matrix(labTest, labPredClass)
        
    accuracy = metrics.accuracy_score(labTest, labPredClass)
        
    return accuracy

########################## Apply Logistic Regression ##########################
def logReg(rData, lData):
    logreg = LogisticRegression()    
    
    respTrain, respTest, labTrain, labTest = train_test_split(rData, lData, random_state=1)
    
    #vect = CountVectorizer(stop_words='english')
    vect = TfidfVectorizer(min_df=1,stop_words='english')
    
    respTrainVec = vect.fit_transform(respTrain)    
    respTestVec = vect.transform(respTest)
    
    logreg.fit(respTrainVec, labTrain)
    
    labPredClass = logreg.predict(respTestVec)                                     
    
    #y_pred_prob = logreg.predict_proba(respTestVec)[:, 1]    
    
    return (metrics.accuracy_score(labTest, labPredClass))

########################## Apply Logistic Regression ##########################
def randFor(rData, lData):    
    randClass = RandomForestClassifier(n_estimators = 100)
    
    respTrain, respTest, labTrain, labTest = train_test_split(rData, lData, random_state=1)    
    
    vect = CountVectorizer(stop_words='english')
    #vect = TfidfVectorizer(min_df=1, max_df=1.0, stop_words='english')
    
    respTrainVec = vect.fit_transform(respTrain)    
    #print respTrainVec.shape
    
    #princy = PCA(n_components=2)
    ##princy.fit(respTrainVec.toarray())
    #X = princy.transform(respTrainVec.toarray())    
    #print X
    #plt.scatter(X[:,0], X[:,1], c=labTrain)
    #plt.show()
    
    respTestVec = vect.transform(respTest)
    
    randClass.fit(respTrainVec, labTrain)
        
    labPredClass = randClass.predict(respTestVec)                                 
        
    #y_pred_prob = randClass.predict_proba(respTestVec)[:, 1]
    
    # examine class distribution
    #print(labTest.value_counts())
    #null_accuracy = labTest.value_counts().head(1) / len(labTest)
    #print('Null accuracy:', null_accuracy)
    
    # print the confusion matrix
    metrics.confusion_matrix(labTest, labPredClass)
    
    return (metrics.accuracy_score(labTest, labPredClass))

########################## Main Function ##########################
def main():
    respData, labelData = readFormatData(DATA_LOCATION)
    
    ### Naive Bayes Classifier from SKLEARN ###
    #NBAccuracy = NBClassSklearn(respData, labelData)
    #print NBAccuracy
    
    ### Logistic Regression from SKLEARN ###
    #logRegAccuracy = logReg(respData, labelData)
    #print logRegAccuracy
    
    ### Random Forest from SKLEARN ###
    randForAccuracy = randFor(respData, labelData)
    print randForAccuracy

if __name__ == "__main__":
    main()

'''
#trainData = list(orgDict[1:450])
#testData = list(orgDict[451:499])
   
#orgDataFrame = pd.DataFrame(orgDict, columns = ['Resp Data','Resp Action Data'])

#print len(respData)

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(respData[0:450])
#X_train_counts.shape

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf     = tf_transformer.transform(X_train_counts)
#X_train_tfidf.shape

#clf = MultinomialNB().fit(X_train_tfidf, respData[451:499])

#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])

#predicted = text_clf.predict(testData)
#np.mean(predicted == target)
'''