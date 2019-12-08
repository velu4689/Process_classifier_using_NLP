# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 17:25:02 2018

@author: velmurugan.m
"""

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

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords

import re
import pickle

import eli5
from IPython.core.display import display, HTML


#DATA_LOCATION = 'D:\Winstream_data\TOPS\FAS_6.xlsx'
DATA_LOCATION = 'D:\Winstream_data\TOPS\second\FAS - Order Correction response details_Updated_last.xlsx''


targetList = [
                'Check & work the order', 
                'Check the Billing/Dir/Order for active TN #',
                'Create the Record in 2nd Drop',
                'Disconnect & Create Record in the location',
                'Hold the order until response',
                'Proceed with Change Process ',
                'Cancel the order'        
              ]


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
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for line in respData:        
        tokens = word_tokenize( (str(line )).lower() ) 
        tokens = tokens[2:]
        
        newVal = map( lambda val: lookup_dict[val] if val in lookup_dict else val, tokens )    
        
        remDate = filter( lambda ThisWord: not re.match('^(?:(?:[0-9]{1,2}[:\/,]){1,2}[0-9]{1,4})$', ThisWord), newVal)
        
        remInt = filter( lambda ThisWord: not re.match('^(\d{1,10}|\d{12})$', ThisWord), remDate)  
        
        remSplCh = filter( lambda ThisWord: not re.match('[^ a-zA-Z0-9]', ThisWord), remInt)  
        
        remDat = filter( lambda ThisWord: not re.match("(u')", ThisWord), remSplCh)      
        
        filSen = [w for w in remDat if not w in stop_words]
        
        lemWord = map( wordnet_lemmatizer.lemmatize, filSen)
        
        newLine = " ".join(lemWord)
    
        newRespData.append(newLine)    
    
    newRespDataSeries = pd.Series( newRespData )
    
    return (newRespDataSeries, labelData)

########################## Apply Random Forest ##########################
def randFor(rData, lData):    
    randClass = RandomForestClassifier(n_estimators = 100)
    
    respTrain, respTest, labTrain, labTest = train_test_split(rData, lData, random_state=1)    
    
    vect = TfidfVectorizer(min_df=1, max_df=1.0, stop_words='english')        
    respTrainVec = vect.fit_transform(respTrain)    

    # To be commented for Pickle Building of Vectorizer
    respTestVec = vect.transform(respTest)
    
    randClass.fit(respTrainVec, labTrain)        
    # To be commented for Pickle Building of Rand Class Model
    labPredClass = randClass.predict(respTestVec)                                 
    
    #display(HTML(eli5.show_weights(randClass, top=5)))
    #print type(eli5.explain_prediction(randClass, respTest[0], vec=vect, target_names=targetList))
    #tDF = eli5.explain_prediction_df(randClass)
    #tDF1 = eli5.show_weights(randClass, vec=vect, target_names=targetList)
    #print type(eli5.show_prediction(randClass, respTest[0], vec=vect, target_names=targetList))     
    
    # Explain the Weights of this Estimator ----------------------------------
    #print eli5.explain_weights(randClass)
    print eli5.format_as_dataframes(eli5.show_weights(randClass))   
    print respTest[0]
    #prediction = eli5.explain_prediction (randClass, respTest[0], vec=vect, target_names=targetList, top=5)
    #weigths = eli5.explain_prediction (randClass, respTest[0], vec=vect, target_names=targetList, top=5)
    #print ( eli5.format_as_dataframes( weigths ) )
    
    # Modify to return specfic class types
    return (metrics.accuracy_score(labTest, labPredClass))


########################## Naive Bayes ######################################
def nltkNBayes(rData, lData):   
    dataSet = list(zip(rData, lData))
       
    all_words = set( word.lower() for passage in dataSet for word in word_tokenize(passage[0]))
        
    t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in dataSet]
    
    #trainData = t[:1500]
    #testData = t[1501:2000]
    
    classifier = nltk.NaiveBayesClassifier.train(t)
    #print classifier.show_most_informative_features(15)
    
    return classifier
    
    # INcase you do not want to create a pikel
    #return nltk.classify.accuracy(classifier, testData)  
    

########################## Main Function ##########################
def main():
    respData, labelData = readFormatData(DATA_LOCATION)

    ### Random Forest from SKLEARN ###
    randForAccuracy = randFor(respData, labelData)
    print randForAccuracy
    
    ### Naive-Bayes from NLTK ###
    #nltkNBayesAcc = nltkNBayes(respData, labelData)
    #filename = "D:\Windstream_ML\Models\WindTOPSNLTK_NB_Classifier.pkl"
    
    #filename = "D:\Windstream_ML\Models\WindTOPSTFIDF.pkl"

    #scalar_pickle = open(filename, 'wb')
    #pickle.dump(nltkNBayesAcc, scalar_pickle)
    #scalar_pickle.close() 
    
    #print nltkNBayesAcc

if __name__ == "__main__":
    main()