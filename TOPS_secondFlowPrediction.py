# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:23:13 2018

@author: velmurugan.m
"""

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
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords

import itertools

import re
import pickle

import eli5
from IPython.core.display import display, HTML


#DATA_LOCATION = 'D:\Winstream_data\TOPS\second\FAS - Order Correction response_2nd Level Action_V1.xlsx'
#DATA_LOCATION = 'D:\Winstream_data\TOPS\second\FAS - Order Correction response details_Updated_last.xlsx'
DATA_LOCATION = r'C:\Users\velmurugan.m\Desktop\FAS - Order Correction response details_Updated.xlsx'


ManActList = [
                'No Action required',
                'Create the order next service in same location',
                'Process as Install order',
                'Process the order with the input address',
                'Process as TOA',
                'Check for Billing & Active location steps',
                'Disconnect the O order & Active the customer',
                'Order already completed & Check the order',
                'Process as Move order',
                'Process the order with the input Unit/LOT #',
                'Order already cancelled & Check the order',
                'Process the order with New TN  #',
                'Check Billing & Disconnect the location',
                'Process the O order first & Process other orders',
                'Process the C order first & Process other orders',
                'Check if loc is Active in MIROR, Hold the order',
                'Check for OC provided',
                'Process ORCAN Process',
                'Process Re-establishment process'	
              ]

RPAInpList = [  'APT',
                'Address',
                'House',
                'LOT',
                'OC',
                'ORCAN',
                'Order',
                'ROOM',
                'Unit',
                'nan'
             ]

##### Confusion #####################
'''
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names)
'''
########################## Data Analysis ###############

def detailedDataAnalysis(df):     
        
    #df['a'] = df['b'].str.contains('|'.join(titles)).astype(int)
    #test = pd.np.where(df.inp_Data.str.contains( '|'.join(inpRPADataList)) )
    
    #print (df.inp_Data[:5]).astype(str)
    
    ### Unique Value Counts of Response & Second Action Data ###
    #respValCounts = df.label_num.value_counts()    
    #print respValCounts
    #respValCounts.plot.bar(rot=0)
    
    #secValCounts = df.sec_label_num.value_counts()
    #print secValCounts
    #secValCounts.plot.bar(rot=0)    
    
    #print df['inp_Data'].isnull().sum()
    
    #reqData = df['label_num'][~df['inp_Data'].isnull()]    
    #print df['label_num'].value_counts()
    #print reqData.value_counts()
    #print df['inp_Data'].str.contains("Order")    
    #print df['inp_Data'].str.contains("Unit")
    #print df['inp_Data'].str.contains("Address")
    #print df['inp_Data'].str.contains("ORCAN")
    #print df['inp_Data'].str.contains("APT")
    #validData =  df.groupby(['label_num'])    
    #grp1 = validData.indices[0]
    #keys = validData.groups.keys()
    #print validData['inp_Data'].count()    
    '''print validData.get_group(1).isnull().sum()
    print validData.get_group(2).isnull().sum()
    print validData.get_group(3).isnull().sum()
    print validData.get_group(4).isnull().sum()
    print validData.get_group(5).isnull().sum()
    print validData.get_group(6).isnull().sum()
    print validData.get_group(7).isnull().sum() '''


########################## Form Multi-Label for predicting Next-Action ########
def formDataMultiLabel(sampData, df):
    
    ####### Convert Response Category into Labelized Binarizer ################
    manActData = df.label_num.unique()
    lb = preprocessing.LabelBinarizer()    
    lb.fit(manActData)                       
    
    tfdLabelNum = lb.transform(df.label_num)
    
    ####### Convert Next Action Category into Labelized Binarizer #############
    nxtActData = df.sec_label_num.unique()
    lb = preprocessing.LabelBinarizer()    
    lb.fit(nxtActData)
    
    tfdSecLabelNum = lb.transform(df.sec_label_num)
       
    ####### Convert Response Category into Labelized Binarizer ################
    inpRPAData     = (df.inp_Data).astype(str)
    inpRPAData     = inpRPAData.apply(lambda x: x.split()[0])
    lab, lev = pd.factorize(inpRPAData)
    
    lb = preprocessing.LabelBinarizer()    
    lb.fit(np.unique(lab))
    
    tfdInpRPAData = lb.transform(lab)
    #print (np.unique(tfdInpRPAData))
    
    #This concatenation is the actual process
    #conCatData    = np.concatenate((tfdLabelNum, tfdSecLabelNum, tfdInpRPAData), axis=1)
    
    ####### Build Multi-Label Prediction Model  ###############################
    respTrain, respTest, labTrain, labTest = train_test_split(sampData, tfdSecLabelNum, random_state=1)

    TR  = tree.DecisionTreeClassifier(criterion = "gini", max_depth=100, min_samples_leaf=2) 
    GNB = GaussianNB()
    RF  = RandomForestClassifier(n_estimators = 100)
    
    classifier = BinaryRelevance(GNB)
    #classifier = ClassifierChain(TR)
    #classifier = LabelPowerset(RF)
    
    vect = TfidfVectorizer(min_df=1, max_df=1.0, stop_words='english')
    respTrainVec = vect.fit_transform(respTrain)
    
    respTestVec = vect.transform(respTest)
    
    classifier.fit(respTrainVec, labTrain)
    predictions = classifier.predict(respTestVec)
    acc = metrics.accuracy_score(labTest, predictions)
    print (acc)
    
    return lab
        
########################## Read Input Data & Format the same ###############
def readFormatData(DATA_LOCATION):
    
    ### Read Data from Excel Sheet ###
    orgDataFrame = pd.read_excel(DATA_LOCATION, sheet_name='Analysed data').iloc[:1997]    
    
    ### Extract Data Frame for Response & Response Action ###
    respData    = orgDataFrame[orgDataFrame.columns[5]]
    respActData = orgDataFrame['Manual Action taken'] 
    respSecData = orgDataFrame['2nd Step']
    inpData     = orgDataFrame["Input Data"]    

    print (respSecData.unique())

    ### Form a new Data Frame with the required fields ###   
    orgDict = list(zip(respData, respActData, respSecData, inpData))
    orgDataFrame = pd.DataFrame(orgDict, columns = ['Resp_Data', 'Resp_Action_Data', 'Secnd_Action_Data', 'inp_Data'])
        
    ### Append Label Num IDs for the Target Class Labels ###
    orgDataFrame['sec_label_num'] = orgDataFrame.Secnd_Action_Data.map({
            																	'No Action required':8,
            																	'Create the order next service in same location':9,
            																	'Process as Install order':10,
            																	'Process the order with the input address':11,
            																	'Process as TOA':12,
            																	'Check for Billing & Active location steps':13,
            																	'Disconnect the O order & Active the customer':14,
            																	'Order already completed & Check the order':15,
                                                                     'Process as Move order':16,
            																	'Process the order with the input Unit/LOT #':17,
            																	'Order already cancelled & Check the order':18,
            																	'Process the order with New TN  #':19,
            																	'Check Billing & Disconnect the location':20,
            																	'Process the O order first & Process other orders':21,
            																	'Process the C order first & Process other orders':22,
            																	'Check if loc is Active in MIROR, Hold the order':23,
            																	'Check for OC provided':24,
            																	'Process ORCAN Process':25,
            																	'Process Re-establishment process':26                                                                  
                                                             })
    
    orgDataFrame['label_num'] = orgDataFrame.Resp_Action_Data.map({
                                                                    'Check & work the order':1, 
                                                                    'Check the Billing/Dir/Order for active TN #':2,
                                                                    'Create the Record in 2nd Drop':3,
                                                                    'Disconnect & Create Record in the location':4,
                                                                    'Hold the order until response':5,
                                                                    'Proceed with Change Process ':6,
                                                                    'Cancel the order':7
                                                             })    
    
    '''    
    lookup_dict = {
                    'svc'   : 'service', 
                    'svr'   : 'service', 
                    'placed': 'place',
                    'tn='   : 'tn',
                    'called': 'call',
                    'cus'   : 'customer', 
                    'cust'  : 'customer',
                    'hses'  : 'house',
                    'autodilaer': 'autodialer' , 
                    'act'   : 'active',
                    'actv'  : 'active', 
                    '2nd'   : 'second',
                    'addtl' : 'additional',
                    'addl'  : 'additional',
                    'cncl'  : 'cancel',
                    'cancelled': 'cancel',
                    'autod' : 'autodialer',
                    'ialer' : 'dialer',
                    'auto-dialer': 'autodialer',
                    'rdy'   : 'ready',   
                    'loc'   : 'location',
                    'dup'   : 'duplicate',
                    'thru'  : 'through'
                  } 
    '''
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
    secData   = orgDataFrame.sec_label_num    
    
    print (respData[:10])
    print (secData[:10])
    
    #detailedDataAnalysis(orgDataFrame)       
              
    newRespData = []
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
        
    for line in respData:      
        #name = (line.split(' ')[1]).lower()
        #line = (' '.join(line.split(' ')[2:])).lower()
        line = line.replace('/', ' ')
        #line = line.replace('auto dialer', 'autodialer')
        #line = line.replace('auto dial'  , 'autodialer')
        #line = line.replace(name, '')
        
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
    
    RPAInpData = formDataMultiLabel(newRespDataSeries, orgDataFrame) 
    
    return (newRespDataSeries, labelData, secData, RPAInpData)
    

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
    
    #labTest[:10]    
    # Modify to return specfic class types
    #plot_confusion_matrix (labTest, labPredClass)
    return (metrics.accuracy_score(labTest, labPredClass))

########################## Decision Tree Classifier ######################################
def DecisionTreeClassifier(rData, lData):
    clf = tree.DecisionTreeClassifier()
    respTrain, respTest, labTrain, labTest = train_test_split(rData, lData, random_state=1)
    
    vect = TfidfVectorizer(min_df=1, max_df=1.0, stop_words='english')        
    respTrainVec = vect.fit_transform(respTrain)    

    # To be commented for Pickle Building of Vectorizer
    respTestVec = vect.transform(respTest)
        
    clf.fit(respTrainVec, labTrain)
    pred = clf.predict_proba(respTestVec)
    
    print (pred)

########################## Naive Bayes ######################################
def multiLabel_SKLearn_GaussianNBayes(rData, lData, sData):
    
    xData = rData.values
    yData = np.array( [lData.values, sData.values] )       
        
    respTrain, respTest, labTrain, labTest = train_test_split(xData, yData, random_state=1)    
    
    classifier = BinaryRelevance(GaussianNB())
    #classifier = ClassifierChain(GaussianNB())
    #classifier = LabelPowerset(GaussianNB())
    
    classifier.fit(respTrain, labTrain)
    predictions = classifier.predict(respTest)
    acc = accuracy_score(labTest, predictions)
    
    return acc
    
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
    respData, labelData, secData, RPAInpData = readFormatData(DATA_LOCATION)

    ### Random Forest from SKLEARN ###
    randForAccuracy = randFor(respData, secData)
    print (randForAccuracy)
    
    # Multi-Label Prediction ####
    #skMultiLearn = multiLabel_SKLearn_GaussianNBayes(respData, labelData, secData)
    #print skMultiLearn
    
    ### Decision Tree Classifier
    #DecTreeAccuracy = DecisionTreeClassifier(respData, secData)
    #print (DecTreeAccuracy)
    
    ### Naive-Bayes from NLTK ###
    #nltkNBayesAcc = nltkNBayes(respData, labelData) 
    
    #print nltkNBayesAcc

if __name__ == "__main__":
    main()