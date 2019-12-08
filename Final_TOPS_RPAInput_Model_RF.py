# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:40:54 2018

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
from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords

import re
import pickle

from IPython.core.display import display, HTML

DATA_LOCATION = 'D:\Winstream_data\TOPS\second\FAS - Order Correction response details_Updated_last.xlsx'


########################## Form Multi-Label for predicting Next-Action ########
def formDataMultiLabel(sampData, df):
       
    ####### Convert Response Category into Labelized Binarizer ################
    inpRPAData     = (df.inp_Data).astype(str)
    inpRPAData     = inpRPAData.apply(lambda x: x.split()[0])
    lab, lev       = pd.factorize(inpRPAData)
    
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
            																	'Process the order with the input Address':24,
            																	'Check for OC provided':25,
            																	'Process ORCAN Process':26,
            																	'Process as move order':27,
            																	'Process Re-establishment process':28                                                                  
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
            
    lookup_dict = {
                    'svc'   : 'service', 
                    'svr'   : 'service', 
                    'placed': 'place',
                    'tn='   : 'tn',
                    'called': 'call',
                    'cus'   : 'customer', 
                    'cust'  : 'customer',                     
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
                    'dup'   : 'duplicate'
                  }   
    
    respData  = orgDataFrame.Resp_Data         
    labelData = orgDataFrame.label_num
    secData   = orgDataFrame.sec_label_num    
                  
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
    
    RPAInpData = formDataMultiLabel(newRespDataSeries, orgDataFrame) 
    
    return (newRespDataSeries, labelData, secData, RPAInpData)


########################## TF-IDF Vectorizer Pickle ###########################
def formTFIDFVectPickl(respData):
    vect = TfidfVectorizer(min_df=1, max_df=1.0, stop_words='english')        
    respVect = vect.fit_transform(respData) 
    
    #### Create Pickle for Tfidf Vectorizer  ####
    pickle.dump(vect, open('TfidfVect.pkl', 'wb'))
    
    return respVect
    
########################## RF Classifier for Manual Action ####################
def RFManAct(rData, lData):    
    randClass = RandomForestClassifier(n_estimators = 100)    
    randClass.fit(rData, lData)      
    
    #### Create Pickle for RFManAct  ####
    pickle.dump(randClass, open('RFManAct.pkl', 'wb'))
                                
########################## RF Classifier for Next Step ########################
def RFSeconStep(rData, lData):    
    randClass = RandomForestClassifier(n_estimators = 100)    
    randClass.fit(rData, lData)      
    
    #### Create Pickle for RFManAct  ####
    pickle.dump(randClass, open('RFSeconStep.pkl', 'wb'))
    
########################## RF Classifier for RPA Input Data ###################
def RFInpData(rData, lData):    
    randClass = RandomForestClassifier(n_estimators = 100)    
    randClass.fit(rData, lData)      
    
    #### Create Pickle for RFManAct  ####
    pickle.dump(randClass, open('RFInpData.pkl', 'wb'))

########################## Main Function ##########################
def main():
    respData, manActData, nxtStepData, RPAInpData = readFormatData(DATA_LOCATION)    
    
    ### Form Vectorizer Pickle #####
    respVectData = formTFIDFVectPickl(respData)

    ### RF Classifier for Manual Action Taken ###
    RFManAct(respVectData, manActData)
    
    ### RF Classifier for Second Step ###
    RFSeconStep(respVectData, nxtStepData)
    
    ### RF Classifier for RPA Input Data ###
    RFInpData(respVectData, RPAInpData)

if __name__ == "__main__":
    main()