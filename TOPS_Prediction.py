# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 17:39:02 2018

@author: velmurugan.m
"""

import pickle
import pandas as pd
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


targetList = (
                'Check & work the order', 
                'Check the Billing/Dir/Order for active TN #',
                'Create the Record in 2nd Drop',
                'Disconnect & Create Record in the location',
                'Hold the order until response',
                'Proceed with Change Process ',
                'Cancel the order'        
              )
        

DATA_LOCATION = 'D:\Winstream_data\TOPS\FAS_4.xlsx'

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def funcForValidatingWholeSet(respData):
    
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
            
    newRespData = []    
    
    for line in respData:        
        tokens = word_tokenize( (str(line )).lower() ) 
        tokens = tokens[2:]
        
        newVal = map( lambda val: lookup_dict[val] if val in lookup_dict else val, tokens )    
        
        remDate = filter( lambda ThisWord: not re.match('^(?:(?:[0-9]{1,2}[:\/,]){1,2}[0-9]{1,4})$', ThisWord), newVal)
        
        remInt = filter( lambda ThisWord: not re.match('^(\d{1,10}|\d{12})$', ThisWord), remDate)  
        
        remSplCh = filter( lambda ThisWord: not re.match('[^ a-zA-Z0-9]', ThisWord), remInt)  
        
        remDat = filter( lambda ThisWord: not re.match("(u')", ThisWord), remSplCh)      
        
        filSen = [w for w in remDat if not w in stop_words]
        
        newLine = " ".join(filSen)
    
        newRespData.append(newLine)    
    
    newRespDataSeries = pd.Series( newRespData )
    
    return (newRespDataSeries)

def readFormatData(inputText):
        
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
                    'hses'  : 'houses',
                    'dup'   : 'duplicate'                   
                  }                  
                
    tokens = word_tokenize( (str(inputText )).lower() ) 
    tokens = tokens[2:]
        
    newVal = map( lambda val: lookup_dict[val] if val in lookup_dict else val, tokens )            
    remDate = filter( lambda ThisWord: not re.match('^(?:(?:[0-9]{1,2}[:\/,]){1,2}[0-9]{1,4})$', ThisWord), newVal)        
    remInt = filter( lambda ThisWord: not re.match('^(\d{1,10}|\d{12})$', ThisWord), remDate)          
    remSplCh = filter( lambda ThisWord: not re.match('[^ a-zA-Z0-9]', ThisWord), remInt)          
    remDat = filter( lambda ThisWord: not re.match("(u')", ThisWord), remSplCh)         
    filSen = [w for w in remDat if not w in stop_words]
    lemWord = map( wordnet_lemmatizer.lemmatize, filSen)
        
    newLine = " ".join(lemWord)            
    
    newRespDataSeries = pd.Series( newLine )    
    return newRespDataSeries


def processData(respData):
   filename = "D:\Windstream_ML\Models\WindTOPSRandForest.pkl"
   tfidfFile = "D:\Windstream_ML\Models\WindTOPSTFIDF.pkl"
      
   TOPSRandFor = open(filename, 'rb')
   TFIDFFile = open(tfidfFile, 'rb')     
   
   randForest = pickle.load(TOPSRandFor)
   TOPSRandFor.close()   
   
   vectorize = pickle.load(TFIDFFile)
   TFIDFFile.close()
   
   #test = "11/1/2017 SGriffith added to autodialer to get cust to call us to confirm address/apt or lot#/multiple hses @ loc/additional svc/etc. put order in cust action 010118 dd 0 wu simple wkfr. 1172 I01148 CUS-ACTION 172-6039 OCTAVIA U CALLAWAY || -"
   cleanData = readFormatData(respData)
   vectData = vectorize.transform(cleanData)
   output = randForest.predict(vectData)
   outText = targetList[output-1]
   
   return outText
   #### For TOPS Bulk Data ###################################################
   '''
   orgDataFrame = pd.read_excel(DATA_LOCATION, sheetname='Sheet1').iloc[2001:2675]   
   respData = orgDataFrame[orgDataFrame.columns[5]]
   
   newOutput = []
      
   for val in respData:   
       cleanData = readFormatData(val)          
       vectData = vectorize.transform(cleanData)
       output = randForest.predict(vectData)       
       outText = targetList[output-1]       
       newOutput.append(outText)
   
   newDataFr = pd.DataFrame({'Response Data': respData, 'Manual Action Taken': newOutput})   
   newDataFr.to_csv("predictdOut.csv")   
   #return targetList[output-1]
   '''


#def main():
#    processData()
    
#if __name__ == "__main__":
#    main()    
