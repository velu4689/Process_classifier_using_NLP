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
import numpy as np


ManActionList = [
                'Check & work the order', 
                'Check the Billing/Dir/Order for active TN #',
                'Create the Record in 2nd Drop',
                'Disconnect & Create Record in the location',
                'Hold the order until response',
                'Proceed with Change Process ',
                'Cancel the order'        
                ]

SecStepList = [
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

'''
RPAInpList = [  
                'APT',
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
'''
RPAInpList = [ 
                'nan', 
                'Address', 
                'Order', 
                'Unit', 
                'APT', 
                'OC', 
                'LOT', 
                'ORCAN', 
                'ROOM',
                'House'
           ]

        

DATA_LOCATION = 'D:\Winstream_data\TOPS\FAS_4.xlsx'
#DATA_LOCATION = 'D:\Winstream_data\TOPS\second\FAS - Order Correction response details_Updated_last.xlsx'

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

def srchAprtment(respData):

    respData = respData.replace('/', ' ')
    words = respData.split(' ')
    for (i, subword) in enumerate(words):
        if (subword == 'apt'): 
            txt = (words[i+1])
     
    return txt    

def srchAddrs(respData):
    return "Address"

def srchhouse(respData):
    return "house"

def srchLot(respData):
    return "Lot"

def srchOC(respData):
    txt = re.findall('[a-zA-Z]{2}\d{5}', respData)
    return txt 

def srchOrcan(respData):
    return "ORCAN"

def srchOrder(respData):
    #txt = re.findall(r"^[a-zA-Z]{1}[0-9]{5}$", respData)
    txt = re.findall('[a-zA-Z]\d{5}', respData)
    return txt
    
def srchRoom(respData):
    return "Room"

def srchUnit(respData):
    respData = respData.replace('/', ' ')
    words = respData.split(' ')
    for (i, subword) in enumerate(words):
        if (subword == 'unit'): 
            txt = (words[i+1])
     
    return txt

def srchNan():
    return "NaN"

def extRPAInpData(ID, respData):
    return {
            1 : srchNan(),
            2 : srchAddrs(respData),
            3 : srchOrder(respData),                
            4 : srchUnit(respData),                
            5 : srchAprtment(respData),
            6 : srchOC(respData),                
            7 : srchLot(respData),
            8 : srchOrcan(respData),                
            9 : srchRoom(respData),
            10: srchhouse(respData)
            }.get(ID, 'default case') 

def processData(respData):
   
   TFIDFPiklLoc   = "D:\Windstream_ML\python_scripts\TfidfVect.pkl"
   ManActPiklLoc  = "D:\Windstream_ML\python_scripts\RFManAct.pkl"
   SecStpPiklLoc  = "D:\Windstream_ML\python_scripts\RFSeconStep.pkl"
   RFInpPiklLoc   = "D:\Windstream_ML\python_scripts\RFInpData.pkl"
      
   vectBin   = open(TFIDFPiklLoc, 'rb')
   ManActBin = open(ManActPiklLoc, 'rb')
   SecStpBin = open(SecStpPiklLoc, 'rb')
   RFInpBin  = open(RFInpPiklLoc, 'rb')
      
   TFIDFVect = pickle.load(vectBin)
   vectBin.close()   
   
   ManActPred = pickle.load(ManActBin)
   ManActBin.close()
   
   SecStpPred = pickle.load(SecStpBin)
   SecStpBin.close()
   
   RPAInpPred = pickle.load(RFInpBin)
   RFInpBin.close()
   
   '''
   #test = "11/1/2017 SGriffith added to autodialer to get cust to call us to confirm address/apt or lot#/multiple hses @ loc/additional svc/etc. put order in cust action 010118 dd 0 wu simple wkfr. 1172 I01148 CUS-ACTION 172-6039 OCTAVIA U CALLAWAY || -"
   orgDataFrame = pd.read_excel(DATA_LOCATION, sheet_name='Sheet1').iloc[2001:]   
   respData     = orgDataFrame[orgDataFrame.columns[5]]
   
   
   testData = respData[343]
   print (testData)
   '''
   
   cleanData = readFormatData(respData)
   vectData = TFIDFVect.transform(cleanData)
   
   # Predict Manual Action to be Taken #############################
   ManActID  = ManActPred.predict(vectData)
   ManActTxt = ManActionList[ManActID.item()-1]
   
   # Predict Second Step to be Taken ###############################
   SecStepID  = SecStpPred.predict(vectData)
   SecStepTxt = SecStepList[SecStepID.item()-8]   
   
   # Predict RPA Input Data to be Given ############################
   RPAInpID  = RPAInpPred.predict(vectData)
   RPAInpTxt = extRPAInpData(RPAInpID.item() + 1, testData)

   # Detect RPA Response Data ###################################### 
   return (ManActTxt, SecStepTxt, RPAInpTxt)

   #### For TOPS Bulk Data ###################################################
   
   '''
   newOutput = []
      
   for val in respData:   
       cleanData = readFormatData(val)          
       vectData = TFIDFVect.transform(cleanData)
       
       SecStepID = SecStpPred.predict(vectData)     
       
       outText = SecStepList[SecStepID.item()-8] 
       #print (outText)
       #print (np.unique(SecStepID))
       #print (SecStepID.item())
       newOutput.append(outText)
   
   newDataFr = pd.DataFrame({'Response Data': respData, 'Second Step': newOutput})   
   newDataFr.to_csv("predictdOut.csv")   
   #return targetList[output-1]
   
   #return (ManActTxt, SecStepTxt, RPAInpTxt) 


#def main():
    #For Indiviudal Data Prediction
    #ManActTxt, SecStepTxt, RPAInpTxt = processData()
    
    #For Bulk Data Prediction
#    processData()
    
    
#if __name__ == "__main__":
#    main()    
