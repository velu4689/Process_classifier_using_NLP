# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:53:28 2018

@author: velmurugan.m
"""

#from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize, pos_tag




#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

#twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#print twenty_train.data

lookup_dict = {
                    'svc'   : 'service', 
                    'svr'   : 'service', 
                    'cus'   : 'customer', 
                    'cust'  : 'customer',                     
                    'actv'  : 'active',                    
                  }

txtData = "If the svc is not good then cust can is actv. If the svr is actv then leave it to cus."
    
tokens = word_tokenize( (str(txtData )).lower() ) 

func = lambda val: lookup_dict[val] if val in lookup_dict else val

newVal = map( lambda val: lookup_dict[val] if val in lookup_dict else val, tokens )     
print newVal                       
#newVal = map( (lambda val: if (inp in lookup_dict) lookup_dict[inp] else inp), tokens)