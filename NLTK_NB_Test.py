# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:24:58 2018

@author: velmurugan.m
"""

from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import nltk

train = [('I love this sandwich.', 'pos'),
('This is an amazing place!', 'pos'),
('I feel very good about these beers.', 'pos'),
('This is my best work.', 'pos'),
("What an awesome view", 'pos'),
('I do not like this restaurant', 'neg'),
('I am tired of this stuff.', 'neg'),
("I can't deal with this", 'neg'),
('He is my sworn enemy!', 'neg'),
('My boss is horrible.', 'neg')]

all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))

t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

print t[1:10]

classifier = NaiveBayesClassifier.train(t)