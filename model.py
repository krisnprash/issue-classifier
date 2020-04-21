# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:31:41 2020

@author: admin
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_excel('Inc_Desc.xls')
testdataset = pd.read_excel('test.xls')

incwords = ['negotiation','qc','nrp','pwc','application','9149','liability','promise','p2p','annual','ideal','payments','payment','mopf','creation','benefit','cofc','account','accounts','bancs','lo','enforcement','arrears','variation','closure','master']
#print(len(incwords))
#For train data
X_trn=[]
sTrn = set()
for i in range(0, len(dataset)):
    desc = dataset['Description'][i].lower()    
    #keys = re.sub(incwords, ' ', desc)
    desc = desc.split()    
    desc = [word for word in desc if word in set(incwords)]
    #print('filtered version:',desc)    
    for i in range(0, len(desc)):
        sTrn.add(desc[i])         
    
    desc = ' '.join(desc)
    X_trn.append(desc)

#For test data    
X_tst = []
sTst = set()
for i in range(0, len(testdataset)):
    testdesc = testdataset['Description'][i].lower()
    testdesc = testdesc.split()
    testdesc = [word for word in testdesc if word in set(incwords)]
    for i in range(0, len(testdesc)):
        sTst.add(testdesc[i])
    
    testdesc = ' '.join(testdesc)
    X_tst.append(testdesc)

xtrnLen = len(sTrn)
xtstLen = len(sTst)

print("A",xtrnLen)
print("B",xtstLen)

# Creating the Bag of Words model For train Data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = xtstLen) if xtrnLen >= xtstLen else CountVectorizer(max_features = xtrnLen)
X = cv.fit_transform(X_trn).toarray()
y = dataset.iloc[:, 1].values

# Creating the Bag of Words model for test data
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = len(X_trn))
X_test = cv.fit_transform(X_tst).toarray()
y_test = testdataset.iloc[:, 1].values


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lclassifier = LogisticRegression(random_state = 0)
lclassifier.fit(X, y)

# Predicting the Test set results
y_pred = lclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

pickle.dump(lclassifier,open('model.pkl','wb'))