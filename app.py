# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:40:34 2020

@author: admin
"""


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


#from StringIO import StringIO

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    #output = request.form.values()
    
    #content = request.form.getvalue('testfile')
    #fileh = StringIO(content)
    # You can now call fileh.read, or iterate over it
    print(request.content_type)
	#data = {}
    try:
        testfile = None
        if ('testfile' in request.files):
            testfile = request.files['testfile']
            testdataset = pd.read_excel(testfile)
            print('data_xls:::::::::::::',testdataset)
            print('if testfile:',testfile)            
            print('##',testfile.filename)        
            #testfileData = io.BytesIO(request.get_data())
            #testfileData = request.get_data()  #This returned only b' 
            #testfileData = testfile.read()    #this returned binary version of file
            #print(':',testfile.stream.read()) # this returned only b'            
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing file', 500
    
    incwords = ['negotiation','qc','nrp','pwc','application','9149','liability','promise','p2p','annual','ideal','payments','payment','mopf','creation','benefit','cofc','account','accounts','bancs','lo','enforcement','arrears','variation','closure','master']
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
     
    xtstLen = len(sTst)
    if xtstLen < 26:
        tmpArr = np.array(list(sTst))
        print("Unique values in incwords that are not in array2:",np.setdiff1d(incwords, tmpArr))
        diffArr = np.setdiff1d(incwords, tmpArr)
        diffStr = ' '.join(diffArr)
        X_tst.append(diffStr) 
        print('xtst aft diff string::::::::::::',X_tst)

        
    
    from sklearn.feature_extraction.text import CountVectorizer    
    #cv = CountVectorizer(max_features = 26)
    X_test = cv.fit_transform(X_tst).toarray()
    #inc_name = testdataset.iloc[:, 0].values
    
    
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    
    finalstr = ''
    for i in range(0, len(testdataset)):
        print('SummaryI->',testdataset['Incident'][i])
        print('y_pred->',y_pred[i])
        if y_pred[i]==0:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Application' + '\n'
        elif y_pred[i]==1:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Payment' + '\n'
        elif y_pred[i]==2:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Cofc' + '\n'
        elif y_pred[i]==3:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Perf Cal' + '\n'
        elif y_pred[i]==4:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Bancs' + '\n'
        elif y_pred[i]==5:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Arrears' + '\n'
        else:
             finalstr = finalstr + testdataset['Summary'][i] + '=======> Enforcement' + '\n'
    
    # def convert_to_name(x):
    #     word_dict = {0:'Application', 1:'Payment', 2:'Cofc', 3:'Perf Cal', 4:'Bancs', 5:'Arrears', 6:'Enforcement'}
    #     return word_dict[word]

    # y_pred = y_pred.apply(lambda x : convert_to_name(x))
    
    # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
             
        
    return render_template('index.html', prediction_text='Attached file value $ {}'.format(finalstr))


if __name__ == "__main__":
    app.run(debug=True)



