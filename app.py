# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:40:34 2020

@author: admin
"""


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import cgi, os
import cgitb; cgitb.enable()
from io import BytesIO

#from StringIO import StringIO

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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
            data_xls = pd.read_excel(testfile)
            print('data_xls:::::::::::::',data_xls)
            print('if testfile:',testfile)            
            print('##',testfile.filename)        
            #testfileData = io.BytesIO(request.get_data())
            #testfileData = request.get_data()  #This returned only b' 
            #testfileData = testfile.read()    #this returned binary version of file
            #print(':',testfile.stream.read()) # this returned only b'
            
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing file', 500
       
    return render_template('index.html', prediction_text='Attached file value $ {}'.format(data_xls))


if __name__ == "__main__":
    app.run(debug=True)



