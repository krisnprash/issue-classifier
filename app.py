# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:40:34 2020

@author: admin
"""


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
#from jira import JIRA
# Reassign an issue:
# # requires issue assign permission, which is different from issue editing˓→permission!
# jira.assign_issue(issue, 'newassignee')


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
    
	#data = {}
    try:
        testfile = None
        if ('testfile' in request.files):
            testfile = request.files['testfile']
            testdataset = pd.read_excel(testfile)
                   
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
    print('Len dataset:',len(testdataset))
    print('Len dataset:',len(testdataset['Description']))
    print('Len dataset:',len(testdataset['Incident']))
    
    # making new data frame with dropped NA values 
    testdataset = testdataset.dropna(axis = 0, how ='all')   
    # comparing sizes of data frames 
    # print("Old data frame length:", len(data), "\nNew data frame length:",  
    #    len(new_data), "\nNumber of rows with at least 1 NA value: ", 
    #    (len(data)-len(new_data))) 
    
    print('Len :',len(testdataset))
    print('Len :',len(testdataset['Description']))
    print('Len :',len(testdataset['Incident']))
    
      
    # testdataset.drop(testdataset.columns[[0, 3, 4, 5, 6, 7, 8]], axis = 1, inplace = True)   
    # print('Len columndrp :',len(testdataset))
    # print('Len columndrop:',len(testdataset['Description']))
    # print('Len columndroop:',len(testdataset['Incident']))
    
    for i in range(0, len(testdataset)):        
        testdesc = str(testdataset['Description'][i]).lower()
        testdesc = testdesc.split()
        testdesc = [word for word in testdesc if word in set(incwords)]
        for i in range(0, len(testdesc)):
            sTst.add(testdesc[i])
        
        testdesc = ' '.join(testdesc)
        X_tst.append(testdesc)
    
    
            
    print('xtst before diff string%%%%%%%%%%%%%',X_tst)
    xtstLen = len(sTst)
    if xtstLen < 26:
        tmpArr = np.array(list(sTst))
        print("Unique values in incwords that are not in xtst:",np.setdiff1d(incwords, tmpArr))
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
        # print('Incident->',testdataset['Incident'][i])
        # print('y_pred->',y_pred[i])
        if y_pred[i]==0:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Application#')
        elif y_pred[i]==1:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Payment#')
        elif y_pred[i]==2:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Cofc#')
        elif y_pred[i]==3:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Perf Cal#')
        elif y_pred[i]==4:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Bancs#')
        elif y_pred[i]==5:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Arrears#')
        else:
             finalstr = str(finalstr) + str(testdataset['Incident'][i]) + str('=======> Enforcement#')
    
    # def convert_to_name(x):
    #     word_dict = {0:'Application', 1:'Payment', 2:'Cofc', 3:'Perf Cal', 4:'Bancs', 5:'Arrears', 6:'Enforcement'}
    #     return word_dict[word]

    # y_pred = y_pred.apply(lambda x : convert_to_name(x))
    
    # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    outputList = finalstr.split('#')
    # output = '\n'.join(outputList)
    # print('outdgdsdgds::::::::',output)
    
        
    
    # Standard basic authentication is now deprecated in Jira from June 2019.
    # An API key is required for basic authentication which now replaces the 'password' requirement. API key's can be generated from:  https://confluence.atlassian.com/cloud/api-tokens-938839638.html.
    # Once the key has been obtained, replace your current Jira object with the following:
    # jira = JIRA(basic_auth=("enter username", "enter API key"), options={'server': "enter server"}).
    
    # options = {'server': jiraURL}
    # jira = JIRA(options, basic_auth=(jiraUserName, jiraPassword))
    # issue = jira.issue('ESS-138581')
    # print issue.fields.project.key
    # print issue.fields.issuetype.name
    # print issue.fields.reporter.displayName
    # print issue.fields.summary
    # print issue.fields.project.id
    
    # group = jira.group_members("jira-users")
    # for users in group:
    # print users
    # If you get handshake error [SSLV3_ALERT_HANDSHAKE_FAILURE]. Kindly install the following modules in python
    # pyOpenSSL
    # ndg-httpsclient
    # pyasn1
    
    # This script shows how to use the client in anonymous mode
    # # against jira.atlassian.com.
    # from jira import JIRA
    # import re
    # # By default, the client will connect to a JIRA instance started from the
    # ˓→Atlassian Plugin SDK
    # # (see https://developer.atlassian.com/display/DOCS/
    # ˓→Installing+the+Atlassian+Plugin+SDK for details).
    #     options = {
    # 'server': 'https://jira.atlassian.com'}
    # jira = JIRA(options)
    # Override this with the options parameter.
    # Get all projects viewable by anonymous users.
    # projects = jira.projects()
    # # Sort available project keys, then return the second, third, and fourth keys.
    # keys = sorted([project.key for project in projects])[2:5]
    # # Get an issue.
    # issue = jira.issue('JRA-1330')
    # # Find all comments made by Atlassians on this issue.
    # atl_comments = [comment for comment in issue.fields.comment.comments
    # if re.search(r'@atlassian.com$', comment.author.emailAddress)]
    # # Add a comment to the issue.
    # jira.add_comment(issue, 'Comment text')
    # # Change the issue's summary and description.
    # issue.update(
    # summary="I'm different!", description='Changed the summary to be different.')
    # # Change the issue without sending updates
    # issue.update(notify=False, description='Quiet summary update.')
    # # You can update the entire labels field like this
    # issue.update(labels=['AAA', 'BBB'])
    # # Or modify the List of existing labels. The new label is unicode with no
    # # spaces
    # issue.fields.labels.append(u'new_text')
    # issue.update(fields={"labels": issue.fields.labels})
    # # Send the issue away for good.
    # issue.delete()
    # # Linking a remote jira issue (needs applinks to be configured to work)
    # issue = jira.issue('JRA-1330')
    # issue2 = jira.issue('XX-23') # could also be another instance
    # jira.add_remote_link(issue, issue2)
    
     #   Reassign an issue:
    # requires issue assign permission, which is different from issue editing˓→permission!
   # jira.assign_issue(issue, 'newassignee')


    
   # return render_template('index.html', prediction_text='Attached file value $ {}'.format(output))    
    return render_template('index.html', prediction_list = outputList)                    
    


if __name__ == "__main__":
    app.run(debug=True)



