{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing the libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_excel('Inc_Desc.xls')\n",
    "testdataset = pd.read_excel('test.xls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "26\n",
      "Set Train: {'creation', 'payment', 'qc', 'liability', 'enforcement', 'closure', 'promise', 'bancs', 'cofc', 'mopf', 'account', 'variation', 'nrp', 'accounts', 'application', 'arrears', 'p2p', 'master', 'negotiation', 'benefit', 'pwc', 'payments', 'annual', '9149', 'ideal', 'lo'}\n",
      "Set Test: {'creation', 'liability', 'payment', 'qc', 'enforcement', 'closure', 'bancs', 'cofc', 'mopf', 'account', 'variation', 'nrp', 'accounts', 'application', 'arrears', 'p2p', 'master', 'negotiation', 'benefit', 'pwc', 'payments', 'annual', '9149', 'ideal', 'lo'}\n"
     ]
    }
   ],
   "source": [
    "incwords = ['negotiation','qc','nrp','pwc','application','9149','liability','promise','p2p','annual','ideal','payments','payment','mopf','creation','benefit','cofc','account','accounts','bancs','lo','enforcement','arrears','variation','closure','master']\n",
    "print(len(incwords))\n",
    "#For train data\n",
    "X_trn=[]\n",
    "sTrn = set()\n",
    "for i in range(0, len(dataset)):\n",
    "    desc = dataset['Description'][i].lower()    \n",
    "    #keys = re.sub(incwords, ' ', desc)\n",
    "    desc = desc.split()    \n",
    "    desc = [word for word in desc if word in set(incwords)]\n",
    "    #print('filtered version:',desc)    \n",
    "    for i in range(0, len(desc)):\n",
    "        sTrn.add(desc[i])         \n",
    "    \n",
    "    desc = ' '.join(desc)\n",
    "    X_trn.append(desc)\n",
    "\n",
    "#For test data    \n",
    "X_tst = []\n",
    "sTst = set()\n",
    "for i in range(0, len(testdataset)):\n",
    "    testdesc = testdataset['Description'][i].lower()\n",
    "    testdesc = testdesc.split()\n",
    "    testdesc = [word for word in testdesc if word in set(incwords)]\n",
    "    for i in range(0, len(testdesc)):\n",
    "        sTst.add(testdesc[i])\n",
    "    \n",
    "    testdesc = ' '.join(testdesc)\n",
    "    X_tst.append(testdesc)\n",
    "\n",
    "print('Set Train:',sTrn)\n",
    "print('Set Test:',sTst)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A 26\n",
      "B 25\n"
     ]
    }
   ],
   "source": [
    "xtrnLen = len(sTrn)\n",
    "xtstLen = len(sTst)\n",
    "\n",
    "print(\"A\",xtrnLen)\n",
    "print(\"B\",xtstLen)\n",
    "\n",
    "# Creating the Bag of Words model For train Data\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "cv = CountVectorizer(max_features = xtstLen) if xtrnLen >= xtstLen else CountVectorizer(max_features = xtrnLen)\n",
    "X = cv.fit_transform(X_trn).toarray()\n",
    "y = dataset.iloc[:, 1].values\n",
    "\n",
    "# Creating the Bag of Words model for test data\n",
    "#from sklearn.feature_extraction.text import CountVectorizer\n",
    "#cv = CountVectorizer(max_features = len(X_trn))\n",
    "X_test = cv.fit_transform(X_tst).toarray()\n",
    "y_test = testdataset.iloc[:, 1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB(priors=None, var_smoothing=1e-09)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting Naive Bayes to the Training set\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "classifier = GaussianNB()\n",
    "classifier.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predicting the Test set results\n",
    "y_pred = classifier.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 3, 0, 0, 0, 0, 0],\n",
       "       [0, 0, 2, 0, 0, 0, 0],\n",
       "       [0, 0, 0, 1, 1, 4, 0],\n",
       "       [0, 0, 0, 0, 2, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 2]], dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7, 25)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(21, 25)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[4 5 6 3 3 1 1 2 0 3 0 0 3 3 4 6 5 1 0 2 3]\n",
      "[4 5 6 3 4 1 1 2 5 5 0 5 5 5 4 6 5 1 0 2 5]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)\n",
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,\n",
       "        0, 1, 1],\n",
       "       [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,\n",
       "        1, 0, 0]], dtype=int64)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "list"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(X_tst)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_tst: ['accounts bancs', 'arrears variation', 'lo enforcement', 'liability p2p', 'annual account', 'mopf payments', 'payment payments', 'cofc closure master', 'creation', '9149', 'qc nrp pwc application creation', 'application', '9149', '', 'account', 'lo', 'negotiation', 'payment', 'qc nrp', 'benefit', 'ideal']\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 1],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "        1, 1, 0],\n",
       "       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "        0, 1, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0]], dtype=int64)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('X_tst:',X_tst)\n",
    "X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='entropy', max_depth=None, max_features='auto',\n",
       "                       max_leaf_nodes=None, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=20,\n",
       "                       n_jobs=None, oob_score=False, random_state=0, verbose=0,\n",
       "                       warm_start=False)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting Random Forest Classification to the Training set\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "randomforestClassifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)\n",
    "randomforestClassifier.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 3, 0, 0, 0, 0, 0],\n",
       "       [0, 0, 2, 0, 0, 0, 0],\n",
       "       [0, 0, 0, 1, 1, 4, 0],\n",
       "       [0, 0, 0, 0, 2, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 2]], dtype=int64)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Predicting the Test set results\n",
    "y_pred = classifier.predict(X_test)\n",
    "cmr = confusion_matrix(y_test, y_pred)\n",
    "cmr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4, 5, 6, 3, 3, 1, 1, 2, 0, 3, 0, 0, 3, 3, 4, 6, 5, 1, 0, 2, 3],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4, 5, 6, 3, 4, 1, 1, 2, 5, 5, 0, 5, 5, 5, 4, 6, 5, 1, 0, 2, 5],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 3, 0, 0, 0, 0, 0],\n",
       "       [0, 0, 2, 0, 0, 0, 0],\n",
       "       [0, 0, 0, 1, 1, 4, 0],\n",
       "       [0, 0, 0, 0, 2, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 2]], dtype=int64)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting Logistic Regression to the Training set\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "lclassifier = LogisticRegression(random_state = 0)\n",
    "lclassifier.fit(X, y)\n",
    "\n",
    "# Predicting the Test set results\n",
    "y_pred = lclassifier.predict(X_test)\n",
    "\n",
    "# Making the Confusion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "cm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[4 5 6 3 3 1 1 2 0 3 0 0 3 3 4 6 5 1 0 2 3]\n",
      "[4 5 6 3 4 1 1 2 5 5 0 5 5 5 4 6 5 1 0 2 5]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)\n",
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 0, 0, 0, 0, 3],\n",
       "       [0, 3, 0, 0, 0, 0, 0],\n",
       "       [0, 0, 2, 0, 0, 0, 0],\n",
       "       [0, 0, 0, 0, 1, 0, 5],\n",
       "       [0, 0, 0, 0, 2, 0, 0],\n",
       "       [0, 0, 0, 0, 0, 2, 0],\n",
       "       [0, 0, 0, 0, 0, 0, 2]], dtype=int64)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting Kernel SVM to the Training set\n",
    "from sklearn.svm import SVC\n",
    "classifier = SVC(kernel = 'rbf', random_state = 0)\n",
    "classifier.fit(X, y)\n",
    "\n",
    "# Predicting the Test set results\n",
    "y_pred = classifier.predict(X_test)\n",
    "\n",
    "# Making the Confusion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[4 5 6 3 3 1 1 2 0 3 0 0 3 3 4 6 5 1 0 2 3]\n",
      "[4 5 6 6 4 1 1 2 6 6 0 6 6 6 4 6 5 1 6 2 6]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)\n",
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(lclassifier,open('model.pkl','wb'))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
