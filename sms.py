import numpy as np
import ast
from csv import reader 
import pandas as pd 
import nltk
import pickle
from pandas import DataFrame
import io
from sklearn.externals import joblib
import pandas

df_sms = pd.read_csv('spam.csv',encoding='latin-1')
df_sms.head()

df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})
print(df_sms.head())
print (len(df_sms))
print(df_sms.tail())
df_sms.label.value_counts()
print(df_sms.describe())
df_sms['length'] = df_sms['sms'].apply(len)
print(df_sms['length'].describe())
print(df_sms.head())

import matplotlib.pyplot as plt
import seaborn as sns
df_sms['length'].plot(bins=50, kind='hist')
plt.show()
df_sms.hist(column='length', by='label', bins=50,figsize=(10,4))
plt.show()

df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})
print(df_sms.shape)
print(df_sms.head())

print(df_sms.tail())
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
lower_case_documents = [d.lower() for d in documents]
print(lower_case_documents)
sans_punctuation_documents = []

import string
for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans("","", string.punctuation)))
    
sans_punctuation_documents
preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]
preprocessed_documents
frequency_list = []

import pprint
from collections import Counter
frequency_list = [Counter(d) for d in preprocessed_documents]
pprint.pprint(frequency_list)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)
count_vector.get_feature_names()
doc_array = count_vector.transform(documents).toarray()
print(doc_array)
frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
print(frequency_matrix)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], df_sms['label'],test_size=0.30,random_state=1)

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
a=naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))
print('Confusion matrix: {}'.format(confusion_matrix(y_test, predictions)))



value = input("Enter your sms: ") 
df_sms1 = pd.read_fwf(io.StringIO(value), header=None, widths=[500])
s1 = df_sms1.iloc[:,0]
test = count_vector.transform(s1)
pred = naive_bayes.predict(test)
if pred == 0:
	print("THE ENTERED SMS IS HAM")

else:
	print("THE ENTERED SMS IS SPAM")