import numpy as np
import pandas as pd
import io
import ast
# import nltk
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')

def index():
	return render_template('index.html')

# prediction function 
def ValuePredictor(to_predict_list): 
	import pandas
	df_sms = pd.read_csv('spam.csv',encoding='latin-1')
	# df_sms['length'].describe()
	df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
	df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})
	#Checking the maximum length of SMS
	df_sms['length'] = df_sms['sms'].apply(len)

	df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})

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
	from collections import Counter

	frequency_list = [Counter(d) for d in preprocessed_documents]
	from sklearn.feature_extraction.text import CountVectorizer
	count_vector = CountVectorizer()
	count_vector.fit(documents)
	count_vector.get_feature_names()
	doc_array = count_vector.transform(documents).toarray()
	frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], df_sms['label'],test_size=0.30,random_state=1)

	count_vector = CountVectorizer()
	training_data = count_vector.fit_transform(X_train)
	testing_data = count_vector.transform(X_test)
	from sklearn.naive_bayes import MultinomialNB
	naive_bayes = MultinomialNB()
	naive_bayes.fit(training_data,y_train)

	df_sms1 = pd.read_fwf(io.StringIO(to_predict_list), header=None, widths=[500])
	s1 = df_sms1.iloc[:,0]

	# df_sms1.astype('string').dtypes
	# print(df_sms1)
	test = count_vector.transform(s1)
	result = naive_bayes.predict(test) 
	return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
	if request.method == 'POST': 
		to_predict_list = request.form["sms"]
		print(to_predict_list)

		result = ValuePredictor(to_predict_list)         
		if int(result)== 0: 
			prediction ='Entered SMS is Ham'
		else: 
			prediction ='Entered SMS is Spam'            
		return render_template("result.html", prediction = prediction) 
