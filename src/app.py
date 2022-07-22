import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import unicodedata
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

def normalize_string(text_string):
    if text_string is not None:
        result = unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else:
        result = None
    return result

df['review'] = df['review'].str.strip()
df['review'] = df['review'].str.lower()
df['review'] = df['review'].apply(normalize_string)
df['review'] = df['review'].str.replace('!','')
df['review'] = df['review'].str.replace(',','')
df['review'] = df['review'].str.replace('&','')
df['review'] = df['review'].str.normalize('NFKC') #reemplaza texto no latino y trata de arreglar
df['review'] = df['review'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True) #saca tipo loooovvveee y deja love, saca las repeticiones de las letras

X = df['review']
y = df['polarity']

#train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.25, random_state=42)

# Vectorizo el texto de reviews a numeros 
vector = CountVectorizer(stop_words='english') #le saco las palabras de ingles
X_train = vector.fit_transform(X_train).toarray() #lo convierto en matriz
X_test = vector.transform(X_test).toarray() #

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

#model.predict(vector.transform(['Love']))

#Grabo el modelo
import pickle

filename = '../models/nb_model.sav'
pickle.dump(model, open(filename,'wb'))