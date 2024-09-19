# Natural Language Processing


# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
# .tsv valores separados por tabuladores "\t" / quoting = 3 = ignora las comillas dobles


# Limpieza de texto
import re      # regular expressions
import nltk    # manejo de texto
nltk.download('stopwords')    # palabras irrelevantes (articulos, preposicion etc) todos los idiomas
from nltk.corpus import stopwords    # lista de palabras descargadas
from nltk.stem.porter import PorterStemmer    # pasar a infinitivo los verbos (quitamos conjugaciones)

corpus = []    # var para luego de coleccion de texto = lista

for i in range(0, 1000):    # hay 0-999 reviews en el dataset
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])    # rm todo lo que NO sean caracteres 
    # ^ = nos quedamos con las letras juntas de la a-zA-Z y sustituimos todo lo que no sea por un espacio ' '
    review = review.lower()    # pasamos a minusculas
    review = review.split()    # dividimos la cadena de caracteres en una lista = palabras aisladas
    ps = PorterStemmer()       # le pasamos ps a las palabras que sobrevivan a la lista
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # recorre todas las palabras y crea nueva lsita con aquellas que NO estén en el diccionario stopwords (si fuera spanish se cambia)
    review = ' '.join(review)    # volver a crear cadena de texto = combinar con espacio = ' '
    corpus.append(review)    # guardamos esto en la var corpus (cuerpo) = mism len que el dataset


# Crear el Bag of Words (las que aparecen con mayor freq)
from sklearn.feature_extraction.text import CountVectorizer    # transforma texto en frecucnai (numero)
cv = CountVectorizer(max_features = 1500)    # limita el num de col = quitamos las de menos freq = ver primero cuantas hay y limitar
X = cv.fit_transform(corpus).toarray()    # crea modelo y transforma corpus en matriz dispersa (.toarray)
y = dataset.iloc[:, 1].values    # etiqueta de la review (positiva 1 o negativa 0)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Ajustar el clasificador en el Conjunto de Entrenamiento - Naive Bayes probabilidad
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
(55+91)/200    #Precision = 73%

