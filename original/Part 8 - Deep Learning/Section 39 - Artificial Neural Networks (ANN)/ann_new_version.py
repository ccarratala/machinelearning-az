# Redes Neuronales Artificales


# Instalar Theano (instalar en terminal del mac)
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras (instalar en terminal del mac)
# conda install -c conda-forge keras


''' Parte 1 - Pre procesado de datos '''
# Usamos la plantilla de clasificación porque el problema que vamos a trater ese este

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values  # omitimos las primeras pq no son utiles
y = dataset.iloc[:, 13].values    # binaria (0,1)

# Primero vamos a transformar las var categoricas 
# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# hacemos 2 labelencoder las var necesitan diferentes codificaciones
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])    # pais
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])    # sexo

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough' )    # creamos los dummies (1 col por categ)

X = onehotencoder.fit_transform(X)
X = X[:, 1:]    # quitamos la primera col (solo en var de pais)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# IMPORTANTE: Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



''' Parte 2 - Construir la RNA '''

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential    # inicializar los parametros
from keras.layers import Dense    # crear las capas


# Inicializar la RNA (arquitectura)
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  # inicializacion aleatoria (tipo distribucion)
                     # units= num de nodos hidden layer (conexion) -> media de inputs y outputs (11+1/2)
                     activation = "relu", input_dim = 11))   # tenemos 11 var = input (11 nodos entrada)
                     # func activacion = relu = rectificador

# se añaden capas igual pero sin añadir input_dim (inputs) -> se conecta con la anterior
# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))

# Añadir la capa de salida (solo 1 nodo = output y cambiamos func activacion)
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))


# Compilar la RNA: conectarse
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# optimizer = algoritmo que optimiza los pesos (tipo gradient, pero usamos adam)
# loss = medir el error (usamos esa pq la y es binaria)


# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)
# batch = tamaño del bloque a ejecutar (obsv) / epochs = veces que pasa por el dataset (iteraciones)



''' Parte 3 - Evaluar el modelo y calcular predicciones finales '''

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)    # saber aquellas superior al threshold establecido (y=TRUE=1)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
