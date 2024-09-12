#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:10:07 2019

@author: juangabriel
"""

# Regresión Lineal Múltiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Evitar la trampa de las variables ficticias (dummy)
# hay que eliminar una porque no hace falta, no aporta mas info
# quitamos la primera col por ejemplo (la 0)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''No hace fata escalar las variables'''


# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
# es igual que con la lineal, no hay especificos
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)


# Hemos metido todas las variables sin comprobar su significación
# Ahora construimos el modelo óptimo utilizando Eliminación hacia atrás
import statsmodels.api as sm

# hay que añadir una col solo con 1s, sera la que corresponde con el intercept
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# ponemos values despues del arr para que la col de 1s aparezca delante


# Paso 1 
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # variables optimas al inicio son todas
SL = 0.05     # p-value

# Paso 2
# creamos nuevo modelo con Ordinary Least Squares (OLS)
# endog es la var que queremos predecir // exog las features (X_opt)
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()  # para ver la significacion como en R
# vemos cual es la var con mayor p-value y decidimos si rm

# Paso 3: quitamos var 2 (x2) porque no es significativa
X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary() # repetimos los pasos

# Paso 4: hemos quitado la var 1 (que en la consola aparece como x2)
X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary() # repetimos

# Paso 5: quitamos la var 4 (x2)
X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary() # repetimos

# Paso 6: quitamos la var 5 (x2) -> aunque habria que valorar porque esta en el limite
X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary() # en principio ya tienen que ser todas significativas

'''Hay vario criterios para elegir el modelo mas optimo, a parte del p-value'''
