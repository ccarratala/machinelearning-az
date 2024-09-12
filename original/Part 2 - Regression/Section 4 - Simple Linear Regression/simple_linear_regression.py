# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # todas las col menos la ultima
y = dataset.iloc[:, 1].values   # solo la ultima col (valor a predecir)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# lo dividimos 20 train y 10 test (N=30)

'''Escalado de variables NO es necesario si es R. Lineal Simple'''



# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
# le pasamos los train set para ajustar el modelo .fit
regression.fit(X_train, y_train)

# Predecir el conjunto de test (crea nueva var con los valores pred)
y_pred = regression.predict(X_test)


# Visualizar los resultados de entrenamiento
# datos reales
plt.scatter(X_train, y_train, color = "red")
# representamos la recta de regresion
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()


# Visualizar los resultados de test
# controlar si hay overfitting (que no se ajuste bien al test_set)
plt.scatter(X_test, y_test, color = "red")
# la recta de la regresion es la misma 
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()
