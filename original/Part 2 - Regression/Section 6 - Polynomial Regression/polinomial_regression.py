# Regresión Polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # se genera un vector si solo pones 1
                                  # para obtener matriz ponemos 1:2 (el 2 no lo cuenta)
y = dataset.iloc[:, 2].values
# solo necesitamos las dos ultimas columnas, no la categorica

''' En este ejmplo no tiene sentidodividir el dataset, necesitamos todos los ejemplos '''
''' Tampoco hace falta escalar las variables '''


# Ajustar la regresión lineal con todo el dataset (hacemos esto para comparar)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresión polinómica con todo el dataset
from sklearn.preprocessing import PolynomialFeatures  
 preprocessing tambien la usamos para escalar
poly_reg = PolynomialFeatures(degree = 4)   # le ponemos de grado 4 (por defecto 2)
# podemos ir tanteando con los degree, vamos incrementandolo
X_poly = poly_reg.fit_transform(X) # transformamos la X en polinomios
# modelo que se encargue del ajuste, se usa el linear
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)    # le pasamos las X polinomicas


# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
# vemos que no es el modelo correcto


# Visualización de los resultados del Modelo Polinómico
# creamos secuencia de datos para que la linea sea mas fluida (no a cachos segun la N)
X_grid = np.arange(min(X), max(X), 0.1) # de entre 0 y 10 con saltos de 0.1
X_grid = X_grid.reshape(len(X_grid), 1) # lo pasamos a vector en columna (traspuesto)

plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
# tiene que prdecir para las X_poly (en este caso es grid por lo anterior)
# podemos poner directamente .fit_transform(X_poly) porque ya se ha pasado a poly antes
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# Predicción de nuestros modelos con nuevos ejemplos
# en este caso solo le pasamos un nuevo dato: 6.5
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # hay que transformar el nuevo dato en poly






