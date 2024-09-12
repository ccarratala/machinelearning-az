# Regresión con Árboles de Decisión

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No hace falta dividir 
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Tmapoco haría falta escalar, pero se puede hacer sin y luego probar con el escalado
# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Arbol de decision para regresiones
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
# en principio se hacen cortes en vertical y horizontal 
# todo eso se puede configurar con los parametros
regression.fit(X, y)

# Predicción de un nuevo ejemplo
y_pred = regression.predict([[6.5]])
print(y_pred)

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")  # recomendable poner las X en vez del grid para visualizar mejor
# poner X_grid para ver los niveles que ha hecho, da un resultado escalonado (parte por la mitad de los niveles)
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

''' Vemos que la prediccion parece perfecta, raro = overfitting
    no podra luego predecir parar nuevos ejemplos
    Esto se soluciona con los Random Forests'''

