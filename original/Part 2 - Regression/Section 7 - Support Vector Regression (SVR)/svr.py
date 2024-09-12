# SVR

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


'''En este caso no hace falta dividir el dataset en subgrupos'''

# Importante: Escalado de variables
from sklearn.preprocessing import StandardScaler
# creamos 2 escaladores porqu las escalas para la X y la y son diferentes
sc_X = StandardScaler()  
sc_y = StandardScaler()  
X = sc_X.fit_transform(X)  # stand las X
y = sc_y.fit_transform(y.reshape(-1,1))  # stand la y
                        # reshape para que sea vector col y no de error


# Ajustar modelo SVR con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf") # gaussian kernel (depende de la distribucion de datos)
regression.fit(X, y)

# Predicción de nuestros modelos con SVR
# hay que escalar el nuevo ejemplo tambien
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
# usamos inverse_transform para que nos devuelva la y sin escalar (para entenderlo mejor)
# ponemos np.array por si queremos predecir mas de un ejemplo (si no, no hace falta)

# Visualización de los resultados del SVR
''' Intentar poner los ejes con invers_transform '''
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

''' No predice bien para el ultimo valor porque lo toma como un outlier, 
el modelo puede crecer a la misma velocidad que el dataset '''
