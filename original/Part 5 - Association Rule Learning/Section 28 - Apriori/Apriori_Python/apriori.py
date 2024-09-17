#Apriori


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):    # recorremos todas de las filas (7501-1)
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
# añadimos a para cada transaccion una lista de listas de los productos comprados (str)
# recorre todas las columnas haciendo un string
    
    
# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2,
                min_lift = 3, min_length = 2)    # ponemos los limites (son %)
# min_support = min numero de veces que se compran los prod (que se compren 1 vez no indica nada)
# calculamos para un min de 3 veces por semana (3*7/7500) = 0.003
# min_support = min veces que se cumpla el siguiente prod despues del primero
# min_lift = numero alto si solo queremos las reglas mas relevantes
# min_length = numero de productos asociados minimo (pan y aceite)


# Visualización de los resultados
results = list(rules)    # crea las reglas de asocicacion
print(results[4])    # sacamos las 4 primeras (las mas fuertes)

