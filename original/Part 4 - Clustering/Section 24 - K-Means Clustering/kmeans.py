# K-Means


# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values    # ingresos / score (nivel de gasto 1-100)

''' No hace falta dividir el conjunto de datos
    Solo se le pasa la X, no las etiquetas'''

# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):   # calculamos k-means para clusters del 1 al 10
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    # inicializacion, num max d iteraciones (stop por si nuna deja de mover el centroide)
    # n_init = num veces que hará k-means 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    # añadimos al final el suma de cuadrados

plt.plot(range(1,11), wcss)    # eje x: 1-10 // eje y: wcss
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)    # se le pasa el fit y el predict


# Visualización de los clusters
# hacemos un scatter de las 5 categorías (nombradas por nosotros o etiqueta si hay)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cautos")
#   cluster 0 metemos la col 0    cluster 0 col 1     size 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Centroides")
# pintamos los centroides de cada cluster / todas las filas col 0 (x) col 1 (y)
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()

