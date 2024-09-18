# Muestreo Thompson


# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


# Algoritmo de Muestreo Thompson (ver comentarios en UCB)
import random
N = 10000
d = 10
number_of_rewards_1 = [0] * d    # veces de reward 1 (formula) * anuncio
number_of_rewards_0 = [0] * d    # veces de no reward 0 (formula) * anuncio
ads_selected = []
total_reward = 0

for n in range(0, N):
    max_random = 0    # max valores aleat encontrados para cada ronda (posibles medias)
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        # generamos 10 valores aleatorios de distribucion beta (formula)
        
        if random_beta > max_random:    # si esos valores superan el aleat se sustituye
            max_random = random_beta
            ad = i    # anuncio seleccionado como mejor probabilidad
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:    # se suma dependiendo de si la reward es 1 o 0 en una var diferente
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward    # suma acumulada de rewards 1



# Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
