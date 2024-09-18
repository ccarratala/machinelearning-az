# Upper Confidence Bound (UCB)


# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


# Algoritmo de UCB
import math
N = 10000    # usuarios del dataset (rondas)
d = 10       # numero de anuncios
number_of_selections = [0] * d    # vector de 0  aumentando segun se seleccione un anuncio en cada ronda
sums_of_rewards = [0] * d    # vecor de 0 aumentando segun hagan click (suma de recompensa)
ads_selected = []    # anuncios mostrados a cada usuario (en cada ronda) llega un punto que solo muestra uno (mejor)
total_reward = 0     # inicializamos a 0, luego el bucle la irá aumentando

for n in range(0, N):    # corremos 10000 rondas (usuarios)
    max_upper_bound = 0    # cada vez que inicie el for empieza con UCB=0, para luego quedarnos con el mayor
    ad = 0    # inicializamos los anuncios con 0 (luego será el elegido como mejor)
    
    for i in range(0, d):    # de 0 hasta 10 (numero de anuncios) muestra los 10 anuncios primero
    
        if(number_of_selections[i]>0):    # para que empiece a calcular después de haberse mostrado una vez (cuando sea positivo)
            average_reward = sums_of_rewards[i] / number_of_selections[i]     # recompensa media (formula) = linea roja
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])    # intervalo de confianza sup (formula) 
                                                                              # n+1 para que la ronda empiece en 1 y no de problemas
            upper_bound = average_reward + delta_i    # UCB
        else:
            upper_bound = 1e400    # le ponemos ese valor alto para que al menos todos los anuncios se seleccionen una vez en las 10 primeras rondas
            
        if upper_bound > max_upper_bound:    # si el actual supera al anterior, nos quedamos con ese anuncio
            max_upper_bound = upper_bound
            ad = i    # anuncio seleccionado (si es mejor que el anterior se cambia)
    
    ads_selected.append(ad)    # añado el mejor anuncio seleccionado para esa ronda
    number_of_selections[ad] = number_of_selections[ad] + 1    # actualizar numero de selecciones 
    reward = dataset.values[n, ad]    # recompensa que se le ha dado al anuncio elegido [ad] en la ronda n (dataset)
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward    # suma de las rewards (ya sea 1 o 0)
    total_reward = total_reward + reward    # suma acumulada de todas las rewards despues de las 10000 rondas
    
    
    
# Histograma de resultados: frecuencia de muestreo de cada anuncio
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
    
    
    
