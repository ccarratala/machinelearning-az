# Redes Neuronales Convolucionales


# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras  

# Necesitamos carpetas train y test set etiquetadas 
# con dos carpetas de imagenes de perros y gatos (o de lo que queramos)
# Esa sería la base de datos
# Con keras importamos y preprocesamos las imagenes


# Parte 1 - Construir el modelo de CNN

# Importar las librerías y paquetes (2D porque son en b/n (si es a color 3D)
from keras.models import Sequential  # inicializar los pesos
from keras.layers import Conv2D      # capa convolucion (filtros)
from keras.layers import MaxPooling2D  # capa max pooling
from keras.layers import Flatten  # hace vector
from keras.layers import Dense    # full connection


# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))
# empezar con pocos filtros para ir tatneando, luego aumentar si va bien
# ojo si las img no son cuadradas (input_shape, tamaño + 3 canales d color)
# tambien si son a color especificar el num de canales 
# a veces no es util analizarlas a color ej. hay perros con diferntes colores


# Paso 2 - Max Pooling
# reduce a la mitad no perdemos demasiada info pero reducimos bien 
# 2x2 es por defecto (no hace falta ponerlo) es el size recomendado
classifier.add(MaxPooling2D(pool_size = (2,2)))  

# Antes de añadir estas capas la precision no superaba el 80% (bad)
# una solucion sería añadir más capas de conv o en el paso 4 hacerla más densa
# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Practica habitual: dos filtros de conv de 32 y una de 64


# Paso 3 - Flattening (pasa a 1D = nodos de la CNN)
classifier.add(Flatten())


# Paso 4 - Full Connection (hidden y output layers)
classifier.add(Dense(units = 128, activation = "relu"))
# units = num de nodos hidden (aprox media de nodos input + output - recomendacion usar potencia de 2)

classifier.add(Dense(units = 1, activation = "sigmoid"))
# clasificaicon binaria = con poner 1 vale
# si queremos saber la prob de perro o gato = usar softmax y poner 2


# Compilar la CNN (unificar todo)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# algoritmo estocasatico = adam (tipo gradient)
# entropia cruzada para calcular error (binaria porque solo tenemos 2 categ)
# usar precision como metrica



# Parte 2 - Ajustar la CNN a las imágenes para entrenar 
''' Ver la documentacion de keras '''
from keras.preprocessing.image import ImageDataGenerator

# limpiar las imagenes
# transformamos las img para tener un abanico mas grande de img
train_datagen = ImageDataGenerator(
        rescale=1./255,  # pixeles tamaño con decimales
        shear_range=0.2,
        zoom_range=0.2,  # % de zoom
        horizontal_flip=True)  # voltea img

test_datagen = ImageDataGenerator(rescale=1./255)    # estas no se modifican porque tiene que ser capaz de identificarlas

# carga de carpetas (poner el directorio donde esten)
training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),  # mantener proporcion que en las capas
                                                    batch_size=32,         # capas
                                                    class_mode='binary')   # 2 categ

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# Ajustamos el modelo con train_set con el modelo CNN creado 
classifier.fit_generator(training_dataset,
                        steps_per_epoch=8000,    # num img que toma por pasada (ponemos el total que tenemos)
                        epochs=25,    
                        validation_data=testing_dataset,  # validamos con el test
                        validation_steps=2000)    # num img que toma (elegimos todas)



