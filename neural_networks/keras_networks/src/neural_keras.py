import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input   # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore


def red_neuronal_keras():
    """
    Función que crea y entrena una red neuronal con Keras para clasificar imágenes de dígitos escritos a mano.
    
    """

    # Cargamos el conjunto de datos MNIST, que tiene imágenes de dígitos escritos a mano
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Mostramos información sobre los datos de entrenamiento

    # Tamaño de las imágenes de entrenamiento (60000 imágenes de 28x28 píxeles)
    print("Forma de los datos de entrenamiento:", train_data_x.shape)

    # Etiqueta de la primera imagen (qué número es)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[10])

    # Tamaño de las imágenes de prueba (10000 imágenes de 28x28 píxeles)
    print("Forma de los datos de prueba:", test_data_x.shape)

    # Mostramos una de las imágenes de entrenamiento para ver cómo se ve
    plt.imshow(train_data_x[10])  # Mostramos la imagen en escala de grises
    plt.title("Ejemplo de una Imagen de Entrenamiento")  # Le ponemos un título a la imagen
    plt.show()  # Mostramos la imagen en una ventana

    # Creamos la red neuronal
    model = Sequential([
        Input(shape=(28 * 28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Configuramos el modelo para que sepa cómo aprender
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Mostramos un resumen de cómo está construida la red neuronal
    print("Resumen del modelo:")
    model.summary()  # Nos dice cuántas capas tiene, cuántos parámetros, etc.

    # Preparamos los datos de entrenamiento para que la red pueda usarlos
    # Aplanamos las imágenes de 28x28 a vectores de 784 valores
    x_train = train_data_x.reshape(60000, 28 * 28)  
    # Normalizamos los valores de los píxeles para que estén entre 0 y 1
    x_train = x_train.astype('float32') / 255  
    # Convertimos las etiquetas a un formato especial (one-hot encoding)
    y_train = to_categorical(train_labels_y)  

    # Preparamos los datos de prueba de la misma manera
    x_test = test_data_x.reshape(10000, 28 * 28)  # Aplanamos las imágenes de prueba
    x_test = x_test.astype('float32') / 255  # Normalizamos los valores de los píxeles
    y_test = to_categorical(test_labels_y)  # Convertimos las etiquetas de prueba

    # Entrenamos la red neuronal
    print("Entrenando la red neuronal...")
    # Entrenamos durante 10 épocas, usando paquetes de 128 imágenes
    model.fit(x_train, y_train, epochs=10, batch_size=128)  

    # Evaluamos la red neuronal con los datos de prueba
    print("Evaluando la red neuronal...")
    # Calculamos la pérdida y la precisión en el conjunto de prueba
    loss, accuracy = model.evaluate(x_test, y_test)  
    print(f"Pérdida: {loss}, Precisión: {accuracy}")  # Mostramos los resultados

    plt.show()
