"""
This script is used to train a Convolutional Neural Network (CNN) to classify the CIFAR-10 dataset.
"""
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint
import numpy
import matplotlib.pyplot as plt


def cifar_classification():
    """
    This function trains a Convolutional Neural Network (CNN) to classify the CIFAR-10 dataset.
    """

    # Cargamos el dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # Cifar10 tiene 10 clases de objetos
    plt.imshow(x_train[2])
    plt.show()

    ## Vamos a normalizar las imágenes
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    
    ## Cantidad de clases
    n_clases = len(numpy.unique(y_train))
    y_train = to_categorical(y_train, n_clases)
    y_test = to_categorical(y_test, n_clases)

    # Division del set de entrenamiento 
    # en entrenamiento y validación
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    filtros = 32
    regularizers_w  = 1e-4

    # Definición del modelo
    model = Sequential()
    
    # Capa convolucional 1
    model.add(Conv2D(
        filtros, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers), 
        x_train.shape[1:]))
    model.add(Activation('relu'))

    # Capa convolucional 2
    model.add(Conv2D(
        filtros, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Capa convolucional 3
    model.add(Conv2D(
        filtros*2, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Capa convolucional 4
    model.add(Conv2D(
        filtros*2, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Capa convolucional 5
    model.add(Conv2D(
        filtros*4, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers)))
    model.add(Activation('relu'))

    # Capa convolucional 6
    model.add(Conv2D(
        filtros*4, 
        (3, 3), 
        'same', 
        regularizers.l2(regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))