"""
This script is used to train a Convolutional Neural Network (CNN) to classify the CIFAR-10 dataset.
"""
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from numpy import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Obteber el directorio actual
current_dir = Path(__file__).parent
models_dir = current_dir.parent / 'trained_model_parameters'

# LOGGING

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
    plt.imshow(x_train[0])
    plt.show()

    ## Vamos a normalizar las imágenes
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    
    ## Cantidad de clases
    n_clases = len(np.unique(y_train))
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
        filters = filtros, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w), 
        input_shape = x_train.shape[1:]))
    model.add(Activation('relu'))

    # Capa convolucional 2
    model.add(Conv2D(
        filters = filtros, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # Capa convolucional 3
    model.add(Conv2D(
        filters = filtros*2, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Capa convolucional 4
    model.add(Conv2D(
        filters = filtros*2, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # Capa convolucional 5
    model.add(Conv2D(
        filters = filtros*4, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))

    # Capa convolucional 6
    model.add(Conv2D(
        filters = filtros*4, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    # Capa de flatten
    model.add(Flatten())
    model.add(Dense(n_clases, activation='softmax'))
    
    # Resumen del modelo
    model.summary()

    # Página de visualización de modelos de redes neuronales 
    # convolucionales: https://poloclub.github.io/cnn-explainer/

    # Compilación del modelo
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='accuracy',
        patience=2,
        verbose=1)
    
    model_checkpoint = ModelCheckpoint(
        filepath=models_dir / 'bestcifar10.keras',
        monitor='accuracy',
        save_best_only=True,
        verbose=1)

    # Entrenamiento del modelo
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(x_valid, y_valid),
        verbose = 2,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping])
 
    # Guardar el modelo
    model_path = models_dir / 'cifar10.keras'
    model.save(model_path)

    # Gráfica de la precisión del modelo
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.show()

    # Evaluación del modelo
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
