import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Datos de entrenamiento
(train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

# Imagen de entrenamiento (1)
print(train_data_x.shape)
print(train_labels_y[0])

### Convert Matplotlib figure to OpenCV format
### export DISPLAY=:0
#fig, ax = plt.subplots()
#fig.canvas.draw()
#image = train_data_x[0]
#cv2.imshow("Matplotlib Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Arquitectura de la red
model = Sequential([
	Input(shape=(28*28,)),
	Dense(512, activation='relu'),
	Dense(10, activation='softmax')
])

# Compilacion
model.compile(
    optimizer='rmsprop', 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Resumen
model.summary()

# Arquitectura de la red
model = Sequential([
	Input(shape=(28*28,)),
	Dense(512, activation='relu'),
	Dense(10, activation='softmax')
])

# Compilacion
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Resumen
model.summary()

# Normalizacion de nuestros datos
x_train = train_data_x.reshape(60000, 28*28)
x_train = x_train.astype('float32')/255
y_train = to_categorical(train_labels_y)

# Datos de prueba
x_test = test_data_x.reshape(10000, 28*28)
x_test = x_test.astype('float32')/255
y_test = to_categorical(test_labels_y)

# Entrenamiento
model.fit(x_train, y_train, epochs=8, batch_size=128)

model.evaluate(x_test, y_test)
print("HOla mundo")
