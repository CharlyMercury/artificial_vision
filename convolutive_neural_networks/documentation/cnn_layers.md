# Componentes de una Red Neuronal Convolucional en Keras

(En Visual Studio Code utilice Ctrl+Shift+V para 
visualizar el archivo.md )

En este documento, se describen las capas comunes utilizadas en una red neuronal convolucional (CNN) implementada con Keras, junto con sus parámetros principales y un ejemplo completo al final.

## 1. Capa Convolucional (`Conv2D`)

La capa convolucional es fundamental en una CNN y se utiliza para extraer características de las imágenes de entrada. Los parámetros principales de esta capa en Keras son:

- **`filters`**: Número de filtros (o kernels) que se aplicarán a la imagen de entrada. Cada filtro detecta una característica específica.
- **`kernel_size`**: Tamaño de cada filtro. Comúnmente se utilizan tamaños como (3,3) o (5,5).
- **`strides`**: Desplazamiento del filtro sobre la imagen. Un valor común es (1,1), lo que significa que el filtro se mueve un píxel a la vez.
- **`padding`**: Método de padding utilizado. Puede ser 'valid' (sin padding) o 'same' (padding que mantiene las dimensiones de la entrada).
- **`activation`**: Función de activación aplicada después de la convolución, como 'relu' o 'sigmoid'.

Ejemplo en Keras:

```python
from tensorflow.keras.layers import Conv2D

conv_layer = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')
```

## 2. Capa de Pooling (`MaxPooling2D`)

Podemos utilizar MaxPooling o AveragePooling, por ahora, solo utilizaremos MaxPooling2d.

La capa de pooling se utiliza para reducir las dimensiones espaciales de las características, disminuyendo la carga computacional y controlando el sobreajuste. Los parámetros principales son:

- **`pool_size`**: Tamaño de la ventana de pooling, comúnmente (2,2).
- **`strides`**: Desplazamiento de la ventana de pooling sobre la imagen. Si no se especifica, por defecto es igual a `pool_size`.
- **`padding`**: Método de padding utilizado, similar a la capa convolucional.

Ejemplo en Keras:

```python
from tensorflow.keras.layers import MaxPooling2D

pooling_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')
```

## 3. Capa de Dropout (`Dropout`)

La capa de dropout se utiliza para reducir el sobreajuste durante el entrenamiento de la red. Desactiva aleatoriamente una fracción de las neuronas en cada iteración. El parámetro principal es:

- **`rate`**: Fracción de neuronas que se desactivarán en cada paso de entrenamiento. Por ejemplo, un valor de 0.5 indica que el 50% de las neuronas se desactivarán aleatoriamente en cada iteración.

Ejemplo en Keras:

```python
from tensorflow.keras.layers import Dropout

dropout_layer = Dropout(rate=0.5)
```

## 4. Capa de Aplanamiento (`Flatten`)

La capa de aplanamiento convierte una matriz de características multidimensional en un vector unidimensional, preparándola para las capas densas posteriores.

Ejemplo en Keras:

```python
from tensorflow.keras.layers import Flatten

flatten_layer = Flatten()
```

## 5. Capa Densa (`Dense`)

La capa densa es una capa totalmente conectada donde cada neurona está conectada a todas las neuronas de la capa anterior. Se utiliza para realizar la clasificación o regresión final. Los parámetros principales son:

- **`units`**: Número de neuronas en la capa.
- **`activation`**: Función de activación aplicada a las neuronas, como 'relu', 'sigmoid' o 'softmax'.

Ejemplo en Keras:

```python
from tensorflow.keras.layers import Dense

dense_layer = Dense(units=128, activation='relu')
```

## Ejemplo Completo de una CNN en Keras

A continuación, se presenta un ejemplo de una CNN simple que combina las capas mencionadas:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])
```

En este ejemplo:

- La capa `Conv2D` aplica 32 filtros de tamaño 3x3 a la imagen de entrada de 64x64 píxeles con 3 canales de color (RGB).
- La capa `MaxPooling2D` reduce las dimensiones espaciales a la mitad.
- La capa `Dropout` desactiva el 25% de las neuronas durante el entrenamiento para prevenir el sobreajuste.
- La capa `Flatten` convierte las características 2D en un vector 1D.
- Las capas `Dense` realizan la clasificación final en 10 clases con una función de activación 'softmax'.

Este modelo ejemplifica cómo se integran las diferentes capas en una CNN para tareas de clasificación de imágenes.

