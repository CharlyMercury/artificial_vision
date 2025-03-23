# Arquitectura de una Red Neuronal Convolucional (CNN) en Keras

(En Visual Studio Code utilice Ctrl+Shift+V para 
visualizar el archivo.md )

Una arquitectura de red neuronal convolucional (CNN) comienza típicamente con una **capa convolucional**. Esta capa toma una imagen de entrada con un alto y ancho determinado, y genera una salida con una **profundidad** que depende del número de **filtros** (o kernels) aplicados. Cada filtro detecta patrones específicos dentro de la imagen, como bordes o texturas.

A continuación, la salida puede pasar a una **capa de max pooling** (`MaxPooling2D`) que se encarga de reducir las dimensiones espaciales (alto y ancho) de la imagen, manteniendo la información más relevante. Esto ayuda a reducir la complejidad del modelo y controlar el overfiting.

Después, se puede aplicar una **capa de Dropout**, que desactiva aleatoriamente una fracción de las neuronas durante el entrenamiento para evitar el sobreajuste.

Este conjunto de capas (Convolucional → MaxPooling → Dropout) se puede **repetir múltiples veces**, dependiendo de la profundidad y complejidad de la arquitectura que deseamos construir.

Una vez finalizadas las capas convolucionales, la salida se **aplana** mediante una **capa `Flatten`**, la cual transforma la representación multidimensional en un vector unidimensional.

Este vector es entonces procesado por una o más **capas densas** (`Dense`), las cuales permiten tomar decisiones o realizar la clasificación final del modelo. La última capa densa generalmente utiliza una función de activación como `softmax` para obtener probabilidades de pertenencia a cada clase.

## Ejemplo Completo en Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(rate=0.25),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])
```

### Descripción del modelo:
- **`Conv2D`**: Detecta características espaciales usando filtros.
- **`MaxPooling2D`**: Reduce la resolución espacial de las características.
- **`Dropout`**: Ayuda a prevenir el sobreajuste.
- **`Flatten`**: Aplana las características para ingresarlas en capas densas.
- **`Dense`**: Realiza decisiones finales, usualmente con `softmax` para clasificación.

Esta estructura modular permite construir CNNs flexibles y adaptables a diferentes tipos de problemas de visión por computadora.

