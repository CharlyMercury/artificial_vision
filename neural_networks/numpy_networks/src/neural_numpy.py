import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

# Crear datasets desde cero - Para un ejemplo de clasificación
def create_dataset(N=1000):
    """
    Genera un conjunto de datos sintético con dos clases usando distribuciones gaussianas.
    
    Parámetros:
    N (int): Número de muestras a generar.
    
    Retorna:
    X (numpy.ndarray): Matriz de características.
    Y (numpy.ndarray): Vector de etiquetas con valores 0 o 1.
    """
    gaussian_quantiles = make_gaussian_quantiles(
        mean=None,
        cov=0.1,
        n_samples=N,
        n_features=2,
        n_classes=2,
        shuffle=True,
        random_state=None
    )
    X, Y = gaussian_quantiles
    Y = Y[:, np.newaxis]  # Convertir a matriz columna
    return X, Y

# Funciones de activación
def sigmoid(x, derivate=False):
    """
    Función de activación sigmoide.
    
    Parámetros:
    x (numpy.ndarray): Entrada a la función de activación.
    derivate (bool): Si es True, calcula la derivada de la función sigmoide.
    
    Retorna:
    numpy.ndarray: Salida de la sigmoide o su derivada.
    """
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1)**2
    else:
        return 1 / (1 + np.exp(-x))

def relu(x, derivate=False):
    """
    Función de activación ReLU.
    
    Parámetros:
    x (numpy.ndarray): Entrada a la función de activación.
    derivate (bool): Si es True, calcula la derivada de la función ReLU.
    
    Retorna:
    numpy.ndarray: Salida de la ReLU o su derivada.
    """
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)

# Función de pérdida
def mse(y, y_hat, derivate=False):
    """
    Calcula el error cuadrático medio (MSE).
    
    Parámetros:
    y (numpy.ndarray): Valores reales.
    y_hat (numpy.ndarray): Predicciones.
    derivate (bool): Si es True, calcula la derivada del MSE.
    
    Retorna:
    float o numpy.ndarray: Valor del error o su derivada.
    """
    if derivate:
        return (y_hat - y)
    else:
        return np.mean((y_hat - y)**2)

# Inicialización de pesos y sesgos
def initialize_parameters_deep(layers_dims):
    """
    Inicializa los pesos y sesgos de la red neuronal.
    
    Parámetros:
    layers_dims (list): Lista con el número de neuronas en cada capa.
    
    Retorna:
    dict: Diccionario con los parámetros de la red (pesos y sesgos).
    """
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1
        parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
    return parameters

# Entrenamiento de la red neuronal
def train(x_data, y_data, learning_rate, params, training=True):
    """
    Propagación hacia adelante y hacia atrás para el entrenamiento de la red neuronal.
    
    Parámetros:
    x_data (numpy.ndarray): Datos de entrada.
    y_data (numpy.ndarray): Etiquetas reales.
    learning_rate (float): Tasa de aprendizaje.
    params (dict): Diccionario con los parámetros de la red.
    training (bool): Si es True, realiza el ajuste de pesos y sesgos.
    
    Retorna:
    numpy.ndarray: Salida de la red neuronal.
    """
    params['A0'] = x_data

    # Propagación hacia adelante
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if training:
        # Backpropagation
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

        # Actualización de parámetros usando gradiente descendente
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate

        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output

# Función principal para entrenar el modelo
def train_model():
    """
    Entrena la red neuronal y visualiza los datos.
    """
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]
    params = initialize_parameters_deep(layers_dims)
    error = []

    for _ in range(50000):
        output = train(X, Y, 0.001, params)
        if _ % 50 == 0:
            print(mse(Y, output))
            error.append(mse(Y, output))

    # Graficar los datos
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
