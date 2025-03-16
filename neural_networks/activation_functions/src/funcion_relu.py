import numpy as np
import matplotlib.pyplot as plt

def graficar_relu(ax):

    # Definimos la función ReLU
    def relu(x):
        """
        Calcula la función ReLu
        - Devuelve 0 si x < 0.
        - Devuelve x si x >= 0.
        """
        return np.maximum(0, x)
    
    # Definimos la derivada de la función ReLU
    def derivada_relu(x):
        """
        Calcula la derivada de la función ReLU.
        - La derivada es 0 si x < 0.
        - La derivada es 1 si x >= 0.
        """
        return np.where(x < 0, 0, 1)  # Derivada de ReLU
    
    # Generamos un rango de valores para x
    x = np.linspace(-10, 10, 1000)  # Crea 1000 puntos entre -10 y 10
    
    # Calculamos los valores de la función ReLU y su derivada
    y = relu(x)
    y_derivada = derivada_relu(x)
    
    # Graficamos la función ReLU
    ax.plot(x, y, label="ReLu", color="green")
    
    # Graficamos la derivada de la función ReLU como una línea punteada
    ax.plot(x, y_derivada, label="Derivada de ReLu", linestyle="--", color="orange")
    
    # Se configura la gráfica
    ax.set_title("Función ReLU y su Derivada")
    ax.set_xlabel("x")
    ax.set_ylabel("ReLu(x)")
    ax.grid(True)
    ax.legend(loc="upper left")