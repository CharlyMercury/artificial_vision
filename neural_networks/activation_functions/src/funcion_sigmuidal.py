import numpy as np
import matplotlib.pyplot as plt

def graficar_sigmoide(ax):

    # Definimos la función sigmoide
    def sigmoide(x):
        """
        Calcula la función sigmoide.
        La sigmoide mapea cualquier valor a un rango entre 0 y 1.
        """
        return 1 / (1 + np.exp(-x))
    
    # Definimos la derivada de la función sigmoide
    def derivada_sigmoide(x):

        return sigmoide(x) * (1 - sigmoide(x))  # Derivada de la sigmoide
    
    # Generamos un rango de valores para x
    x = np.linspace(-10, 10, 1000)  # Crea 1000 puntos entre -10 y 10
    
    # Calculamos los valores de la función sigmoide y su derivada
    y = sigmoide(x)
    y_derivada = derivada_sigmoide(x)
    
    # Graficamos la función sigmoide
    ax.plot(x, y, label="Sigmoide", color="blue")
    
    # Graficamos la derivada de la función sigmoide como una línea punteada
    ax.plot(x, y_derivada, label="Derivada de la Sigmoide", linestyle="--", color="red")
    
    # Se configura la gráfica
    ax.set_title("Función Sigmoide y su Derivada")
    ax.set_xlabel("x")
    ax.set_ylabel("Sigmoide(x)")
    ax.grid(True)
    ax.legend(loc="upper left")