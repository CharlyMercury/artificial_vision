import numpy as np
import matplotlib.pyplot as plt

def graficar_sinusoidal(ax):

    # Definimos la función sinusoidal
    def sinusoidal(x):
        
        #Se calcula la función sinusoidal (seno).
        #La función sinusoidal oscila entre -1 y 1.
       
        return np.sin(x)
    
    # Definimos la derivada de la función sinusoidal
    def derivada_sinusoidal(x):
       
        #Calcula la derivada de la función sinusoidal.
        #La derivada de sin(x) es cos(x).
        
        return np.cos(x)  # Derivada de sin(x) es cos(x)
    
    # Generamos un rango de valores para x
    x = np.linspace(-10, 10, 1000)  # Crea 1000 puntos entre -10 y 10
    
    # Calculamos los valores de la función sinusoidal y su derivada
    y = sinusoidal(x)
    y_derivada = derivada_sinusoidal(x)
    
    # Graficamos la función sinusoidal
    ax.plot(x, y, label="Sinusoidal", color="lime")
    
    # Graficamos la derivada de la función sinusoidal como una línea punteada
    ax.plot(x, y_derivada, label="Derivada de la Sinusoidal", linestyle="--", color="purple")
    
    # Se configura la gráfica
    ax.set_title("Función Sinusoidal y su Derivada")
    ax.set_xlabel("x")
    ax.set_ylabel("Sinusoidal(x)")
    ax.grid(True)
    ax.legend(loc="upper left")
    
