import numpy as np
import matplotlib.pyplot as plt

def graficar_tanh(ax):

    # Definimos la función Tangente Hiperbólica (TanH)
    def tanh(x):
        
        #Calcula la función Tangente Hiperbólica (TanH).
        #La TanH mapea cualquier valor a un rango entre -1 y 1.
        
        return np.tanh(x)
    
    # Definimos la derivada de la función TanH
    def derivada_tanh(x):
        
        #Calcula la derivada de la función Tangente Hiperbólica (TanH).
       
        return 1 - np.tanh(x)**2  # Derivada de TanH
    
    # Generamos un rango de valores para x
    x = np.linspace(-10, 10, 1000)  # Crea 1000 puntos entre -10 y 10
    
    # Calculamos los valores de la función TanH y su derivada
    y = tanh(x)
    y_derivada = derivada_tanh(x)
    
    # Graficamos la función TanH
    ax.plot(x, y, label="TanH", color="red")
    
    # Graficamos la derivada de la función TanH como una línea punteada
    ax.plot(x, y_derivada, label="Derivada de TanH", linestyle="--", color="blue")
    
    # Se configura la gráfica
    ax.set_title("Función TanH y su Derivada")
    ax.set_xlabel("x")
    ax.set_ylabel("TanH(x)")
    ax.grid(True)
    ax.legend(loc="upper left")


  