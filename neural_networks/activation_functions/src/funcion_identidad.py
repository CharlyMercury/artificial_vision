import numpy as np
import matplotlib.pyplot as plt

def graficar_identidad(ax):
    
    # Definimos la función identidad
    def identidad(x):
    
        #Se calcula la función identidad.
        #La función identidad devuelve el mismo valor de entrada.
     
        return x
    
    # Definimos la derivada de la función identidad
    def derivada_identidad(x):
        
        #Calcula la derivada de la función identidad.
        #La derivada de la identidad es 1 para todos los valores de x.
      
        return np.ones_like(x)  # La derivada de x es 1 para todo x
    
    # Generamos un rango de valores para x
    x = np.linspace(-10, 10, 1000)  # Crea 1000 puntos entre -10 y 10
    
    # Calculamos los valores de la función identidad y su derivada
    y = identidad(x)
    y_derivada = derivada_identidad(x)
    
    # Graficamos la función identidad
    ax.plot(x, y, label="Identidad", color="orange")
    
    # Graficamos la derivada de la función identidad como una línea punteada
    ax.plot(x, y_derivada, label="Derivada de la Identidad", linestyle="--", color="lime")
    
    # Se configura la gráfica
    ax.set_title("Función Identidad y su Derivada")
    ax.set_xlabel("x")
    ax.set_ylabel("Identidad(x)")
    ax.grid(True)
    ax.legend(loc="upper left")