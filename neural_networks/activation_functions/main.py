"""
    Módulo principal para graficar todas las funciones de activación
"""
import matplotlib.pyplot as plt
from src.funcion_identidad import graficar_identidad
from src.funcion_relu import graficar_relu
from src.funcion_sigmuidal import graficar_sigmoide
from src.funcion_sinusoidal import graficar_sinusoidal
from src.funcion_tanh import graficar_tanh


def graficar_todo():
    """
        Entry point
    """
    # Crear una figura con 2 filas y 4 columnas de subgráficas
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Aju sta el tamaño si es necesario
    # Graficar cada función en su respectiva subgráfica
    graficar_sigmoide(axs[0, 0])
    graficar_relu(axs[0, 1])
    graficar_tanh(axs[0, 2])
    graficar_identidad(axs[1, 0])
    graficar_sinusoidal(axs[1, 1])

    # Ajustar el espacio entre subgráficas
    plt.tight_layout()

    # Mostrar la figura con todas las gráficas
    plt.show()


if __name__ == "__main__":
    graficar_todo()
