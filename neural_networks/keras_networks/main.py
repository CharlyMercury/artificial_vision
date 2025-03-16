"""
    
    Módulo principal que inicia el entrenamiento de la red neuronal.

"""
from src.neural_keras import red_neuronal_keras


def main():
    """
    Función principal que inicia el entrenamiento de la red neuronal.
    """
    print("Iniciando el entrenamiento de la red neuronal...")
    red_neuronal_keras()


# Punto de entrada del script
if __name__ == "__main__":
    main()
