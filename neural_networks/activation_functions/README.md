# 📌 Funciones de Activación para Redes Neuronales

## 📚 Información General

Este proyecto contiene la implementación y visualización de ocho funciones de activación comúnmente utilizadas en redes neuronales, junto con sus respectivas derivadas. La implementación se realizó en **Python 3.11**, utilizando las bibliotecas **NumPy** y **Matplotlib** para los cálculos y generación de gráficos.

---
---
## 📋 Requisitos
Para ejecutar este proyecto, es necesario instalar las siguientes bibliotecas de Python:

- **NumPy:** Para realizar cálculos numéricos.
- **Matplotlib:** Para generar y visualizar gráficas.

Instalación:
```bash
pip install numpy matplotlib
```

---
## 🗂️ Estructura del Proyecto
```
TAREA_3/
│
├── scr/
│   ├── Function_Identidad.py        # Función Identidad y su derivada
│   ├── Function_Relu.py             # Función ReLU y su derivada
│   ├── Function_Sigmuidal.py        # Función Sigmoide y su derivada
│   ├── Function_Sinusoidal.py       # Función Sinusoidal y su derivada
│   ├── Function_TanH.py             # Función Tangente Hiperbólica y su derivada
│
├── .gitignore                       # Archivos ignorados por Git
├── main.py                          # Script principal para generar gráficas
├── README.md                        # Documentación del proyecto
└── Requirements.txt                 # Lista de dependencias
```

---
## ⚙️ Funcionalidades
Este proyecto incluye la implementación y graficado de las siguientes funciones de activación:


. **Sigmoide (Sigmoid Function)**  
   - **Función:** Mapea valores a un rango entre 0 y 1.
   - **Derivada:** sigmoide(x) * (1 - sigmoide(x)).

. **Tangente Hiperbólica (TanH)**  
   - **Función:** Mapea valores a un rango entre -1 y 1.
   - **Derivada:** 1 - tanh(x)^2.

. **ReLU (Rectified Linear Unit)**  
   - **Función:** Devuelve 0 si x < 0; caso contrario, devuelve x.
   - **Derivada:** 0 si x < 0; 1 si x >= 0.


 **Sinusoidal (Sinusoidal Function)**  
   - **Función:** sin(x).
   - **Derivada:** cos(x).

.**Identidad (Identity Function)**  
   - **Función:** Devuelve el mismo valor de entrada.
   - **Derivada:** 1 en todos los valores de x.

---
## 🚀 Instrucciones de Ejecución
Sigue estos pasos para clonar el repositorio, instalar las dependencias y generar las gráficas:

1. **Clonar el repositorio** 🖥️:
```bash
git clone https://github.com/EmanueldCristo/Aactivation_functions.git
```

2. **Crear y activar un entorno virtual** 🛠️:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv nombre_del_entorno
     .\nombre_del_entorno\Scripts\Activate
     ```
   - **Unix/Linux/macOS:**
     ```bash
     python -m venv nombre_del_entorno
     source nombre_del_entorno/bin/activate
     ```
   - Para desactivar el entorno:
     ```bash
deactivate
```

3. **Instalar las dependencias** 📦:
```bash
pip install -r Requirements.txt
```

4. **Ejecutar el script principal** 🚀:
```bash
python main.py
```

5. **Visualizar las gráficas** 📊:
   - Las gráficas se mostrarán en una ventana emergente.
   - Para guardar las imágenes, puedes modificar `main.py` agregando:
     ```python
     plt.savefig("ruta/de/la/grafica.png")
     ```

---
## 🛠️ Tecnologías Utilizadas
- **Python 3.12.4**
- **NumPy** (cálculos numéricos)
- **Matplotlib** (visualización de gráficos)

---

### bye