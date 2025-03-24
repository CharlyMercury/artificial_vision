"""

    This module contains the convolution kernels used in the convolutional layers of the CNN.

    Main python modules used:
    - numpy (pip install numpy)
    - opencv-python (pip install opencv-python)
    - matplotlib (pip install matplotlib)

    Página de ayuda
    - https://setosa.io/ev/image-kernels/

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Diccionario de Kernels de Convolución
kernels = {
    "Identidad": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),

    "Blur": (1 / 9) * np.ones((3, 3)),

    "Gaussiano": (1 / 16) * np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]),

    "Sobel_x": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]),

    "Sobel_y": np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ]),

    "Laplaciano": 3* np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]),

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
}


kernel_choose_list = list(kernels.keys())


def convolution_kernel():
    """
        Apply different convolution kernels to an image.
    """

    # Cargamos la imagen utilizando OpenCV

    current_path = Path(__file__).parent
    images_folder = current_path.parent / "images"
    image_path = images_folder / 'example_1.jpg'
    img = cv2.imread(image_path)

    # Convertimos la imagen a RGB (viniendo del formato que lee opencv)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img_rgb.shape)

    # Convertimos la imagen a escalas de gris (para detección de bordes)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)

    # Configuramos los plots
    axs_orgl = plt.subplots(1, 2, figsize=(16, 8))[1]

    axs_orgl[0].imshow(img_rgb)
    axs_orgl[0].set_title("Imagen original")
    axs_orgl[0].axis('off')

    axs_orgl[1].imshow(img_gray, cmap='gray')
    axs_orgl[1].set_title("Imagen escala de grises")
    axs_orgl[1].axis('off')

    plt.show()

    # Creamos una figura con 2 filas y 4 columnas
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    # Aplicamos cada kernels del diccionario a la imagen
    for i, (name, kernel) in enumerate(kernels.items()):
        if 'Sobel' in name or 'Laplaciano' in name:
            img_to_use = img_gray
        else:
            img_to_use = img_rgb

        convolved = cv2.filter2D(src=img_to_use, ddepth=-1, kernel=kernel)

        cmap_type = 'gray' if len(convolved.shape) == 2 else None

        axs[i].imshow(convolved, cmap=cmap_type)
        axs[i].set_title(name)
        axs[i].axis('off')

    # Quitamos los ejes de la última imagen
    axs[-1].axis('off')
    plt.tight_layout()
    plt.show()


def convolution_video():

    kernel = choose_one_kernel()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
    
        if not ret:
            break
    
        filtered = cv2.filter2D(frame, -1, kernel)
        cv2.imshow("Filtered Video", filtered)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def choose_one_kernel():

    while True:

        kernel_choose = input("Choose one kernel to apply: \n"
                                "0. Identidad (imagen original)\n"
                                "1. Blur (para suavizar el ruido)\n"
                                "2. Gaussiano (suavizado natural)\n"
                                "3. Sobel_x (detecta bordes horizontales)\n"
                                "4. Sobel_y (detecta bordes verticales)\n"
                                "5. Laplaciano (detecta bordes)\n"
                                "6. Sharpen (realza detalles):\n"
                                "Choose one option: ")
        
        try:
            kernel = kernels.get(kernel_choose_list[int(kernel_choose)])
            return kernel
        except Exception as err:
            print(" \n Opción Inválida \n ")
    