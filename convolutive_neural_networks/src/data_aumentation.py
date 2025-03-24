"""
    Data augmentation for the dataset

    Las operaciones de data aumentation que se pueden realizar son:
    a) Rotación de la imagen - fill_mode = 'nearest', 'constant', 'reflect' o 'wrap'
    b) width_shift_range - Rango de traslación horizontal
    c) height_shift_range - Rango de traslación vertical
    d) brightness_range - Rango de brillo
    e) zoom_range - Rango de zoom

"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = Path(__file__).parent
image_dir = parent_dir.parent / 'images'
train_dir = image_dir / 'train'


def data_augmentation():

    data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.4, 1.5]
    )

    img_path = image_dir / 'cat.jpeg'
    img = load_img(img_path)
    
    x = img_to_array(img)
    print(x.shape)
    x = x.reshape((1,) + x.shape)
    print(x.shape)

    i = 0
    qty_images = 10
    for batch in data_generator.flow(x, batch_size=1):
        # plt.figure()
        imgplot = plt.imshow(array_to_img(batch[0]))
        # plt.show()
        i += 1
        if i % qty_images == 0:
            break


"""def data_augmentation_from_directory():
    data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.4, 1.5]
    )

    img_path = image_dir / 'cat.jpeg'
    img = load_img(img_path)
    
    x = img_to_array(img)
    print(x.shape)
    x = x.reshape((1,) + x.shape)
    print(x.shape)

    i = 0
    qty_images = 10
    for batch in data_generator.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary'
    ):
        # plt.figure()
        imgplot = plt.imshow(array_to_img(batch[0]))
        # plt.show()
        i += 1
        if i % qty_images == 0:
            break"""