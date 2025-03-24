"""
    Main file for the project.  
    
    This file is used to run the project and test the CNN.

"""
import sys
from src.managing_images import managing_images
from src.convolution_kernels import convolution_kernel, choose_one_kernel
from src.cifar_classification import cifar_classification
from src.data_aumentation import data_augmentation


def run(program_to_run):
    """
        Run the project.
    """
    if program_to_run == 'conv_ker':
        convolution_kernel()
    elif program_to_run == 'mng_imgs':
        managing_images()
    elif program_to_run == 'conv_video':
        choose_one_kernel()
    elif program_to_run == 'cifar':
        cifar_classification()
    elif program_to_run == 'data_augmentation':
        data_augmentation()


if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['mng_imgs', 'conv_ker', 'conv_video', 'cifar', 'data_augmentation']:
        run(sys.argv[1])
    else:
        print("Invalid command. Please use one of the following commands:")
        print("mng_imgs, conv_ker, conv_video", "cifar", "data_augmentation")
