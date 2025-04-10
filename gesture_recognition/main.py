"""
    # This script is the main entry point for the gesture recognition application.
"""
from src.dataset_creation import create_image

def main():
    """
    Main function to run the gesture recognition application.
    """
    label = "pulgar_abajo" 
    create_image(label)


if __name__ == "__main__":
    main()