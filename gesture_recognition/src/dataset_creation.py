"""
This script captures images from the webcam and saves them to a specified directory.
"""
import cv2
import os


def create_image(label):
    """

    hOLA Cris
    Capture images from the webcam and save them to a specified directory.
    
    params:
    label (str): The label for the gesture (e.g., "pulgar_arriba").
    """
    # Create a directory to save the images
    output_dir = f"data/{label}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i = 0

    while i < 200: 
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (64, 64))
        cv2.imwrite(f"{output_dir}/{label}_{i}.jpg", img)
        i += 1

        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
