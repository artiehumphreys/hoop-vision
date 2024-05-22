import get_frames
import numpy as np
import cv2

def get_field_outline(path : str, precision : int = 5):
    image = cv2.imread(path)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

get_frames.extract_frames(5)
get_field_outline('data/frame5.jpg')
    