import get_frames
import numpy as np
import cv2

def get_field_outline(path : str, precision : int = 5):
    image = cv2.imread(path)
    if image is None:
        print("Couldn't load image")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_wood = np.array([10, 40, 50])
    upper_wood = np.array([30, 255, 255])

    hardwood_mask = cv2.inRange(hsv_image, lower_wood, upper_wood)

    cv2.imshow("Mask", hardwood_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

get_frames.extract_frames(5)
get_field_outline('data/frame95.jpg')
    