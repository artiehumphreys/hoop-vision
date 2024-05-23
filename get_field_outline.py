import get_frames
import numpy as np
import cv2

def segment_hardwood(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_wood = np.array([10, 40, 50])
    upper_wood = np.array([30, 255, 255])

    hardwood_mask = cv2.inRange(hsv_image, lower_wood, upper_wood)

    kernel = np.ones((5, 5), np.uint8)
    hardwood_mask = cv2.morphologyEx(hardwood_mask, cv2.MORPH_CLOSE, kernel)
    hardwood_mask = cv2.morphologyEx(hardwood_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(hardwood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def get_field_outline(path: str):
    image = cv2.imread(path)
    if image is None:
        print("Couldn't load image")
        return

    # Segment the hardwood
    contours = segment_hardwood(image)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Original Image with Detected Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    

get_frames.extract_frames(5)
get_field_outline('data/frame95.jpg')
    