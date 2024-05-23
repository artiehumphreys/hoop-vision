import os
import numpy as np
import cv2
from inference_sdk import InferenceHTTPClient
import supervision as sv
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')
print(api_key)

def segment_hardwood(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_wood = np.array([10, 40, 50])
    upper_wood = np.array([30, 255, 255])

    hardwood_mask = cv2.inRange(hsv_image, lower_wood, upper_wood)

    kernel = np.ones((5, 5), np.uint8)
    hardwood_mask = cv2.morphologyEx(hardwood_mask, cv2.MORPH_CLOSE, kernel)
    hardwood_mask = cv2.morphologyEx(hardwood_mask, cv2.MORPH_OPEN, kernel)

    return hardwood_mask

def get_field_outline(path: str):
    image = cv2.imread(path)
    if image is None:
        print("Couldn't load image")
        return
    
    return segment_hardwood(image)