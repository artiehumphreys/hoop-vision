import cv2 
import os 

cam = cv2.VideoCapture("./samples/Donovan_Mitchell_three.mp4") 
  
try: 
    if not os.path.exists('data'): 
        os.makedirs('data') 

except OSError: 
    print ('Failed to create directory') 
