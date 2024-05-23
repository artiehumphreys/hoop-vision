import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import cv2

load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')

def detect_players_with_roboflow(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Couldn't load image")
        return

    img = Image.open(image_path)

    buffered = BytesIO()

    img.save(buffered, quality=100, format="JPEG")

    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    project_id = 'basketball-w2xcw'
    model_id = 1
    confidence = 0.6
    iou_thresh = 0.5
    api_url = f'https://detect.roboflow.com/{project_id}/{model_id}?api_key={api_key}&confidence={confidence}&overlap={iou_thresh}'

    response = requests.post(api_url, data=img_str, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        predictions = response.json()
        print(predictions)
        for prediction in predictions['predictions']:
            if (prediction['class'] == 'person' or prediction['class'] == 'ball'):
                width = int(prediction['width'])
                height = int(prediction['height'])
                start_x = int(prediction['x'] + width/2)
                start_y = int(prediction['y'] + height/2)
                cv2.rectangle(image, (start_x, start_y), (int(start_x - width), int(start_y - height)), (0, 255, 0), 2)
                cv2.putText(image, prediction['class'] + " " + str(round(prediction['confidence'], 2)), (int(start_x - width), int(start_y - height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Original Image with Detected Players", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to get a response from RoboFlow API")

# Example usage:
detect_players_with_roboflow('data/frame140.jpg')