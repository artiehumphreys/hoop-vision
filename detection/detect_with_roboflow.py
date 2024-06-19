import requests
import cv2
from PIL import Image
import base64
from io import BytesIO


def make_request(img_str, project_id: str, model_id: int = 1):
    confidence = 0.55
    iou_thresh = 0.5
    # https://inference.roboflow.com/quickstart/run_model_on_image/#run-inference-on-a-v1-route
    api_url = f"https://detect.roboflow.com/{project_id}/{model_id}?api_key={api_key}&confidence={confidence}&overlap={iou_thresh}"

    response = requests.post(
        api_url, data=img_str, headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        print("Failed to get a response from RoboFlow API")
        return
    return response.json()


def load_and_encode_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Couldn't load image")

    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, quality=100, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode("ascii")

    return image, img_str


def process_predictions(predictions):
    ball_y = rim_y = None
    player_positions = []
    epsilon = 10

    for prediction in predictions["predictions"]:
        width = int(prediction["width"])
        height = int(prediction["height"])
        x = int(prediction["x"] + width / 2)
        y = int(prediction["y"] + height / 2 - epsilon)
        match prediction["class"]:
            case "ball":
                ball_y = prediction["y"]
            case "rim":
                rim_y = prediction["y"]
            case "person":
                player_positions.append((x, y, width, height))

    return ball_y, rim_y, player_positions
