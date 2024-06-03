import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import cv2
from detection import detect_shot
from modeling import plot_players as pp

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

img_width = 0
img_height = 0


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


def draw_players(image, player_positions, in_court):
    epsilon = 10
    for i in range(len(player_positions)):
        if in_court[i]:
            x, y, width, height = player_positions[i]
            cv2.rectangle(
                image,
                (int(x), int(y + epsilon)),
                (int(x - width), int(y - height + epsilon)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                "Player",
                (int(x - 10), int(y - 10) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )


def detect_shot_with_ball(image, rim_y, ball_y):
    shot = False
    if rim_y and ball_y:
        shot = detect_shot.detect_shot(rim_y, ball_y)
    cv2.putText(
        image,
        f"Shot: {str(shot)}",
        (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return shot


def detect_players_with_roboflow(image_path: str):
    try:
        image, img_str = load_and_encode_image(image_path)
    except ValueError as e:
        print(e)
        return

    project_id = "basketball-w2xcw"
    model_id = 1

    predictions = make_request(img_str, project_id, model_id)
    ball_y, rim_y, player_positions = process_predictions(predictions)
    in_court = pp.is_in_court(img_str, player_positions)
    draw_players(image, player_positions, in_court)
    shot = detect_shot_with_ball(image, rim_y, ball_y)
    display_image(image)

    return img_str, player_positions


def display_image(image):
    cv2.imshow("Original Image with Detected Players", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
