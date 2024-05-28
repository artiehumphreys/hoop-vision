import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import cv2
from detection import detect_shot

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


def detect_players_with_roboflow(image_path: str):
    rim_y = None
    ball_y = None
    player_positions = []
    image = cv2.imread(image_path)
    if image is None:
        print("Couldn't load image")
        return

    img = Image.open(image_path)

    buffered = BytesIO()

    img.save(buffered, quality=100, format="JPEG")

    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode("ascii")

    project_id = "basketball-w2xcw"
    model_id = 1

    predictions = make_request(img_str, project_id, model_id)

    # https://universe.roboflow.com/ownprojects/basketball-w2xcw/model/1
    shot = False

    for prediction in predictions["predictions"]:
        width = int(prediction["width"])
        height = int(prediction["height"])
        x = int(prediction["x"] + width / 2)
        y = int(prediction["y"] + height / 2)
        match prediction["class"]:
            case "ball":
                ball_y = prediction["y"]
            case "rim":
                rim_y = prediction["y"]
            case _:
                player_positions.append((x, y))
    #     cv2.rectangle(
    #         image,
    #         (int(x + width / 2), y),
    #         (int(x - width / 2), int(y - height)),
    #         (0, 255, 0),
    #         2,
    #     )
    #     cv2.putText(
    #         image,
    #         f"{prediction['class']} {str(round(prediction['confidence'], 2))}",
    #         (int(x - width), int(y - height) - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (0, 255, 0),
    #         2,
    #     )
    # if rim_y and ball_y:
    #     shot = detect_shot.detect_shot(rim_y, ball_y)
    # cv2.putText(
    #     image,
    #     f"Shot: {str(shot)}",
    #     (10, 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     2,
    # )
    # cv2.imshow("Original Image with Detected Players", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_str, player_positions
