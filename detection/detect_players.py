import base64
import requests
from PIL import Image
from dotenv import load_dotenv
import os
import cv2
from detection import detect_shot
from modeling import plot_players as pp
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
from io import BytesIO

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


def detect_players_with_mask_crnn(image_path: str):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    model.eval()
    img = Image.open(image_path)
    transform = T.ToTensor()
    transormed_img = transform(img)
    with torch.no_grad():
        pred = model([transormed_img])

    threshold = 0.5
    scores = pred[0]["scores"]
    high_conf_indices = scores > threshold
    labels = pred[0]["labels"][high_conf_indices]
    player_indices = labels == 1
    boxes = pred[0]["boxes"][high_conf_indices][player_indices]
    masks = pred[0]["masks"][high_conf_indices][player_indices]

    player_positions = [
        (boxes[i, 2].item(), boxes[i, 3].item())
        for i in range(len(boxes))
        # if boxes[i, 3] - boxes[i, 1] > 60
    ]

    image, img_str = load_and_encode_image(image_path=image_path)
    in_court = pp.is_in_court(img_str, player_positions)

    filtered_boxes = [i for i in range(len(in_court)) if in_court[i]]
    filtered_masks = masks[filtered_boxes]

    masks = masks[filtered_boxes]
    boxes = boxes[filtered_boxes]

    original_img = cv2.imread(image_path)
    final_img = original_img.copy()

    for i in range(filtered_masks.shape[0]):
        mask = filtered_masks[i, 0] > 0.5
        play_mask = mask.numpy().astype("uint8") * 255

        colored_mask = np.zeros_like(original_img)
        colored_mask[:, :, 0] = play_mask
        final_img = cv2.addWeighted(final_img, 1, colored_mask, 0.5, 0)

    display_image(image=final_img)


def display_image(image):
    cv2.imshow("Original Image with Detected Players", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
