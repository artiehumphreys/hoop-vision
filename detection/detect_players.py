import base64
import requests
from PIL import Image
from dotenv import load_dotenv
import os
import cv2
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np


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

    box_height = boxes[:, 3] - boxes[:, 1]
    box_width = boxes[:, 2] - boxes[:, 0]
    min_height = 90
    min_width = 60
    # TODO: Use roboflow to detect court boundaries
    filtered_boxes = (box_width >= min_width) & (box_height >= min_height)
    masks = masks[filtered_boxes]
    boxes = boxes[filtered_boxes]

    original_img = cv2.imread(image_path)
    final_img = original_img.copy()

    for i in range(masks.shape[0]):
        mask = masks[i, 0] > 0.5
        play_mask = mask.numpy().astype("uint8") * 255

        colored_mask = np.zeros_like(original_img)
        colored_mask[:, :, 0] = play_mask
        final_img = cv2.addWeighted(final_img, 1, colored_mask, 0.5, 0)

    cv2.imshow("Players Detected", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
