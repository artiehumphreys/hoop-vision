import base64
import requests
from PIL import Image
from io import BytesIO
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
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
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

    max_box_area = 4750
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    individual_indices = box_areas > max_box_area
    masks = masks[individual_indices]
    boxes = boxes[individual_indices]

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


# rim_y = None
# ball_y = None
# player_positions = []
# image = cv2.imread(image_path)
# if image is None:
#     print("Couldn't load image")
#     return

# img = Image.open(image_path)

# buffered = BytesIO()

# img.save(buffered, quality=100, format="JPEG")

# img_bytes = base64.b64encode(buffered.getvalue())
# img_str = img_bytes.decode("ascii")

# project_id = "basketball-w2xcw"
# model_id = 1

# predictions = make_request(img_str, project_id, model_id)

# # https://universe.roboflow.com/ownprojects/basketball-w2xcw/model/1
# shot = False

# for prediction in predictions["predictions"]:
#     width = int(prediction["width"])
#     height = int(prediction["height"])
#     x = int(prediction["x"] + width / 2)
#     y = int(prediction["y"] + height / 2)
#     match prediction["class"]:
#         case "ball":
#             ball_y = prediction["y"]
#         case "rim":
#             rim_y = prediction["y"]
#         case _:
#             player_positions.append((x, y))
# #     cv2.rectangle(
# #         image,
# #         (int(x + width / 2), y),
# #         (int(x - width / 2), int(y - height)),
# #         (0, 255, 0),
# #         2,
# #     )
# #     cv2.putText(
# #         image,
# #         f"{prediction['class']} {str(round(prediction['confidence'], 2))}",
# #         (int(x - width), int(y - height) - 10),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.5,
# #         (0, 255, 0),
# #         2,
# #     )
# # if rim_y and ball_y:
# #     shot = detect_shot.detect_shot(rim_y, ball_y)
# # cv2.putText(
# #     image,
# #     f"Shot: {str(shot)}",
# #     (10, 10),
# #     cv2.FONT_HERSHEY_SIMPLEX,
# #     0.5,
# #     (0, 255, 0),
# #     2,
# # )
# # cv2.imshow("Original Image with Detected Players", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# return img_str, player_positions
