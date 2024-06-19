import cv2
from detection import detect_jersey, detect_with_roboflow
from pre_processing import image_loader
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
from PIL import Image

img_width = 0
img_height = 0


def is_in_court(img_str, player_positions):
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect_with_roboflow.make_request(img_str, project_id, model_id)
    points = []
    for prediction in predictions["predictions"]:
        points += [(point["x"], point["y"]) for point in prediction["points"]]

    num = len(points)
    j = num - 1
    in_court = []
    for player_position in player_positions:
        inside = False
        x = player_position[0]
        y = player_position[1]

        for i in range(num):
            xi, yi = points[i]
            xj, yj = points[j]

            intersect = ((yi + 5 > y) != (yj + 5 > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            )
            if intersect:
                inside = not inside
            j = i

        in_court.append(inside)

    return in_court


def detect_players_with_roboflow(image_path: str):
    try:
        image, img_str = image_loader.load_and_encode_image(image_path)
    except ValueError as e:
        print(e)
        return

    project_id = "basketball-w2xcw"
    model_id = 1

    predictions = detect_with_roboflow.make_request(img_str, project_id, model_id)
    _, _, player_positions = detect_with_roboflow.process_predictions(predictions)
    in_court = is_in_court(img_str, player_positions)
    draw_players(image, player_positions, in_court)
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
        (boxes[i, 2].item(), boxes[i, 3].item()) for i in range(len(boxes))
    ]

    _, img_str = image_loader.load_and_encode_image(image_path=image_path)
    in_court = is_in_court(img_str, player_positions)

    filtered_boxes = [
        i
        for i in range(len(in_court))
        if in_court[i]
        # Avoid getting heads from people in the crowd
        and (boxes[i, 3] - boxes[i, 1]) >= 1.2 * (boxes[i, 2] - boxes[i, 0])
    ]
    filtered_masks = masks[filtered_boxes]

    boxes = boxes[filtered_boxes]

    process_player_masks(image_path, boxes, filtered_masks)


def process_player_masks(image_path, boxes, filtered_masks):
    original_img = cv2.imread(image_path)
    original_img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    final_img = original_img.copy()

    for i in range(filtered_masks.shape[0]):
        mask = filtered_masks[i, 0] > 0.5
        play_mask = mask.numpy().astype("uint8") * 255

        player_img = cv2.bitwise_and(original_img_hsv, original_img_hsv, mask=play_mask)
        colored_mask = np.zeros_like(original_img)
        colored_mask[:, :, 0] = play_mask
        final_img = cv2.addWeighted(final_img, 1, colored_mask, 0.5, 0)
        cv2.putText(
            final_img,
            detect_jersey.get_teams_from_jersey(player_img),
            (int(boxes[i, 0].item()), int(boxes[i, 1].item())),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    display_image(image=final_img)


def display_image(image):
    cv2.imshow("Original Image with Detected Players", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
