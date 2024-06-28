import datetime
from functools import cache

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from shapely.geometry import Point, Polygon
from torchvision import transforms as T

from detection.jersey_detector import JerseyDetector


class PlayerDetector:
    def __init__(self, court_bounds):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.court_bounds = court_bounds
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        ).to(self.device)
        self.model.eval()

    def is_in_court(self, player_positions):
        court_polygon = Polygon(
            [
                (
                    (point[0], point[1] + 25)
                    if point[1] < 360
                    else (point[0], point[1] - 10)
                )
                for point in self.court_bounds
            ]
        )

        in_court = []
        for player_position in player_positions:
            player_point = Point(player_position)
            in_court.append(court_polygon.contains(player_point))

        return in_court

    @cache
    def non_maximum_suppression(self, masks, scores):
        def calculate_mask_iou(mask1, mask2):
            mask1 = mask1 > 0.5
            mask2 = mask2 > 0.5
            intersection = (mask1 & mask2).float().sum((1, 2))
            union = (mask1 | mask2).float().sum((1, 2))
            iou = intersection / union
            return iou

        iou_thresh = 0.25
        combined = list(zip(masks, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        masks, scores = zip(*combined)

        masks = torch.stack(masks)
        scores = torch.tensor(scores)

        valid_indices = []
        for i in range(len(masks)):
            keep = True
            for j in range(len(valid_indices)):
                if calculate_mask_iou(masks[i], masks[valid_indices[j]]) > iou_thresh:
                    keep = False
                    break
            if keep:
                valid_indices.append(i)

        return valid_indices

    def detect_players_with_mask_rcnn(self, image_path: str):
        start = datetime.datetime.now()
        print(f"beginning detection: {start}")
        img = Image.open(image_path)
        transform = T.ToTensor()
        transformed_img = transform(img)
        with torch.no_grad():
            pred = self.model([transformed_img])
        end = datetime.datetime.now()
        print(f"ended detection: {end - start}")
        threshold = 0.7
        scores = pred[0]["scores"]
        high_conf_indices = scores > threshold
        labels = pred[0]["labels"][high_conf_indices]
        player_indices = labels == 1
        boxes = pred[0]["boxes"][high_conf_indices][player_indices]
        masks = pred[0]["masks"][high_conf_indices][player_indices]
        scores = scores[high_conf_indices][player_indices]
        non_overlapped_indices = self.non_maximum_suppression(masks, scores)
        boxes = boxes[non_overlapped_indices]
        masks = masks[non_overlapped_indices]

        player_positions = [
            (boxes[i, 2].item(), boxes[i, 3].item()) for i in range(len(boxes))
        ]

        in_court = self.is_in_court(player_positions)

        filtered_boxes = [i for i in range(len(in_court)) if in_court[i]]
        filtered_masks = masks[filtered_boxes]
        boxes = boxes[filtered_boxes]

        return self.process_player_masks(image_path, boxes, filtered_masks)

    def process_player_masks(self, image_path, boxes, filtered_masks):
        original_img = cv2.imread(image_path)
        original_img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        original_img_ycrcb = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)
        final_img = original_img.copy()
        player_positions = []
        player_imgs = []
        for i in range(filtered_masks.shape[0]):
            mask = filtered_masks[i, 0] > 0.5
            play_mask = mask.numpy().astype("uint8") * 255
            player_img = cv2.bitwise_and(
                original_img_ycrcb, original_img_hsv, mask=play_mask
            )
            player_imgs.append(player_img)
            colored_mask = np.zeros_like(original_img)
            colored_mask[:, :, 0] = play_mask
            final_img = cv2.addWeighted(final_img, 1, colored_mask, 0.5, 0)
            player_positions.append([(boxes[i, 2].item(), boxes[i, 3].item()), ""])
            cv2.putText(
                final_img,
                "o",
                (int(boxes[i, 2].item()), int(boxes[i, 3].item())),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        self.player_imgs = player_imgs
        jersey_detector = JerseyDetector(player_imgs)
        teams = jersey_detector.assign_teams()
        for i in range(len(teams)):
            player_positions[i][1] = teams[i]
            cv2.putText(
                final_img,
                str(player_positions[i][1]),
                (int(boxes[i, 0].item()), int(boxes[i, 1].item())),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        self.display_image(final_img)
        return player_positions

    def display_image(self, image):
        cv2.imshow("Original Image with Detected Players", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_players(self, image, player_positions, in_court):
        epsilon = 10
        for i in range(len(player_positions)):
            if not in_court[i]:
                continue
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
