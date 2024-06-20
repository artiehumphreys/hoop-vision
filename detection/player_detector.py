import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms as T
from detection import jersey_detector, roboflow_detector
from pre_processing import image_loader


class PlayerDetector:
    def __init__(self, image_loader):
        self.roboflow_detector = roboflow_detector.RoboflowDetector()
        self.jersey_detector = jersey_detector.JerseyDetector()
        self.image_loader = image_loader

    def is_in_court(self, img_str, player_positions):
        project_id = "basketball_court_segmentation"
        model_id = 2
        predictions = self.roboflow_detector.make_request(img_str, project_id, model_id)
        points = []
        for prediction in predictions["predictions"]:
            points += [(point["x"], point["y"]) for point in prediction["points"]]

        num = len(points)
        j = num - 1
        in_court = []
        for player_position in player_positions:
            inside = False
            x, y = player_position[0], player_position[1]

            for i in range(num):
                xi, yi = points[i]
                xj, yj = points[j]

                intersect = ((yi + 10 > y) != (yj + 10 > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi
                )
                if intersect:
                    inside = not inside
                j = i

            in_court.append(inside)

        return in_court

    def detect_players_with_roboflow(self):
        try:
            image, img_str = self.image_loader.load_and_encode_image()
        except ValueError as e:
            print(e)
            return

        project_id = "basketball-w2xcw"
        model_id = 1

        predictions = self.roboflow_detector.make_request(img_str, project_id, model_id)
        _, _, player_positions = self.roboflow_detector.process_predictions(predictions)
        in_court = self.is_in_court(img_str, player_positions)
        self.draw_players(image, player_positions, in_court)
        self.display_image(image)

        return img_str, player_positions

    def detect_players_with_mask_rcnn(self, image_path: str):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        model.eval()
        img = Image.open(image_path)
        transform = T.ToTensor()
        transformed_img = transform(img)
        with torch.no_grad():
            pred = model([transformed_img])

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

        _, img_str = self.image_loader.load_and_encode_image()
        in_court = self.is_in_court(img_str, player_positions)

        filtered_boxes = [
            i
            for i in range(len(in_court))
            if in_court[i]
            and (boxes[i, 3] - boxes[i, 1]) >= 1.2 * (boxes[i, 2] - boxes[i, 0])
        ]
        filtered_masks = masks[filtered_boxes]
        boxes = boxes[filtered_boxes]

        return self.process_player_masks(image_path, boxes, filtered_masks)

    def process_player_masks(self, image_path, boxes, filtered_masks):
        original_img = cv2.imread(image_path)
        original_img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        final_img = original_img.copy()
        player_positions = []

        for i in range(filtered_masks.shape[0]):
            mask = filtered_masks[i, 0] > 0.5
            play_mask = mask.numpy().astype("uint8") * 255

            player_img = cv2.bitwise_and(
                original_img_hsv, original_img_hsv, mask=play_mask
            )
            colored_mask = np.zeros_like(original_img)
            colored_mask[:, :, 0] = play_mask
            final_img = cv2.addWeighted(final_img, 1, colored_mask, 0.5, 0)
            team = self.jersey_detector.get_teams_from_jersey(player_img)
            player_positions.append(((boxes[i, 2].item(), boxes[i, 3].item()), team))
            cv2.putText(
                final_img,
                team,
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
