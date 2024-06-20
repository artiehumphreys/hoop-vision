import cv2
import numpy as np
from detection import roboflow_detector
from icecream import ic


class HomographyCalculator:
    def __init__(self):
        self.roboflow_detector = roboflow_detector.RoboflowDetector()

    def calculate_homography_from_points(self, src_points, dst_points):
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        return H

    def apply_homography(self, H, points):
        points = np.array(points, dtype=np.float32)
        points = np.array([points])
        transformed_points = cv2.perspectiveTransform(points, H)
        for coord in transformed_points[0]:
            coord[0] += 300
            coord[1] -= 100
        ic(transformed_points)
        return transformed_points[0]

    def fetch_points_for_homography(self, img_str):
        project_id = "basketball_court_segmentation"
        model_id = 2
        lowest_paint = highest_paint = right_most_paint = left_most_paint = None
        predictions = self.roboflow_detector.make_request(img_str, project_id, model_id)
        for prediction in predictions["predictions"]:
            points = [(point["x"], point["y"]) for point in prediction["points"]]
            if prediction["class"] == "three_second_area":
                lowest_paint = max(points, key=lambda p: p[1])
                highest_paint = min(points, key=lambda p: p[1])
                right_most_paint = max(points, key=lambda p: p[0])
                left_most_paint = min(points, key=lambda p: p[0])
            elif prediction["class"] == "two_point_area":
                right_most_three = right_most_paint = max(points, key=lambda p: p[0])
        return [
            lowest_paint,
            highest_paint,
            right_most_paint,
            left_most_paint,
            # right_most_three,
        ]
