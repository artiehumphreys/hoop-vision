from detection import detect_players as detect
import cv2
import numpy as np


def calculate_homography(src_points, dst_points):
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    return H


def apply_homography(H, points):
    points = np.array(points, dtype=np.float32)
    points = np.array([points])
    transformed_points = cv2.perspectiveTransform(points, H)
    return transformed_points[0]


def fetch_points_for_homography(img_str):
    lowest_point = highest_point = right_most_point = left_most_point = None
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    for prediction in predictions["predictions"]:
        points = [(point["x"], point["y"]) for point in prediction["points"]]
        if prediction["class"] == "three_second_area":
            lowest_point = max(points, key=lambda p: p[1])
            highest_point = min(points, key=lambda p: p[1])
            right_most_point = max(points, key=lambda p: p[0])
            left_most_point = min(points, key=lambda p: p[0])
    return [lowest_point, highest_point, right_most_point, left_most_point]
