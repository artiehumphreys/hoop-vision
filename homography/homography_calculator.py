import cv2
import numpy as np
from detection import roboflow_detector


class HomographyCalculator:
    def __init__(self):
        self.roboflow_detector = roboflow_detector.RoboflowDetector()

    def calculate_homography_from_points(self, src_points, dst_points):
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        H, _ = cv2.findHomography(dst_points, src_points)
        # im_in = cv2.imread("data/frame55.jpg")
        # img_out = cv2.warpPerspective(im_in, H, (3000, 2500))
        # cv2.imshow("Original Image with Detected Players", img_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return H

    def calculate_vectors(self, src_points, left_corner):
        dst_points = [
            (src_point[0][0] - left_corner[0], src_point[0][1] - left_corner[1])
            for src_point in src_points
        ]
        dst_points = np.array(dst_points, dtype=np.float32)

        return dst_points

    def apply_homography(self, H, points):
        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1, 1, 2)

        transformed_points = cv2.perspectiveTransform(points, H)

        transformed_points = transformed_points.reshape(-1, 2)
        mid_y = 440 / 2
        for coord in transformed_points:
            coord[1] = 2 * mid_y - coord[1]
        return transformed_points

    def fetch_points_for_homography(self, img_str):
        project_id = "basketball_court_segmentation"
        model_id = 2
        lowest_paint = highest_paint = right_most_paint = left_most_paint = None
        predictions = self.roboflow_detector.make_request(img_str, project_id, model_id)
        for prediction in predictions["predictions"]:
            points = [(point["x"], point["y"]) for point in prediction["points"]]
            if prediction["class"] == "court":
                lowest_paint = max(points, key=lambda p: p[1])
                highest_paint = min(points, key=lambda p: p[1])
                right_most_paint = max(points, key=lambda p: p[0])
                left_most_paint = min(points, key=lambda p: p[0])
        return [
            lowest_paint,
            highest_paint,
            right_most_paint,
            left_most_paint,
        ]
