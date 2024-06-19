import cv2
import numpy as np


class JerseyDetector:
    def __init__(self):
        self.lower_maroon = np.array([25, 125, 50])
        self.upper_maroon = np.array([200, 255, 255])
        self.lower_white = np.array([80, 50, 150])
        self.upper_white = np.array([200, 150, 255])

    def get_teams_from_jersey(self, player_img):
        player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HLS)

        nz_count_maroon = self.count_nonzero_pixels(
            player_hsv, self.lower_maroon, self.upper_maroon
        )
        nz_count_white = self.count_nonzero_pixels(
            player_hsv, self.lower_white, self.upper_white
        )

        if abs(nz_count_maroon - nz_count_white) > 1500:
            return "Referee"
        elif nz_count_maroon > nz_count_white:
            return "Cavs"
        return "Bulls"

    def count_nonzero_pixels(self, image_hsv, lower_bound, upper_bound):
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HLS2BGR)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(res)
