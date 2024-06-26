import cv2
import numpy as np


class JerseyDetector:
    def __init__(self, player_imgs):
        self.lower_maroon = np.array([25, 125, 50])
        self.upper_maroon = np.array([200, 255, 255])
        self.lower_white = np.array([80, 50, 150])
        self.upper_white = np.array([200, 150, 255])
        self.player_imgs = player_imgs
        self.histogram = None

    def create_histogram(self):
        hist_size = 16
        histogram = np.zeros(hist_size)
        h_bins = np.linspace(0, 256, hist_size + 1)

        for player_img in self.player_imgs:
            player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            hue_channel = player_hsv[:, :, 0]

            hist, _ = np.histogram(hue_channel, bins=h_bins)
            histogram += hist

        self.histogram = histogram

    def get_teams(self):
        top_bins = np.argsort(self.histogram)[::-1][:3]
        for player_img in self.player_imgs:
            

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
