import sys

import cv2
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


class JerseyDetector:
    def __init__(self, player_imgs):
        self.lower_maroon = np.array([25, 125, 50])
        self.upper_maroon = np.array([200, 255, 255])
        self.lower_white = np.array([80, 50, 150])
        self.upper_white = np.array([200, 150, 255])
        self.player_imgs = player_imgs
        self.histogram = None

    def create_histogram(self):
        hist_size = 32
        histogram = np.zeros(hist_size)
        h_bins = np.linspace(0, 256, hist_size + 1)

        for player_img in self.player_imgs:
            hue_channel = player_img[:, :, 0]
            non_zero_hue = hue_channel[hue_channel > 0]
            hist, _ = np.histogram(non_zero_hue, bins=h_bins)
            histogram += hist
        self.histogram = histogram

    def get_teams(self):
        top_bins = np.argsort(self.histogram)[::-1][:3]
        hsv_ranges = [(bin * 8, (bin + 1) * 8) for bin in top_bins]
        teams = ["team1", "team2", "team3"]
        player_teams = []
        print(hsv_ranges)
        for player_img in self.player_imgs:
            team = "TBD"
            hue_channel = player_img[:, :, 1]
            non_zero_hue = hue_channel[hue_channel > 0]
            dominant_hue = np.median(non_zero_hue)
            print(dominant_hue)
            for j, (low, high) in enumerate(hsv_ranges):
                if low <= dominant_hue <= high:
                    team = teams[j]
                    break

            player_teams.append(dominant_hue)
        return player_teams

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
