import math
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans

np.set_printoptions(threshold=sys.maxsize)


class JerseyDetector:
    def __init__(self, player_imgs):
        self.player_imgs = player_imgs
        self.histogram = None
        self.peak_cr_values = []

    def set_player_imgs(self, player_imgs):
        self.player_imgs = player_imgs

    # https://cliveunger.github.io/pdfs/Basketball_Player_Tracking.pdf
    def find_peak_cr_values(self):
        peak_crs = []
        for player_img in self.player_imgs:
            _, cr_channel, _ = cv2.split(player_img)
            non_zero_cr = cr_channel[cr_channel > 0]
            peak_cr = np.bincount(non_zero_cr).argmax()
            peak_crs.append(peak_cr)
        self.peak_cr_values = peak_crs

    def create_histogram(self):
        if not self.peak_cr_values:
            self.find_peak_cr_values()

        hist_size = 32
        histogram, _ = np.histogram(self.peak_cr_values, bins=hist_size, range=(0, 256))
        self.histogram = histogram

    def get_teams(self):
        if self.histogram is None:
            self.create_histogram()

        threshold = 12
        kmeans = KMeans(n_clusters=2).fit(np.array(self.peak_cr_values).reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_.flatten()
        teams_hue_ranges = [
            (center - threshold, center + threshold) for center in cluster_centers
        ]

        return teams_hue_ranges

    def assign_teams(self):
        teams_hue_ranges = self.get_teams()
        teams_hue_ranges.sort(key=lambda x: x[0])
        teams = ["team1", "team2"]
        player_teams = []
        print(teams_hue_ranges)
        for player_img in self.player_imgs:
            _, cr_channel, _ = cv2.split(player_img)
            non_zero_cr = cr_channel[cr_channel > 0]
            median_cr = np.median(non_zero_cr)
            team = "Referee"
            for j, (low, high) in enumerate(teams_hue_ranges):
                if math.floor(low) <= median_cr < math.floor(high):
                    team = teams[j]
                    break
            player_teams.append(team)
        return player_teams
