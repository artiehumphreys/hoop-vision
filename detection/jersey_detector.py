import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans


np.set_printoptions(threshold=sys.maxsize)


class JerseyDetector:
    def __init__(self, player_imgs):
        self.player_imgs = player_imgs
        self.histogram = None
        self.peak_hues = []

    # https://cliveunger.github.io/pdfs/Basketball_Player_Tracking.pdf
    def find_peak_hues(self):
        peak_hues = []
        for player_img in self.player_imgs:
            _, cr_channel, cb_channel = cv2.split(player_img)
            non_zero_cr = cr_channel[cr_channel > 0]
            peak_cr = np.bincount(non_zero_cr).argmax()
            non_zero_cb = cr_channel[cb_channel > 0]
            peak_cb = np.bincount(non_zero_cb).argmax()
            print(peak_cr, peak_cb)
            peak_hues.append(peak_cr)
        self.peak_hues = peak_hues

    def create_histogram(self):
        if not self.peak_hues:
            self.find_peak_hues()

        hist_size = 32
        histogram, _ = np.histogram(self.peak_hues, bins=hist_size, range=(0, 256))
        self.histogram = histogram

    def get_teams(self):
        if self.histogram is None:
            self.create_histogram()

        kmeans = KMeans(n_clusters=2).fit(np.array(self.peak_hues).reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_.flatten()
        teams_hue_ranges = [(center - 10, center + 10) for center in cluster_centers]

        return teams_hue_ranges

    def assign_teams(self):
        teams_hue_ranges = self.get_teams()
        teams = ["team1", "team2"]
        player_teams = []
        print(teams_hue_ranges)
        for player_img in self.player_imgs:
            _, cr_channel, cb_channel = cv2.split(player_img)
            non_zero_hue = cr_channel[cr_channel > 0]
            dominant_hue = np.median(non_zero_hue)
            print(f"dominant: {dominant_hue}")
            team = "Referee"
            for j, (low, high) in enumerate(teams_hue_ranges):
                if low <= dominant_hue <= high:
                    team = teams[j]
                    break
            player_teams.append(team)
        return player_teams
