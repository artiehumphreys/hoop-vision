import matplotlib.pyplot as plt
from detection import detect_players as detect
import cv2
import numpy as np


def get_teams_from_jersey(player_img):
    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HLS)
    lower_maroon = np.array([25, 125, 50])
    upper_maroon = np.array([200, 255, 255])

    lower_white = np.array([80, 50, 150])
    upper_white = np.array([200, 150, 255])
    # Check for maroon jersey
    mask_maroon = cv2.inRange(player_hsv, lower_maroon, upper_maroon)
    res_maroon = cv2.bitwise_and(player_img, player_img, mask=mask_maroon)
    res_maroon = cv2.cvtColor(res_maroon, cv2.COLOR_HLS2BGR)
    res_maroon = cv2.cvtColor(res_maroon, cv2.COLOR_BGR2GRAY)
    nz_count_maroon = cv2.countNonZero(res_maroon)

    mask_white = cv2.inRange(player_hsv, lower_white, upper_white)
    res_white = cv2.bitwise_and(player_img, player_img, mask=mask_white)
    res_white = cv2.cvtColor(res_white, cv2.COLOR_HLS2BGR)
    res_white = cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY)
    nz_count_white = cv2.countNonZero(res_white)

    # return str(nz_count_maroon) + ", " + str(nz_count_white)

    if abs(nz_count_maroon - nz_count_white) > 1500:
        return "Referee"
    elif nz_count_maroon > nz_count_white:
        return "Cavs"
    return "Bulls"
