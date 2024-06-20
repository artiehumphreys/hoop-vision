from detection.player_detector import PlayerDetector
from modeling.court_drawer import CourtDrawer
from pre_processing.image_loader import ImageLoader
from homography.homography_calculator import HomographyCalculator
from icecream import ic
import numpy as np


def main():
    img_path = "data/frame55.jpg"
    img = ImageLoader(img_path)
    detector = PlayerDetector(img)
    drawer = CourtDrawer()
    calc = HomographyCalculator()
    court_corners = drawer.right_bounds
    _, encoded_img = img.load_and_encode_image()
    player_positions = detector.detect_players_with_mask_rcnn(image_path=img_path)
    camera_view_corners = calc.fetch_points_for_homography(encoded_img)
    ic(camera_view_corners, court_corners, player_positions)
    drawer.plot_transformed_positions(
        player_positions, camera_view_corners, court_corners, True
    )
    ic(calc.calculate_vectors(player_positions))


def vectors():
    img_path = "data/frame55.jpg"
    img = ImageLoader(img_path)
    detector = PlayerDetector(img)
    drawer = CourtDrawer()
    calc = HomographyCalculator()
    _, encoded_img = img.load_and_encode_image()
    player_positions = detector.detect_players_with_mask_rcnn(image_path=img_path)
    camera_view_corners = calc.fetch_points_for_homography(encoded_img)
    left_corner = camera_view_corners[0]
    vectors = calc.calculate_vectors(player_positions, left_corner)
    ic(vectors)
    drawer.plot_vectors(vectors)


if __name__ == "__main__":
    vectors()
