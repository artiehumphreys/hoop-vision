from detection.player_detector import PlayerDetector
from modeling.court_drawer import CourtDrawer
from pre_processing.image_loader import ImageLoader
from homography.homography_calculator import HomographyCalculator
import os
import re


def extract_number(filename):
    match = re.search(r"frame(\d+)", filename)
    return int(match.group(1)) if match else -1


def main():
    directory = "data"
    files = os.listdir(directory)
    sorted_files = sorted(files, key=extract_number)
    for filename in sorted_files:
        if not filename.endswith("jpg"):
            continue
        img_path = os.path.join("data", filename)
        print(img_path)
        img = ImageLoader(img_path)
        detector = PlayerDetector(img)
        drawer = CourtDrawer()
        calc = HomographyCalculator()
        court_corners = drawer.right_bounds
        _, encoded_img = img.load_and_encode_image()
        player_positions = detector.detect_players_with_mask_rcnn(image_path=img_path)
        camera_view_corners = calc.fetch_points_for_homography(encoded_img)
        drawer.plot_transformed_positions(
            player_positions, camera_view_corners, court_corners
        )


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
    drawer.plot_vectors(vectors)


if __name__ == "__main__":
    main()
