from modeling import plot_players as pp
from modeling import draw_basketball_court as db
from detection import detect_players as dp
from homography import calculate_points as cp
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def main():
    court_corners = [(-60, 140), (60, -50), (-60, -50), (60, 140)]
    img_path = "data/frame55.jpg"
    # pp.extract_frames(5)
    decoded_img, player_positions = dp.detect_players_with_roboflow(image_path=img_path)
    camera_view_corners = cp.fetch_points_for_homography(decoded_img)
    db.plot_transformed_positions(player_positions, camera_view_corners, court_corners)


if __name__ == "__main__":
    main()
