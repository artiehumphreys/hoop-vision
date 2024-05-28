from modeling import plot_players as pp
from modeling import draw_basketball_court as db
from detection import detect_players as dp
from homography import calculate_points as cp


def main():
    court_corners = [(-60, 142.5), (60, -47.5), (-60, -47.5), (60, 142.5)]
    img_path = "data/frame30.jpg"
    # pp.extract_frames(5)
    decoded_img, player_positions = dp.detect_players_with_roboflow(image_path=img_path)
    pp.plot_player_positions(decoded_img, player_positions)
    camera_view_corners = cp.fetch_points_for_homography(decoded_img)
    print(camera_view_corners, player_positions)
    db.plot_transformed_positions(player_positions, camera_view_corners, court_corners)


if __name__ == "__main__":
    main()
