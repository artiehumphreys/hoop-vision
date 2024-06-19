from detection import player_detector
from modeling import court_drawer
from pre_processing import image_loader
from homography import homography_calculator
from icecream import ic


def main():
    court_corners = [(190, 192.5), (310, 2.5), (310, 2.5), (190, 2.5)]
    img_path = "data/frame10.jpg"
    # pp.extract_frames(5)
    img = image_loader.ImageLoader(img_path)
    detector = player_detector.PlayerDetector(img)
    drawer = court_drawer.CourtDrawer()
    calc = homography_calculator.HomographyCalculator()
    _, encoded_img = img.load_and_encode_image()
    player_positions = detector.detect_players_with_mask_rcnn(image_path=img_path)
    camera_view_corners = calc.fetch_points_for_homography(encoded_img)
    ic(camera_view_corners, court_corners, player_positions)
    drawer.plot_transformed_positions(
        player_positions, camera_view_corners, court_corners
    )


if __name__ == "__main__":
    main()
