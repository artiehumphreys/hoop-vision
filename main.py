from detection import player_detector


def main():
    court_corners = [(190, 192.5), (310, 2.5), (310, 2.5), (190, 2.5)]
    img_path = "data/frame10.jpg"
    # pp.extract_frames(5)
    detector = player_detector.PlayerDetector()
    detector.detect_players_with_mask_rcnn(image_path=img_path)
    # court_bounds = pp.get_court_bound(img_path)
    # camera_view_corners = cp.fetch_points_for_homography(decoded_img)
    # db.plot_transformed_positions(player_positions, camera_view_corners, court_corners)


if __name__ == "__main__":
    main()
