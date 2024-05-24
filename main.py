from modeling import plot_players as players
from pre_processing import get_frames as pp
from detection import detect_players as dp
from datetime import datetime

def main():
    img_path = "data/frame50.jpg"
    # pp.extract_frames(5)
    decoded_img, player_positions = dp.detect_players_with_roboflow(image_path=img_path)
    players.plot_player_positions(img_str = decoded_img, player_positions = player_positions)
main()
