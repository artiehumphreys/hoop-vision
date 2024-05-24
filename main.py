from modeling import plot_players as players
from pre_processing import get_frames as pp
from detection import detect_players as dp

def main():
    img_path = "data/frame80.jpg"
    # pp.extract_frames(5)
    decoded_img, player_positions = dp.detect_players_with_roboflow(image_path=img_path)
    players.plot_players_on_court(player_positions = player_positions, image_width = 1280, image_height = 720)

main()
