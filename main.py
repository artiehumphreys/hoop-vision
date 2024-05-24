from modeling import plot_players as pp
from detection import detect_players as dp
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def main():
    img_path = "data/frame50.jpg"
    # pp.extract_frames(5)
    decoded_img, player_positions = dp.detect_players_with_roboflow(image_path=img_path)
    pp.plot_player_positions(decoded_img, player_positions)


if __name__ == "__main__":
    main()
