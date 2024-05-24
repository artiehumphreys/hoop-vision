import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection import detect_players as detect

def plot_player_positions(img_str, player_positions):
    court_positions = []
    _, ax = plt.subplots(figsize=(10, 7))
    project_id = 'basketball_court_segmentation'
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    for prediction in predictions['predictions']:
        points = [(point['x'], point['y']) for point in prediction['points']]
        match prediction['class']:
            case 'two_point_area':
                color = 'blue'
            case 'three_second_area':
                color = 'red'
            case 'court':
                color = 'green'
            case _:
                color = 'gray'
        polygon = patches.Polygon(points, closed=True, fill=True, edgecolor=color, alpha=0.5)
        ax.add_patch(polygon)
    for (x_img, y_img) in player_positions:
        x_court = x_img
        y_court = y_img
        court_positions.append((x_court, y_court))

    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.set_aspect('equal', adjustable='box')
    for pos in court_positions:
        ax.plot(pos[0], pos[1], 'o', markersize=10, color='blue')
    plt.gca().invert_yaxis()
    # plt.legend()
    plt.title('Basketball Court Areas')
    plt.show()
