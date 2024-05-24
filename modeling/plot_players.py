import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection import detect_players
import draw_basketball_court as court

def plot_player_positions(img_str, player_positions):
    fig, ax = plt.subplots(figsize=(10, 7))
    project_id = 'basketball_court_segmentation'
    model_id = 2
    predictions = detect_players.make_request(img_str, project_id, model_id)
    for prediction in predictions['predictions']:
        points = [(point['x'], point['y']) for point in prediction['points']]
        if prediction['class'] == 'two_point_area':
            color = 'blue'
        elif prediction['class'] == 'three_second_area':
            color = 'red'
        elif prediction['class'] == 'court':
            color = 'green'
        else:
            color = 'gray'
        polygon = patches.Polygon(points, closed=True, fill=True, edgecolor=color, alpha=0.5)
        ax.add_patch(polygon)

    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title('Basketball Court Areas')
    plt.show()

def plot_players_on_court(player_positions, image_width, image_height):
    court_width = 500
    court_height = 470
    
    x_scale = court_width / image_width
    y_scale = court_height / image_height
    
    court_positions = []
    for (x_img, y_img) in player_positions:
        x_court = (x_img - image_width / 2) * x_scale
        y_court = (y_img - image_height / 2) * y_scale
        court_positions.append((x_court, y_court))
    
    ax = court.draw_basketball_court()
    
    for pos in court_positions:
        print(pos)
        ax.plot(pos[0], pos[1], 'o', markersize=10, color='blue')
    
    plt.show()