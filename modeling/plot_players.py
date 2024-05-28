import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection import detect_players as detect


def fetch_points_for_homography(img_str):
    lowest_point = highest_point = right_most_point = left_most_point = None
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    for prediction in predictions["predictions"]:
        points = [(point["x"], point["y"]) for point in prediction["points"]]
        if prediction["class"] == "three_second_area":
            lowest_point = max(points, key=lambda p: p[1])
            highest_point = min(points, key=lambda p: p[1])
            right_most_point = max(points, key=lambda p: p[0])
            left_most_point = min(points, key=lambda p: p[0])
    return lowest_point, highest_point, right_most_point, left_most_point


def plot_player_positions(img_str, player_positions):
    _, ax = plt.subplots(figsize=(10, 7))
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    for prediction in predictions["predictions"]:
        points = [(point["x"], point["y"]) for point in prediction["points"]]
        match prediction["class"]:
            case "two_point_area":
                color = "blue"
            case "three_second_area":
                color = "red"
            case "court":
                color = "green"
            case _:
                color = "gray"
        polygon = patches.Polygon(
            points, closed=True, fill=True, edgecolor=color, alpha=0.5
        )
        ax.add_patch(polygon)

    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.set_aspect("equal", adjustable="box")
    for pos in player_positions:
        ax.plot(pos[0], pos[1], "o", markersize=10, color="blue")
    plt.gca().invert_yaxis()
    # plt.legend()
    plt.title("Basketball Court Areas")
    plt.show()
