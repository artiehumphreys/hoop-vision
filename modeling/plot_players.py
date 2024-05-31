import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection import detect_players as detect
from homography import calculate_points as cp


def get_court_bound(img_str, player_positions):
    _, ax = plt.subplots(figsize=(10, 7))
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    bounds = {}
    for prediction in predictions["predictions"]:
        if prediction["class"] == "court":
            bounds = {point["x"]: point["y"] for point in prediction["points"]}

    print(bounds)
