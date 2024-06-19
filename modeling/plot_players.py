import matplotlib.pyplot as plt
from detection import detect_players as detect
from homography import calculate_points as cp


def is_in_court(img_str, player_positions):
    _, ax = plt.subplots(figsize=(10, 7))
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    points = []
    for prediction in predictions["predictions"]:
        points += [(point["x"], point["y"]) for point in prediction["points"]]

    num = len(points)
    j = num - 1
    in_court = []
    for player_position in player_positions:
        inside = False
        x = player_position[0]
        y = player_position[1]

        for i in range(num):
            xi, yi = points[i]
            xj, yj = points[j]

            intersect = ((yi + 5 > y) != (yj + 5 > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            )
            if intersect:
                inside = not inside
            j = i

        in_court.append(inside)

    return in_court
