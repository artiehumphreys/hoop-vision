import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection import detect_players as detect
from homography import calculate_points as cp
from PIL import Image
from io import BytesIO
import cv2
import base64
from collections import defaultdict


def get_court_bound(image_path):
    _, ax = plt.subplots(figsize=(10, 7))
    image = cv2.imread(image_path)
    if image is None:
        print("Couldn't load image")
        return

    img = Image.open(image_path)

    buffered = BytesIO()

    img.save(buffered, quality=100, format="JPEG")

    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode("ascii")
    project_id = "basketball_court_segmentation"
    model_id = 2
    predictions = detect.make_request(img_str, project_id, model_id)
    bounds = defaultdict(list)

    for prediction in predictions["predictions"]:
        if prediction["class"] == "court":
            for point in prediction["points"]:
                bounds[point["x"]].append(point["y"])

    # Determine the minimum and maximum `y` values for each `x` coordinate
    bounds = {x: (min(y_values), max(y_values)) for x, y_values in bounds.items()}

    return bounds
