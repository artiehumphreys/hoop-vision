import cv2
import numpy as np


# https://people.cs.nycu.edu.tw/~yushuen/data/BasketballVideo15.pdf
def extract_court_pixels_ycrcb(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    _, cr_channel, cb_channel = cv2.split(ycrcb_image)

    hist_size = 16
    cr_bins = np.linspace(0, 256, hist_size + 1)
    cb_bins = np.linspace(0, 256, hist_size + 1)

    histogram = np.zeros((hist_size, hist_size))

    height, width = cr_channel.shape

    dmax = height / 2.0
    weight_matrix = np.zeros((height, width))
    for i in range(height):
        di = min(i, height - i)
        weight_matrix[i, :] = di / dmax

    for i in range(height):
        for j in range(width):
            cr_value = cr_channel[i, j]
            cb_value = cb_channel[i, j]
            cr_bin = np.digitize(cr_value, cr_bins) - 1
            cb_bin = np.digitize(cb_value, cb_bins) - 1
            histogram[cr_bin, cb_bin] += weight_matrix[i, j]

    dominant_bin = np.unravel_index(np.argmax(histogram, axis=None), histogram.shape)

    cr_bin_center = (cr_bins[dominant_bin[0]] + cr_bins[dominant_bin[0] + 1]) / 2
    cb_bin_center = (cb_bins[dominant_bin[1]] + cb_bins[dominant_bin[1] + 1]) / 2

    threshold = 14  # adjustable
    court_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if (
                abs(cr_channel[i, j] - cr_bin_center) < threshold + 3
                and abs(cb_channel[i, j] - cb_bin_center) < threshold
            ):
                court_mask[i, j] = 255

    return court_mask


# https://people.cs.nycu.edu.tw/~yushuen/data/BasketballVideo15.pdf
def detect_court_boundary(image):
    court_mask = extract_court_pixels_ycrcb(image)

    contours, _ = cv2.findContours(
        court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = max(contours, key=cv2.contourArea)
    vertices = cv2.convexHull(largest_contour)

    lowest_court = max(vertices, key=lambda x: x[0][1])
    highest_court = min(vertices, key=lambda x: x[0][1])
    right_most_court = max(vertices, key=lambda x: x[0][0])
    left_most_court = min(vertices, key=lambda x: (x[0][0], x[0][1]))

    cv2.drawContours(image, [vertices], -1, (0, 255, 0), 2)
    cv2.imshow("hull", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return vertices, [
        lowest_court,
        highest_court,
        right_most_court,
        left_most_court,
    ]
