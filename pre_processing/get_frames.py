import cv2
import os


def create_dir():
    try:
        if not os.path.exists("data"):
            os.makedirs("data")

    except OSError:
        print("Failed to create directory")


def extract_frames(frames_per_second: int = 10):

    create_dir()

    cam = cv2.VideoCapture("samples/Donovan_Mitchell_three.mp4")

    if not cam.isOpened():
        print("Error: Could not open video.")
        return

    frame_rate = cam.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate // frames_per_second)

    frame_count = 0
    while True:
        ret, frame = cam.read()
        if ret and frame_count % frame_interval == 0:
            name = "./data/frame" + str(frame_count) + ".jpg"
            print("Creating..." + name)
            cv2.imwrite(name, frame)
            frame_count += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
