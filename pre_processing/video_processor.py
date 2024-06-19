import cv2
import os


class VideoProcessor:
    def __init__(self, video_path, output_dir="data"):
        self.video_path = video_path
        self.output_dir = output_dir

    def create_dir(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except OSError as e:
            print(f"Failed to create directory: {e}")

    def extract_frames(self, frames_per_second: int = 10):
        self.create_dir()

        cam = cv2.VideoCapture(self.video_path)

        if not cam.isOpened():
            print("Error: Could not open video.")
            return

        frame_rate = cam.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate // frames_per_second)

        frame_count = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            name = os.path.join(self.output_dir, f"frame{frame_count}.jpg")
            print(f"Creating {name}")
            cv2.imwrite(name, frame)
            frame_count += 1

        cam.release()
        cv2.destroyAllWindows()
