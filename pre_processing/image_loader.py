import cv2
import base64
from PIL import Image
from io import BytesIO


class ImageLoader:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Couldn't load image")
        return image

    def encode_image(self):
        img = Image.open(self.image_path)
        buffered = BytesIO()
        img.save(buffered, quality=100, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue())
        img_str = img_bytes.decode("ascii")
        return img_str

    def load_and_encode_image(self):
        image = self.load_image()
        img_str = self.encode_image()
        return image, img_str
