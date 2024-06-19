import cv2
import base64
from PIL import Image
from io import BytesIO


def load_and_encode_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Couldn't load image")

    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, quality=100, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode("ascii")

    return image, img_str
