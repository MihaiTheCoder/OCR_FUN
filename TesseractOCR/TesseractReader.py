import pytesseract
from PIL import Image


class TesseractReader:

    def get_string(self, image_path):
        return pytesseract.image_to_string(Image.open(image_path))

    def get_data(self, image_path):
        return pytesseract.image_to_data(Image.open(image_path))

    def get_boxes(self, image_path):
        return pytesseract.image_to_boxes(Image.open(image_path))
