from unittest import TestCase

from TesseractOCR.TesseractReader import TesseractReader

ocr_expected_data = [{'file': r'..\TestImages\1_13.jpg', 'content':'''BELIEVE IN YOURSELF!
HAVE FAITH IN YOUR ABILITIES!
WITHOUT A HUMBLE BUT REASONABLE
CONFIDENCE IN YOUR OWN POWERS

YOU CANNOT BE SUCCESSFUL OR
HAPPY.

NORMAN VINCENT PEALE

SUCCESS.com'''}]


class OcrTest(TestCase):

    def setUp(self):
        self.ocr_object = TesseractReader()

    def get_images_data(self):
        for data in ocr_expected_data:
            yield data['file'], data['content']

    def test_get_string(self):
        for file, content in self.get_images_data():
            text = self.ocr_object.get_string(file)
            assert text == content

    def test_get_data(self):
        for file, content in self.get_images_data():
            data = self.ocr_object.get_data(file)
            print(data)

    def test_get_boxes(self):
        for file, content in self.get_images_data():
            data = self.ocr_object.get_boxes(file)
            print(data)


