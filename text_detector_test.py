from unittest import TestCase

import cv2

from TextDetectors.OpenCVObjectDetector import OpenCVObjectDetector
from TextDetectors.available_models import EAST_MODEL

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
        self.ocr_object = OpenCVObjectDetector(EAST_MODEL)

    def get_images_data(self):
        for data in ocr_expected_data:
            yield data['file'], data['content']

    def test_get_boxes(self):
        for file, content in self.get_images_data():
            data = self.ocr_object.get_boxes(file)
            img = cv2.imread(file)
            for box in data:
                img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 3)

            cv2.imwrite("result.png", img)
            # cv2.imshow("x", img)
            # cv2.waitKey(0)
            print(data)


