import time
from nms import nms
from cv2 import cv2

from TextDetectors.ModelDownloader import ModelDownloader
from TextDetectors.available_models import EastModel
from opencv_utils import decode




def scale_box(box, ratio_width, ratio_height):
    return int(box[0] * ratio_width), int(box[1] * ratio_height), int(box[2] * ratio_width), int(box[3] * ratio_height)


class OpenCVObjectDetector:
    def __init__(self, model_downloader:EastModel, width=320, height=320, min_score_confidence=0.5, min_nms_treshold=0.4):
        self.model_downloader = model_downloader
        self.min_nms_treshold = min_nms_treshold
        self.min_confidence = min_score_confidence
        self.width = width
        self.height = height

        self.net = self.model_downloader.read_model()

    def get_boxes(self, image_path):
        image = cv2.imread(image_path)
        (origHeight, origWidth) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (self.width, self.height)
        ratioWidth = origWidth / float(newW)
        ratioHeight = origHeight / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (imageHeight, imageWidth) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True,
                                     crop=False)

        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.model_downloader.forward(self.net)
        end = time.time()
        print(f"time for forward pass:{end-start}")
        (boxes, confidences, baggage) = decode(scores, geometry, self.min_confidence)

        functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]

        indicies = nms.boxes(boxes, confidences, nms_function=functions[0], confidence_threshold=self.min_confidence,
                             nsm_threshold=self.min_nms_treshold)

        return [scale_box(boxes[i], ratioWidth, ratioHeight) for i in indicies]
