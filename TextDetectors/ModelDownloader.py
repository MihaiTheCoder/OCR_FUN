import os
from abc import ABC, abstractmethod

import cv2


class ModelDownloader(ABC):
    def __init__(self, destination):
        self.destination = destination

    @abstractmethod
    def download(self):
        pass

    def download_or_get_path(self):
        if os.path.exists(self.destination):
            return self.destination

        self.download()

        return self.destination

    def read_model(self):
        return cv2.dnn.readNet(self.download_or_get_path())

    @abstractmethod
    def forward(self, net):
        pass
