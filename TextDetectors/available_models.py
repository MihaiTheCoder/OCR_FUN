from TextDetectors.GoogleDriveModelDownloader import GoogleDriveModelDownloader


class EastModel(GoogleDriveModelDownloader):
    def __init__(self):
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        self.east_model_path = './Models/frozen_east_text_detection.pb'
        super().__init__('1WRmjVM90MdL_OSY9Ysl-vxetBVFh1fNA', False, self.east_model_path)

    def forward(self, net):
        return net.forward(self.layerNames)


EAST_MODEL = EastModel()
