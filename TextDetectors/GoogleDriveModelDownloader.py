from google_drive_downloader import GoogleDriveDownloader as gdd

from TextDetectors.ModelDownloader import ModelDownloader


class GoogleDriveModelDownloader(ModelDownloader):
    def __init__(self, file_id, unzip, destination):
        self.unzip = unzip
        self.file_id = file_id
        super().__init__(destination)

    def download(self):
        gdd.download_file_from_google_drive(file_id=self.file_id,
                                            dest_path=self.destination,
                                            unzip=self.unzip)
