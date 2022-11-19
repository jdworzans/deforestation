import os
import shutil
import requests
import zipfile
from pathlib import Path

DATA_DIR = 'data'


def unzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def download_file(url, folder_name):
    local_filename = url.split('/')[-1]
    path = Path(folder_name, local_filename)
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


def download_and_unzip_deforestation_images():
    filename = download_file(
        'https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/train_test_data.zip',
        DATA_DIR)
    unzip(Path(DATA_DIR, filename), DATA_DIR)
    os.remove(Path(DATA_DIR, filename))


def download_train_test():
    download_file(
        'https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/train.csv',
        DATA_DIR)
    download_file(
        'https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/test.csv',
        DATA_DIR)


if __name__ == '__main__':
    download_train_test()
    download_and_unzip_deforestation_images()
