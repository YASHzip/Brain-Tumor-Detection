import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_extract(dataset, dest_dir="data"):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=".", unzip=False)

    zip_filename = dataset.split("/")[-1] + ".zip"
    print(f"Downloaded to {zip_filename}")

    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")

if __name__ == "__main__":
    download_and_extract("masoudnickparvar/brain-tumor-mri-dataset")