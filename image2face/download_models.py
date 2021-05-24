import os
import zipfile
import shutil
from pathlib import Path
from .retinaface.utils import download_file_from_drive


def download_models():
    dir_path = Path(__file__).parent
    zip_file = dir_path / "models.zip"

    if os.path.exists(zip_file):
        return

    print("Downloading models...")
    download_file_from_drive("1QTZyChGlaZZDAU267AZAgO4HZwnYbj4f", zip_file)
    print("Extracting models...")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dir_path)

        origin_retinaface_path = str(dir_path / "pretrained_models/retinaface")
        origin_arcface_path = str(dir_path / "pretrained_models/arcface")

        new_retinaface_path = str(dir_path / "retinaface/weights/")
        new_arcface_path = str(dir_path / "arcface/weights/")

        if not os.path.exists(new_retinaface_path):
            os.mkdir(new_retinaface_path)

        if not os.path.exists(new_arcface_path):
            os.mkdir(new_arcface_path)

        for file_name in os.listdir(origin_retinaface_path):
            shutil.move(origin_retinaface_path + "/" + file_name, new_retinaface_path)

        for file_name in os.listdir(origin_arcface_path):
            shutil.move(origin_arcface_path + "/" + file_name, new_arcface_path)
