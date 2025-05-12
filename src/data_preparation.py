import zipfile
import os
import sys
from tqdm import tqdm
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:  # Explicitly open in read mode
            members = zip_ref.infolist()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}"):
                zip_ref.extract(member, extract_to)
    except zipfile.BadZipFile as e:
        print(
            f"Error: {zip_path} is not a valid ZIP file or is corrupted.  Skipping. Error Details: {e}")
        return  

    except FileNotFoundError as e:
        print(
            f"Error: {zip_path} not found.  Make sure the file exists. Error Details: {e}")
        sys.exit(1) 


if __name__ == "__main__":
    try:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "test"), exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)
    train_zip_path = os.path.join(RAW_DATA_DIR, "train.zip")
    test_zip_path = os.path.join(RAW_DATA_DIR, "test.zip")

    if not os.path.exists(train_zip_path):
        print(
            f"Error: {train_zip_path} not found.  Make sure the file exists.")
        sys.exit(1)
    if not os.path.exists(test_zip_path):
        print(f"Error: {test_zip_path} not found.  Make sure the file exists.")
        sys.exit(1)
    extract_zip(
        train_zip_path,
        str(PROCESSED_DATA_DIR / "train")
    )
    extract_zip(
        test_zip_path,
        str(PROCESSED_DATA_DIR / "test")
    )
    extract_zip(
        os.path.join(RAW_DATA_DIR, "trainLabels.csv.zip"),
        str(PROCESSED_DATA_DIR)
    )
