import zipfile
import os
import sys
from tqdm import tqdm
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def combine_zips(zip_prefix, num_parts, output_name):
    """Combine multi-part zip files"""
    if sys.platform == "win32":
        os.system(f'copy /b "{zip_prefix}.*" "{output_name}"')
    else:
        os.system(f'cat "{zip_prefix}".* > "{output_name}"')

def extract_zip(zip_path, extract_to):
    """Extract files with progress bar"""
    with zipfile.ZipFile(zip_path) as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}"):
            try:
                zip_ref.extract(member, extract_to)
            except zipfile.BadZipFile:
                print(f"Skipped corrupt file: {member.filename}")

if __name__ == "__main__":
    # Create processed directories
    (PROCESSED_DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

    # Process training data
    train_zip = RAW_DATA_DIR / "train.zip"
    combine_zips(
        str(RAW_DATA_DIR / "train.zip"), 
        5,
        str(PROCESSED_DATA_DIR / "full_train.zip")
    )
    extract_zip(
        str(PROCESSED_DATA_DIR / "full_train.zip"),
        str(PROCESSED_DATA_DIR / "train")
    )

    # Process test data
    combine_zips(
        str(RAW_DATA_DIR / "test.zip"), 
        7,
        str(PROCESSED_DATA_DIR / "full_test.zip")
    )
    extract_zip(
        str(PROCESSED_DATA_DIR / "full_test.zip"),
        str(PROCESSED_DATA_DIR / "test")
    )

    # Extract labels
    extract_zip(
        str(RAW_DATA_DIR / "trainLabels.csv.zip"),
        str(PROCESSED_DATA_DIR)
    )
