
"""
Download playing card photos from the shared Google Drive folder.

Creates local dataset directories, downloads new images, and converts
iPhone HEIC photos to JPEG for OpenCV.
"""

import shutil
import tempfile
from pathlib import Path

import gdown
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

FOLDER_ID = "1PHnC9VpQLjnUSbdeKTIo35dhUNGg487V"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

FOLDER_NAMES = (
    "Blackjack",
    "Baccarat",
    "Pai-Gow-Poker",
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"
}


def convert_heic(source, destination):
    """Convert an iPhone HEIC/HEIF photo to JPEG."""
    with Image.open(source) as image:
        image.convert("RGB").save(
            destination, "JPEG", quality=90
        )


def sync_dataset():
    """Download the shared dataset and copy new images locally."""

    # Create the local dataset directories.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for folder_name in FOLDER_NAMES:
        (DATA_DIR / folder_name).mkdir(
            parents=True, exist_ok=True
        )

    print("\nChecking Google Drive for dataset updates...")

    downloaded = 0
    skipped = 0

    # Download to a temporary directory first.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        result = gdown.download_folder(
            url=FOLDER_URL,
            output=str(temp_path),
            quiet=True,
        )

        if result is None:
            raise RuntimeError(
                "Could not download the shared Google Drive folder. "
                "Check that Anyone with the link has Viewer access."
            )

        # Locate the game folders inside the downloaded dataset.
        for folder_name in FOLDER_NAMES:
            matches = [
                path
                for path in temp_path.rglob(folder_name)
                if path.is_dir()
            ]

            if not matches:
                raise RuntimeError(
                    f"Could not find Drive folder: {folder_name}"
                )

            remote_folder = matches[0]
            local_folder = DATA_DIR / folder_name

            for source in remote_folder.iterdir():
                if not source.is_file():
                    continue

                extension = source.suffix.lower()

                if extension not in IMAGE_EXTENSIONS:
                    continue

                is_heic = extension in {".heic", ".heif"}

                if is_heic:
                    destination = (
                        local_folder / source.stem
                    ).with_suffix(".jpg")
                else:
                    destination = local_folder / source.name

                # Skip images that already exist.
                if destination.exists():
                    skipped += 1
                    continue

                print(f"Adding {folder_name}/{source.name}")

                if is_heic:
                    convert_heic(source, destination)
                else:
                    shutil.copy2(source, destination)

                downloaded += 1

    print(
        f"Dataset sync complete: "
        f"{downloaded} new, {skipped} already present."
    )


if __name__ == "__main__":
    sync_dataset()
