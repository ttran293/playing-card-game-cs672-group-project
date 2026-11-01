
# Playing Card Detection

CMSC 672 Computer Vision Group Project

Detects and identifies playing cards in photographs using the pretrained Roboflow model `playing-cards-ow27d/4`.

The program supports photographs of Blackjack, EZ Baccarat, and Pai Gow Poker. It automatically resizes large images, performs object detection using overlapping image tiles, removes duplicate detections, and saves annotated images and text reports.

## Requirements

- Python 3.12 (recommended)
- Git
- Internet connection for setup and dataset downloads
- Roboflow API key

Python 3.14 is not supported by the project's inference dependencies.

## 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd playing-card-game-cs672-group-project
```

Replace `<repository-url>` with the actual GitHub repository URL.

## 2. Setup

The `setup.sh` script automatically:

- Creates a Python virtual environment (`.venv`)
- Installs the required Python packages
- Creates the dataset directories
- Downloads the shared Google Drive dataset

Run setup initially and whenever you want to update the dataset or dependencies.

### Windows

1. Install [Python 3.12](https://www.python.org/downloads/) and [Git for Windows](https://git-scm.com/downloads/win).
2. Open the repository folder in File Explorer.
3. Right-click inside the folder and select **Open Git Bash here**.
4. Run:

```bash
bash setup.sh
```

Alternatively, from PowerShell:

```powershell
& "C:\Program Files\Git\bin\bash.exe" setup.sh
```

### Linux

1. Install Python 3.12, Git, and Python virtual environment support using your distribution's package manager.
2. Open a terminal in the repository folder.
3. Run:

```bash
bash setup.sh
```

### macOS

1. Install Python 3.12 and Git if they are not already installed.
2. Open Terminal in the repository folder.
3. Run:

```bash
bash setup.sh
```

**Note:** Python 3.12 must be installed and accessible from the terminal.

## 3. Configure Your Roboflow API Key

1. Sign in at [Roboflow](https://app.roboflow.com/).
2. Open [Roboflow API Settings](https://app.roboflow.com/settings/api).
3. Copy your API key.
4. Create a file named `.env` in the repository's root directory.
5. Add the following line:

```dotenv
ROBOFLOW_API_KEY=your_key_here
```

Replace `your_key_here` with your actual API key.

**Do not commit `.env` to GitHub.** It is excluded by `.gitignore`.

## 4. Dataset

The dataset is hosted on Google Drive rather than GitHub.

**[Open the shared Google Drive dataset](https://drive.google.com/drive/folders/1PHnC9VpQLjnUSbdeKTIo35dhUNGg487V?usp=sharing)**

It contains photographs organized into three categories:

- `Blackjack`
- `EZ-Baccarat`
- `Pai-Gow-Poker`

The setup script automatically downloads the dataset into `data/raw/`.

### Project Directory Structure

```text
playing-card-game-cs672-group-project/
├── .venv/
├── .env
├── data/
│   └── raw/
│       ├── Blackjack/
│       ├── EZ-Baccarat/
│       └── Pai-Gow-Poker/
├── setup.sh
├── download_dataset.py
├── detect_cards.py
├── card.py
├── requirements.txt
└── README.md
```

### Updating the Dataset

New photographs will be added to the shared Google Drive folder throughout the project.

To download new photographs, rerun:

```bash
bash setup.sh
```

The downloader skips existing local filenames when copying downloaded images into `data/raw/`.

The synchronization process may temporarily download files to check for updates.

iPhone HEIC/HEIF photographs are automatically converted to JPEG for compatibility with OpenCV.

**Dataset synchronization occurs during setup, not during normal detection runs.**

## 5. Run Card Detection

After setup and API key configuration, run `detect_cards.py`.

### Windows (PowerShell)

Run using the default image:

```powershell
.\.venv\Scripts\python.exe detect_cards.py
```

Or process a specific image:

```powershell
.\.venv\Scripts\python.exe detect_cards.py "data/raw/Blackjack/IMG_0221.JPG"
```

### Linux and macOS

Run using the default image:

```bash
./.venv/bin/python detect_cards.py
```

Or process a specific image:

```bash
./.venv/bin/python detect_cards.py "data/raw/Blackjack/IMG_0221.JPG"
```

Replace `IMG_0221.JPG` with the name of an existing photograph.

### Selecting a Default Image

You can specify a default image directly in `detect_cards.py` by modifying `IMAGE_CANDIDATES`:

```python
IMAGE_CANDIDATES = (
    "data/raw/Blackjack/IMG_0221.JPG",
)
```

When the script runs without an image argument, it uses the first existing image listed in `IMAGE_CANDIDATES`.

A command-line image path overrides the default selection.

## 6. Image Processing

The detection pipeline performs the following operations:

1. **Image resizing:** Images larger than 1280 pixels on their longest side are reduced while preserving their aspect ratio. Smaller images are not enlarged.
2. **Tiled inference:** The image is divided into overlapping tiles to help identify small card markings.
3. **Multiscale detection:** Tiles are processed at multiple scales using the Roboflow model.
4. **Duplicate removal:** Overlapping predictions are filtered using Intersection over Union (IoU).
5. **Output generation:** Bounding boxes, card labels, confidence scores, and position information are saved.

The original input photographs are not modified.

Detection coordinates refer to the resized processing image when resizing occurs.

The first detection run may take additional time because the pretrained model must be downloaded. Inference may be slower on systems using CPU processing rather than GPU acceleration.

## 7. Output Files

For an input photograph named:

```text
IMG_0221.JPG
```

The program creates three files in the same directory:

| Output File | Description |
|---|---|
| `IMG_0221_result.png` | Annotated image with detected card bounding boxes and labels |
| `IMG_0221_result.txt` | Detected cards, ranks, suits, confidence scores, coordinates, and dimensions |
| `IMG_0221_raw.txt` | Raw model predictions before duplicate removal |

These outputs can be used for debugging, evaluation, and later stages of the project, including card clustering and game identification.

## 8. Git and Dataset Files

**Do not commit dataset photographs, model outputs, virtual environments, or API keys to GitHub.**

Make sure `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
*.pyc
data/
```

The dataset remains on Google Drive, and each team member downloads their own local copy using `setup.sh`.

## Troubleshooting

### Python Not Found

Install Python 3.12 and make sure it is available from your terminal.

On Windows, ensure Python is added to PATH.

### Virtual Environment Creation Fails

Verify that your Python installation includes the `venv` module.

Some Linux distributions require a separate Python virtual environment package.

### Google Drive Download Fails

Check your internet connection and verify that the shared dataset folder is accessible through the Google Drive link.

### Missing Roboflow API Key

Make sure `.env` exists in the repository's root directory and contains:

```dotenv
ROBOFLOW_API_KEY=your_key_here
```

### Image Not Found

Verify that the image exists under `data/raw/` and that the filename and extension match exactly.

Filenames are case-sensitive on Linux and commonly on macOS.

### Detection Is Slow

The program uses multiple overlapping tiles and inference scales.

CPU-only processing can take longer than GPU processing.
