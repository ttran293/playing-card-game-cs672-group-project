# Playing Card Detection and Game Classification

CMSC 672 Computer Vision Group Project

Detects and identifies playing cards in photographs using the pretrained Roboflow model `playing-cards-ow27d/4`.

The program supports photographs of **Blackjack, EZ Baccarat, and Pai Gow Poker**. It automatically resizes large images, performs object detection using overlapping image tiles, removes duplicate detections, groups cards into spatial clusters, and uses rule-based classification to identify the game.

The program saves annotated images and text reports containing detected cards, clusters, and game classifications.

## Requirements

- Python 3.12 (recommended)
- Git
- Internet connection for setup and dataset downloads
- Roboflow API key

Python 3.14 is not supported by the project's current inference dependencies.

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

[Open the shared Google Drive dataset](https://drive.google.com/drive/folders/1PHnC9VpQLjnUSbdeKTIo35dhUNGg487V?usp=sharing)

It contains photographs organized into three categories:

- `Blackjack`
- `Baccarat`
- `Pai-Gow-Poker`

The setup script automatically downloads the dataset into `data/raw/`.

### Project Directory Structure

```text
playing-card-game-cs672-group-project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point (/api/health)
│   │   ├── detect_cards.py      # Current detection and DBSCAN pipeline
│   │   ├── card.py
│   │   ├── game_classifier.py
│   │   └── hands.py             # Experimental alternative grouping
│   └── requirements.txt
├── frontend/                   # React + JavaScript + Vite
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── scripts/                    # Dataset download tools
├── evaluation/                 # Offline grouping evaluation
├── examples/                   # Existing sample images and outputs
├── data/raw/                   # Downloaded photos (ignored)
├── results/                    # Generated analysis (ignored)
├── .env.example
├── requirements.txt            # Includes backend requirements
├── setup.sh
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

## 5. Run Card Detection and Classification

After setup and API key configuration, run `detect_cards.py`.

### Windows (PowerShell)

Run using the default image:

```powershell
.\.venv\Scripts\python.exe -m backend.app.detect_cards
```

Or process a specific image:

```powershell
.\.venv\Scripts\python.exe -m backend.app.detect_cards "data/raw/Blackjack/IMG_0221.JPG"
```

### Linux and macOS

Run using the default image:

```bash
./.venv/bin/python -m backend.app.detect_cards
```

Or process a specific image:

```bash
./.venv/bin/python -m backend.app.detect_cards "data/raw/Blackjack/IMG_0221.JPG"
```

Replace `IMG_0221.JPG` with the name of an existing photograph.

### Selecting a Default Image

You can specify a default image directly in `backend/app/detect_cards.py` by modifying `IMAGE_CANDIDATES`:

```python
IMAGE_CANDIDATES = (
    "data/raw/Blackjack/IMG_0221.JPG",
)
```

When the script runs without an image argument, it uses the first existing image listed in `IMAGE_CANDIDATES`.

A command-line image path overrides the default selection.

## 6. Image Processing Pipeline

The program performs the following operations:

1. **Image resizing:** Images larger than 1280 pixels on their longest side are reduced while preserving their aspect ratio.
2. **Tiled inference:** The image is divided into overlapping tiles to help identify small card markings.
3. **Multiscale detection:** Tiles are processed at multiple scales using the Roboflow model.
4. **Duplicate removal:** Overlapping predictions are filtered using Intersection over Union (IoU), followed by additional filtering of nearby duplicate card labels.
5. **Card clustering:** Detected cards are grouped into spatial clusters using DBSCAN.
6. **Game classification:** The number of cards in each cluster is analyzed to determine the game.
7. **Output generation:** Annotated images and text reports are saved.

The original input photographs are not modified.

Detection coordinates refer to the resized processing image when resizing occurs.

The first detection run may take additional time because the pretrained model must be downloaded. Inference may be slower on systems using CPU processing rather than GPU acceleration.

## 7. Card Clustering

The program uses **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to group detected cards based on their positions in the image.

Card coordinates are normalized relative to image dimensions before clustering.

The current implementation uses:

```python
DBSCAN(eps=0.112, min_samples=1)
```

The `eps` parameter controls the maximum neighborhood distance between detected cards. The `min_samples=1` setting ensures every detected card is assigned to a cluster.

Each cluster represents a group of nearby cards, which can correspond to a player's hand or a dealer's hand.

Clusters are assigned numeric IDs and displayed using different colors in the annotated output.

**Current limitation:** Hands positioned close together may be merged into a single cluster, while cards positioned far apart may be separated incorrectly. Clustering parameters may require further refinement for different table layouts.

## 8. Game Classification

The initial game classifier uses **rule-based logic** rather than a trained neural network.

It analyzes the number of detected clusters and the number of cards in each cluster.

The current classification rules are evaluated in the following order:

| Game | Classification Rule |
|---|---|
| Pai Gow Poker | At least two seven-card clusters, or at least two five-card clusters and two two-card clusters |
| EZ Baccarat | Exactly two clusters, each containing two or three cards |
| Blackjack | At least two clusters, with no more than one single-card cluster |
| Unknown | None of the above conditions are satisfied |

The Blackjack rule allows one single-card cluster because the dealer's face-up card may overlap the face-down card, making two physical cards appear as one detection.

The classifier returns one of four labels:

- `Blackjack`
- `EZ Baccarat`
- `Pai Gow Poker`
- `Unknown`

### Example Classification Output

```text
Clusters: 2
Detected game: EZ Baccarat
  Cluster 1 (2 card(s)): 4H, 6C
  Cluster 2 (3 card(s)): 3H, QD, 3D
```

The game classification is displayed in the terminal and included in the saved text report.

### Current Limitations

The rule-based classifier depends on accurate card detection and clustering.

Missing cards, duplicate detections, incorrect card labels, and merged clusters can cause incorrect classifications.

Some Blackjack and Baccarat layouts may also have similar cluster structures.

Future development may include training a machine-learning classifier using labeled photographs from the three game categories.

## 9. Output Files

For an input photograph named:

```text
IMG_0221.JPG
```

The program creates three files:

| Output File | Location | Description |
|---|---|---|
| `IMG_0221_result.png` | `results/images/` | Annotated image with card labels, bounding boxes, and cluster IDs |
| `IMG_0221_result.txt` | `results/txt/` | Game classification, clusters, card labels, confidence scores, and coordinates |
| `IMG_0221_raw.txt` | `results/txt/` | Raw model predictions before duplicate removal |

These outputs can be used for debugging, evaluation, and later stages of the project, including game-specific hand evaluation and winner determination.

**Note:** Images with identical filenames from different dataset folders currently produce the same output filenames and may overwrite previous results.

## 10. Git and Dataset Files

**Do not commit dataset photographs, generated model outputs, virtual environments, or API keys to GitHub.**

Make sure `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
*.pyc
data/
results/
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

### Incorrect Game Classification

Check the detected card labels and cluster sizes in the output text report.

Incorrect classifications may result from duplicate card detections, missing cards, or clusters that combine multiple hands.

### Inference Warnings

The inference library may display warnings about optional models, deprecated features, or unavailable GPU providers.

These warnings do not necessarily prevent the playing-card detection model from running.
## Local web development

The web application currently has a React shell and a FastAPI health endpoint.
Image upload and analysis endpoints have not been implemented yet.
Run commands from the repository root unless noted otherwise.

Install backend dependencies (Python 3.12):

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

Start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Install Node.js 22.12+ (or a supported newer LTS). In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally http://localhost:5173).
Vite forwards `/api` requests to the local FastAPI server.
Run `npm run build` inside `frontend` to create `frontend/dist`.
On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `./.venv/bin/python`.

Dataset utilities now live under `scripts/`:

```powershell
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\python.exe scripts/download_blackjack.py
```

`evaluation/check_clusters.py` is an experimental legacy evaluator, separate from
DBSCAN in the current pipeline. It requires the missing `card_from_corners`
adapter and labeled local blackjack images before it can run. It is not part of
web startup or the main detection command.
