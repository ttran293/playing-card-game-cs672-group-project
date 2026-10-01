# Playing card detection

Detects playing cards in an image with the Roboflow model `playing-cards-ow27d/4` and saves an annotated image plus a text file.

## Setup

Roboflow Inference needs Python 3.10, 3.11, 3.12, or 3.13. Python 3.14 is not supported. This project uses Python 3.12.

From the project folder, create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Install the packages in `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## API key

1. Sign in at [roboflow.com](https://app.roboflow.com/).
2. Open **Account → Roboflow Keys** (or go to [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)).
3. Copy your private API key.
4. Create a `.env` file in this folder:

```
ROBOFLOW_API_KEY=your_key_here
```

Do not commit `.env`. It is listed in `.gitignore`.

## Run

Put an image in this folder named `ex_generated_image.png` or `cards.png`, then:

```powershell
.\.venv\Scripts\python.exe detect_cards.py
```

Or pass any image path:

```powershell
.\.venv\Scripts\python.exe detect_cards.py path\to\photo.jpg
```

The script writes two files next to the input image:

- `<name>_result.png` — boxes and labels drawn on the image
- `<name>_result.txt` — each card, confidence, center, and size

The first run downloads the model and needs a network connection.
