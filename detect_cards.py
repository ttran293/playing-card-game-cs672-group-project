
"""
Detect playing cards in an image using a pretrained Roboflow model.

Large images are resized before processing to improve performance.
The image is divided into overlapping tiles, and each tile is
processed at multiple scales to help detect small card markings.
Duplicate detections are removed before saving the results.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

from card import read_card


load_dotenv()

# Model settings
MODEL_ID = "playing-cards-ow27d/4"
CONFIDENCE_THRESHOLD = 0.1
OVERLAP_LIMIT = 0.5

# Image preprocessing settings
MAX_IMAGE_DIMENSION = 1280

# Tile settings. Smaller tiles help detect cards in larger table images.
TILE_WIDTH = 200
TILE_HEIGHT = 160
TILE_OVERLAP = 0.30
SCALES = (1.0, 2.0, 3.0)

IMAGE_CANDIDATES = (
    "data/raw/Blackjack/IMG_0222.JPG",
)


def resolve_image():
    """Find the input image from the command line or default filenames."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        return path

    for name in IMAGE_CANDIDATES:
        path = Path(name)
        if path.is_file():
            return path

    raise SystemExit("No input image found.")


def standardize_image(image):
    """
    Resize large images to a consistent maximum dimension.

    Preserve the original aspect ratio and avoid enlarging
    smaller images. The original image file is not modified.
    """
    height, width = image.shape[:2]

    scale = min(
        1.0,
        MAX_IMAGE_DIMENSION / max(width, height),
    )

    if scale == 1.0:
        return image

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def tile_positions(length, tile_size):
    """Calculate starting positions for overlapping tiles."""
    if length <= tile_size:
        return [0]

    # Move each tile forward by its size minus the overlap.
    stride = max(1, int(tile_size * (1 - TILE_OVERLAP)))
    last = length - tile_size

    positions = list(range(0, last + 1, stride))

    # Make sure the last tile reaches the edge of the image.
    if positions[-1] != last:
        positions.append(last)

    return positions


def intersection_over_union(a, b):
    """
    Calculate the overlap between two bounding boxes.

    IoU is the intersection area divided by the combined area.
    It is used to identify duplicate detections.
    """
    left = max(a.x - a.width / 2, b.x - b.width / 2)
    top = max(a.y - a.height / 2, b.y - b.height / 2)
    right = min(a.x + a.width / 2, b.x + b.width / 2)
    bottom = min(a.y + a.height / 2, b.y + b.height / 2)

    width = max(0, right - left)
    height = max(0, bottom - top)

    intersection = width * height
    union = a.width * a.height + b.width * b.height - intersection

    return intersection / union if union > 0 else 0.0


def clean_boxes(predictions):
    """
    Remove duplicate detections using IoU.

    Higher-confidence predictions are kept first. Any later
    prediction that overlaps an existing one too much is removed.
    """
    ordered = sorted(
        predictions,
        key=lambda p: p.confidence,
        reverse=True,
    )

    kept = []

    for prediction in ordered:
        duplicate = any(
            intersection_over_union(prediction, previous)
            >= OVERLAP_LIMIT
            for previous in kept
        )

        if not duplicate:
            kept.append(prediction)

    return kept


def detect_tiled(model, image):
    """
    Run card detection on overlapping tiles at multiple scales.

    Detected coordinates are converted back to the standardized
    image coordinates so they can be used for clustering later.
    """
    height, width = image.shape[:2]

    x_positions = tile_positions(width, TILE_WIDTH)
    y_positions = tile_positions(height, TILE_HEIGHT)

    predictions = []

    for scale in SCALES:
        count = 0

        for y in y_positions:
            for x in x_positions:
                # Crop the current tile from the image.
                tile = image[
                    y:min(y + TILE_HEIGHT, height),
                    x:min(x + TILE_WIDTH, width),
                ]

                # Test the tile at different resolutions.
                if scale != 1.0:
                    tile = cv2.resize(
                        tile,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )

                result = model.infer(
                    tile,
                    confidence=CONFIDENCE_THRESHOLD,
                )[0]

                for prediction in result.predictions:
                    # Predictions are relative to the scaled tile.
                    # Convert them back to the standardized image coordinates.
                    prediction.x = prediction.x / scale + x
                    prediction.y = prediction.y / scale + y
                    prediction.width /= scale
                    prediction.height /= scale

                    predictions.append(prediction)
                    count += 1

        print(f"Scale {scale}: {count} raw detections")

    return predictions


def create_annotations(image, predictions):
    """Draw bounding boxes and card labels on the processed image."""
    boxes = []

    for p in predictions:
        # Convert center coordinates to bounding box corners.
        boxes.append([
            p.x - p.width / 2,
            p.y - p.height / 2,
            p.x + p.width / 2,
            p.y + p.height / 2,
        ])

    xyxy = np.array(
        boxes, dtype=np.float32
    ).reshape(-1, 4)

    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{read_card(p).label} {p.confidence:.0%}"
        for p in predictions
    ]

    # Use index-based colors since class IDs are not provided.
    annotated = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX
    ).annotate(
        scene=image.copy(),
        detections=detections,
    )

    annotated = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX
    ).annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )

    return annotated


def main():
    """Load the image, detect cards, and save the results."""
    image_path = resolve_image()
    image = cv2.imread(str(image_path))

    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    # Record the original dimensions before preprocessing.
    original_height, original_width = image.shape[:2]

    # Standardize large images before running inference.
    image = standardize_image(image)

    height, width = image.shape[:2]

    print(f"Image: {image_path}")
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Processing resolution: {width}x{height}")
    print(f"Tile size: {TILE_WIDTH}x{TILE_HEIGHT}")

    # Load the pretrained Roboflow model.
    model = get_model(
        model_id=MODEL_ID,
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    print("\nRunning tiled inference...")

    raw = detect_tiled(model, image)

    # Remove duplicate predictions from overlapping tiles and scales.
    kept = clean_boxes(raw)

    print(f"\nRaw detections: {len(raw)}")
    print(f"After duplicate removal: {len(kept)}")

    # Build the text report containing the detected cards.
    lines = [
        f"Image: {image_path}",
        f"Original resolution: {original_width}x{original_height}",
        f"Processing resolution: {width}x{height}",
        f"Raw detections: {len(raw)}",
        f"Kept detections: {len(kept)}",
    ]

    for p in kept:
        card = read_card(p)

        lines.append(
            f"{card.label} -> rank {card.rank}, "
            f"suit {card.suit_name} "
            f"| confidence {card.confidence:.1%} "
            f"| center ({card.x:.0f},{card.y:.0f}) "
            f"| size {card.width:.0f}x{card.height:.0f}"
        )

    annotated = create_annotations(image, kept)

    # Save outputs using the input image's filename.
    result_image = image_path.with_name(
        f"{image_path.stem}_result.png"
    )
    result_text = image_path.with_name(
        f"{image_path.stem}_result.txt"
    )
    raw_text = image_path.with_name(
        f"{image_path.stem}_raw.txt"
    )

    if not cv2.imwrite(str(result_image), annotated):
        raise SystemExit(f"Could not save image: {result_image}")

    result_text.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # Keep raw predictions for debugging and later evaluation.
    raw_lines = []

    for p in raw:
        raw_lines.append(
            f"{p.class_name} "
            f"| confidence {p.confidence:.3f} "
            f"| center ({p.x:.1f},{p.y:.1f}) "
            f"| size {p.width:.1f}x{p.height:.1f}"
        )

    raw_text.write_text(
        "\n".join(raw_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nSaved image: {result_image}")
    print(f"Saved text: {result_text}")
    print(f"Saved raw predictions: {raw_text}")


if __name__ == "__main__":
    main()
