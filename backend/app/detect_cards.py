
"""
Detect playing cards in an image using a pretrained Roboflow model.

Large images are resized before processing to improve performance.
The image is divided into overlapping tiles, and each tile is
processed at multiple scales to help detect small card markings.
Duplicate detections are removed before saving the results.
"""

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model
from sklearn.cluster import DBSCAN
from .game_classifier import classify_game

from .card import read_card


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Model settings
MODEL_ID = "playing-cards-ow27d/4"
CONFIDENCE_THRESHOLD = 0.1
OVERLAP_LIMIT = 0.25

# Image preprocessing settings
MAX_IMAGE_DIMENSION = 1280

# Tile settings. Smaller tiles help detect cards in larger table images.
TILE_WIDTH = 200
TILE_HEIGHT = 160
TILE_OVERLAP = 0.30
SCALES = (1.0, 2.0, 3.0)

# Images larger than this (on their longest side) are downscaled before
# tiling so the tile parameters stay consistent across all image sizes.
MAX_INFERENCE_DIM = 1280

IMAGE_CANDIDATES = (
    "data/raw/Pai-Gow-Poker/IMG_0224.JPG",
)


def resolve_image():
    """Find the input image from the command line or default filenames."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        return path

    for name in IMAGE_CANDIDATES:
        path = PROJECT_ROOT / name
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


def deduplicate(predictions):
    """
    Remove near-duplicate detections of the same card label.

    Within each label group, detections whose centers are within
    half a card-width of an already-kept detection are dropped
    (same corner seen twice). Detections far apart with the same
    label are kept (two physical cards of the same type).
    """
    if not predictions:
        return []
    avg_size = sum(max(p.width, p.height) for p in predictions) / len(predictions)
    eps = avg_size * 8.0

    by_label = {}
    for pred in predictions:
        by_label.setdefault(pred.class_name.upper(), []).append(pred)

    kept = []
    for preds in by_label.values():
        # Sort by y ascending so the topmost corner is preferred when
        # collapsing face-card duplicate detections (top vs bottom corner).
        # Use confidence as a tiebreaker for detections at similar y.
        ordered = sorted(preds, key=lambda p: (p.y, -p.confidence))
        label_kept = []
        for pred in ordered:
            if any(math.hypot(pred.x - k.x, pred.y - k.y) < eps for k in label_kept):
                continue
            label_kept.append(pred)
        kept.extend(label_kept)
    return kept


def cluster_cards(predictions, image_width, image_height):
    """
    Group detected cards into spatial clusters using DBSCAN.

    Coordinates are normalized to [0, 1] so eps is resolution-independent.
    min_samples=1 ensures every card belongs to a cluster.

    Returns a list of cluster IDs, one per prediction (same order).
    """
    if not predictions:
        return []

    coords = np.array([
        [p.x / image_width, p.y / image_height]
        for p in predictions
    ])

    labels = DBSCAN(eps=0.112, min_samples=1).fit_predict(coords)
    return labels.tolist()


def detect_tiled(model, image):
    """
    Run card detection on the image.

    Large images are downscaled to MAX_INFERENCE_DIM on the longest side
    before inference so the model receives a consistent resolution.
    Detected coordinates are scaled back to the original image size.
    """
    height, width = image.shape[:2]

    x_positions = tile_positions(width, TILE_WIDTH)
    y_positions = tile_positions(height, TILE_HEIGHT)

    longest = max(width, height)
    if longest > MAX_INFERENCE_DIM:
        pre_scale = MAX_INFERENCE_DIM / longest
        infer_image = cv2.resize(
            image, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_AREA
        )
    else:
        pre_scale = 1.0
        infer_image = image

    result = model.infer(infer_image, confidence=CONFIDENCE_THRESHOLD)[0]

    predictions = []
    for prediction in result.predictions:
        prediction.x /= pre_scale
        prediction.y /= pre_scale
        prediction.width /= pre_scale
        prediction.height /= pre_scale
        predictions.append(prediction)

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



def create_annotations(image, predictions, cluster_ids):
    """Draw bounding boxes and card labels color-coded by cluster."""
    boxes = []
    for p in predictions:
        boxes.append([
            p.x - p.width / 2,
            p.y - p.height / 2,
            p.x + p.width / 2,
            p.y + p.height / 2,
        ])

    xyxy = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    class_id = np.array(cluster_ids, dtype=int)

    detections = sv.Detections(xyxy=xyxy, class_id=class_id)

    labels = [
        f"{read_card(p).label} {p.confidence:.0%} [C{cid + 1}]"
        for p, cid in zip(predictions, cluster_ids)
    ]

    annotated = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.CLASS
    ).annotate(
        scene=image.copy(),
        detections=detections,
    )

    annotated = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.CLASS
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

    # Remove overlapping boxes, then collapse same-label near-duplicates.
    kept = deduplicate(clean_boxes(raw))
    for p in kept:
        print(
            f"{p.class_name}: "
            f"({p.x:.0f}, {p.y:.0f}) "
            f"size={p.width:.0f}x{p.height:.0f}"
        )

    # Cluster cards spatially.
    cluster_ids = cluster_cards(kept, width, height)
    n_clusters = len(set(cluster_ids)) if cluster_ids else 0

    print(f"\nRaw detections: {len(raw)}")
    print(f"After duplicate removal: {len(kept)}")
    print(f"Clusters found: {n_clusters}")

    # Group cards by cluster for reporting.
    clusters = {}
    for pred, cid in zip(kept, cluster_ids):
        clusters.setdefault(cid, []).append(read_card(pred))

    game = classify_game(clusters)
    print(f"Detected game: {game}")

    # Build the text report.
    lines = [
        f"Image: {image_path}",
        f"Original resolution: {original_width}x{original_height}",
        f"Processing resolution: {width}x{height}",
        f"Raw detections: {len(raw)}",
        f"Kept detections: {len(kept)}",
        f"Clusters: {n_clusters}",
        f"Detected game: {game}",
    ]

    for cid in sorted(clusters):
        cards_in_cluster = clusters[cid]
        card_labels = ", ".join(c.label for c in cards_in_cluster)
        lines.append(
            f"  Cluster {cid + 1} ({len(cards_in_cluster)} card(s)): {card_labels}"
        )

    lines.append("")
    for p in kept:
        card = read_card(p)
        lines.append(
            f"{card.label} -> rank {card.rank}, "
            f"suit {card.suit_name} "
            f"| confidence {card.confidence:.1%} "
            f"| center ({card.x:.0f},{card.y:.0f}) "
            f"| size {card.width:.0f}x{card.height:.0f}"
        )

    annotated = create_annotations(image, kept, cluster_ids)

    # Save outputs to results/ subfolders.
    results_dir = PROJECT_ROOT / "results"
    (results_dir / "images").mkdir(parents=True, exist_ok=True)
    (results_dir / "txt").mkdir(parents=True, exist_ok=True)

    result_image = results_dir / "images" / f"{image_path.stem}_result.png"
    result_text = results_dir / "txt" / f"{image_path.stem}_result.txt"
    raw_text = results_dir / "txt" / f"{image_path.stem}_raw.txt"

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
