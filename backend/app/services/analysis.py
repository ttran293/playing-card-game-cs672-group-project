"""Read an upload, run detection, and prepare the API result."""

import base64
from dataclasses import asdict
from io import BytesIO
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

model = None


def read_image(contents):
    try:
        with Image.open(BytesIO(contents)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("Choose an image smaller than 40 megapixels.")
            source = ImageOps.exif_transpose(source)
            pixels = np.array(source.convert("RGB"))
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Could not read this image. Try a JPEG, PNG, or WebP file.")


def load_detector(pipeline):
    global model
    if model is None:
        model = pipeline.get_model(
            model_id=pipeline.MODEL_ID,
            api_key=os.environ["ROBOFLOW_API_KEY"],
        )
    return model


def encode_image(image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode annotated image")
    encoded_text = base64.b64encode(encoded).decode("ascii")
    return "data:image/png;base64," + encoded_text


def analyze_image(contents):
    image = read_image(contents)

    # Import the heavy inference package only for valid image uploads.
    from ..logic import detect_cards as pipeline

    started = time.perf_counter()
    image = pipeline.standardize_image(image)
    height, width = image.shape[:2]
    detector = load_detector(pipeline)
    raw = pipeline.detect_tiled(detector, image)
    kept = pipeline.clean_boxes(raw)
    kept = pipeline.deduplicate(kept)
    cluster_ids = pipeline.cluster_cards(kept, width, height)

    clusters = {}
    cards = []
    for prediction, cluster_id in zip(kept, cluster_ids):
        try:
            card = pipeline.read_card(prediction)
        except SystemExit:
            raise RuntimeError("Model returned an unsupported card label")

        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(card)

        card_data = asdict(card)
        card_data["cluster_id"] = cluster_id + 1
        cards.append(card_data)

    cluster_results = []
    for cluster_id in sorted(clusters):
        labels = []
        for card in clusters[cluster_id]:
            labels.append(card.label)
        cluster_results.append({"id": cluster_id + 1, "cards": labels})

    annotated = pipeline.create_annotations(image, kept, cluster_ids)
    return {
        "game": pipeline.classify_game(clusters),
        "cards": cards,
        "clusters": cluster_results,
        "raw_count": len(raw),
        "width": width,
        "height": height,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "annotated_image": encode_image(annotated),
    }
