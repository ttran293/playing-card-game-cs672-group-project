import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

load_dotenv()

IMAGE_CANDIDATES = (
    "ex_generated_image.png",
    "ex_generated_image.jpg",
    "ex_generated_image.jpeg",
    "cards.png",
    "cards.jpg",
)


def resolve_image() -> Path:
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
        return image_path

    for name in IMAGE_CANDIDATES:
        image_path = Path(name)
        if image_path.is_file():
            return image_path

    raise SystemExit(
        "Image not found. Add ex_generated_image.png or cards.png, "
        "or pass an image path."
    )


def intersection_over_union(a, b) -> float:
    a_left, a_top = a.x - a.width / 2, a.y - a.height / 2
    a_right, a_bottom = a.x + a.width / 2, a.y + a.height / 2
    b_left, b_top = b.x - b.width / 2, b.y - b.height / 2
    b_right, b_bottom = b.x + b.width / 2, b.y + b.height / 2

    shared_width = min(a_right, b_right) - max(a_left, b_left)
    shared_height = min(a_bottom, b_bottom) - max(a_top, b_top)
    if shared_width <= 0 or shared_height <= 0:
        return 0.0

    shared = shared_width * shared_height
    union = a.width * a.height + b.width * b.height - shared
    return shared / union


RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "JOKER"}
SUITS = {
    "C": "clubs",
    "D": "diamonds",
    "H": "hearts",
    "S": "spades",
}


@dataclass
class Card:
    label: str
    rank: str
    suit: str | None
    confidence: float
    x: float
    y: float
    width: float
    height: float


def read_card(prediction) -> Card:
    label = prediction.class_name.upper()
    if label == "JOKER":
        rank, suit = "JOKER", None
    elif label.startswith("10") and label[2:] in SUITS:
        rank, suit = "10", label[2:]
    elif len(label) == 2 and label[0] in RANKS and label[1] in SUITS:
        rank, suit = label[0], label[1]
    else:
        raise SystemExit(f"Unrecognized card label: {prediction.class_name}")

    return Card(
        label=label,
        rank=rank,
        suit=suit,
        confidence=prediction.confidence,
        x=prediction.x,
        y=prediction.y,
        width=prediction.width,
        height=prediction.height,
    )


def clean_boxes(predictions, overlap_limit=0.5):
    ordered = sorted(predictions, key=lambda card: card.confidence, reverse=True)
    kept = []
    for card in ordered:
        if any(
            intersection_over_union(card, previous) >= overlap_limit
            for previous in kept
        ):
            continue
        kept.append(card)
    return kept


image_path = resolve_image()
image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Could not read image: {image_path}")

model = get_model(
    model_id="playing-cards-ow27d/4",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

result = model.infer(image, confidence=0.5)[0]
detected = list(result.predictions)
kept = clean_boxes(detected)
cards = [read_card(card) for card in kept]
result.predictions = kept
detections = sv.Detections.from_inference(result)
labels = [f"{card.label} {card.confidence:.0%}" for card in cards]

annotated = sv.BoxAnnotator().annotate(scene=image.copy(), detections=detections)
annotated = sv.LabelAnnotator().annotate(
    scene=annotated, detections=detections, labels=labels
)

result_image = image_path.with_name(f"{image_path.stem}_result.png")
result_text = image_path.with_name(f"{image_path.stem}_result.txt")
if not cv2.imwrite(str(result_image), annotated):
    raise SystemExit(f"Could not save image: {result_image}")

lines = [
    f"Image: {image_path}",
    f"Detected {len(detected)} card(s)",
    f"Kept {len(cards)} after removing stacked boxes",
]
for card in cards:
    suit = SUITS[card.suit] if card.suit else "none"
    lines.append(
        f"{card.label} → rank {card.rank}, suit {suit} "
        f"| {card.confidence:.1%} "
        f"| center=({card.x:.0f}, {card.y:.0f}) "
        f"| size={card.width:.0f}x{card.height:.0f}"
    )
result_text.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
print(f"Saved image: {result_image}")
print(f"Saved text: {result_text}")
