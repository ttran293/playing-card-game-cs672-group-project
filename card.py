from dataclasses import dataclass

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

    @property
    def suit_name(self) -> str:
        if self.suit is None:
            return "none"
        return SUITS[self.suit]


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
