import json
from collections import Counter
from pathlib import Path

from backend.app.card import card_from_corners
from backend.app.hands import blackjack_hands

ROOT = Path(__file__).resolve().parents[1] / "data" / "blackjack"
SPLIT = ROOT / "splits" / "mixed_ace_seven"


def cards_from_points(points) -> list:
    return [card_from_corners(label, corners) for corners, label in points]


def labels(cards) -> Counter:
    return Counter(card.label for card in cards)


def check_split(name: str) -> tuple[int, int, int]:
    samples = json.loads((SPLIT / f"{name}.json").read_text(encoding="utf-8"))
    checked = matched = missing = 0
    for sample in samples.values():
        image_path = ROOT / sample["img_path"]
        if not image_path.is_file():
            missing += 1
            continue
        cards = cards_from_points(sample["dealer_card_points"]) + cards_from_points(
            sample["player_card_points"]
        )
        hands = blackjack_hands(cards)
        dealer_ok = labels(hands.get("dealer", [])) == labels(
            cards_from_points(sample["dealer_card_points"])
        )
        player_ok = labels(hands.get("player", [])) == labels(
            cards_from_points(sample["player_card_points"])
        )
        checked += 1
        matched += dealer_ok and player_ok
    return checked, matched, missing


def main() -> None:
    for name in ("train", "val"):
        checked, matched, missing = check_split(name)
        print(
            f"{name}: {matched}/{checked} hands matched, "
            f"{missing} images not on disk"
        )


if __name__ == "__main__":
    main()
