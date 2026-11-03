from .card import Card


def _edge_gap(a: Card, b: Card) -> tuple[float, float]:
    gap_x = abs(a.x - b.x) - (a.width + b.width) / 2
    gap_y = abs(a.y - b.y) - (a.height + b.height) / 2
    return gap_x, gap_y


def same_hand(a: Card, b: Card, gap_factor: float = 1.0) -> bool:
    gap_x, gap_y = _edge_gap(a, b)
    return gap_x <= gap_factor * min(a.width, b.width) and gap_y <= gap_factor * min(
        a.height, b.height
    )


def cluster_cards(cards: list[Card], gap_factor: float = 1.0) -> list[list[Card]]:
    parent = list(range(len(cards)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, card in enumerate(cards):
        for j in range(i + 1, len(cards)):
            if same_hand(card, cards[j], gap_factor):
                union(i, j)

    piles: dict[int, list[Card]] = {}
    for index, card in enumerate(cards):
        piles.setdefault(find(index), []).append(card)
    return list(piles.values())


def blackjack_hands(cards: list[Card], gap_factor: float = 1.0) -> dict[str, list[Card]]:
    piles = cluster_cards(cards, gap_factor)
    piles.sort(key=lambda pile: sum(card.y for card in pile) / len(pile))
    hands: dict[str, list[Card]] = {}
    if not piles:
        return hands
    hands["dealer"] = sorted(piles[0], key=lambda card: card.x)
    players = piles[1:]
    for index, pile in enumerate(players):
        name = "player" if len(players) == 1 else f"player_{index + 1}"
        hands[name] = sorted(pile, key=lambda card: card.x)
    return hands
