
from collections import Counter


def classify_game(clusters):
    """
    Classify the game based on the number of cards
    in each detected cluster.
    """
    sizes = [len(cards) for cards in clusters.values()]

    if not sizes:
        return "Unknown"

    counts = Counter(sizes)

    # Pai Gow Poker
    if counts[7] >= 2:
        return "Pai Gow Poker"

    if counts[5] >= 2 and counts[2] >= 2:
        return "Pai Gow Poker"

    # EZ Baccarat
    if len(sizes) == 2 and all(2 <= n <= 3 for n in sizes):
        return "EZ Baccarat"

    # Blackjack
    if len(sizes) >= 2 and sizes.count(1) <= 1:
        return "Blackjack"
    
    return "Unknown"
