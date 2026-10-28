from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

REPO_ID = "JackFurby/blackjack"
DEST = Path(__file__).resolve().parent / "data" / "blackjack"
FILES = [
    "splits/mixed_ace_seven/classes.txt",
    "splits/mixed_ace_seven/concepts.txt",
    "splits/mixed_ace_seven/info.txt",
    "splits/mixed_ace_seven/train.json",
    "splits/mixed_ace_seven/train.pkl",
    "splits/mixed_ace_seven/val.json",
    "splits/mixed_ace_seven/val.pkl",
]


def main() -> None:
    for filename in FILES:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DEST,
        )
        print(filename)


if __name__ == "__main__":
    main()
