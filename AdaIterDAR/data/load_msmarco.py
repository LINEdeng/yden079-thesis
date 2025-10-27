from datasets import load_dataset
import json
from pathlib import Path


def load_msmarco(subset: str = "train", max_samples: int = None):
    print(f"Loading MS MARCO ({subset}) from Hugging Face...")
    dataset = load_dataset("ms_marco", "v2.1")

    if subset not in dataset:
        print(f"Split '{subset}' not found, using 'train' by default.")
        subset = "train"

    data = dataset[subset]
    records = data[:] if not max_samples else data[:max_samples]
    print(f"Loaded {len(records)} MS MARCO samples.")
    return records


def save_msmarco(data, file_path: str = "data/msmarco.json"):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(data)} samples to {file_path.resolve()}...")
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Save complete.")


if __name__ == "__main__":
    subset = "train"
    max_samples = 1000
    data = load_msmarco(subset=subset)
    save_msmarco(data)
