from datasets import load_dataset
import json
from pathlib import Path


def load_pubmedqa(subset: str = "pqa_labeled", max_samples: int = None):
    print(f"Loading PubMedQA ({subset}) from Hugging Face...")
    dataset = load_dataset("pubmed_qa", subset)
    data = dataset["train"]
    records = data[:] if not max_samples else data[:max_samples]
    print(f"Loaded {len(records):,} PubMedQA samples (subset: {subset}).")
    return records

def save_pubmedqa(data, file_path: str = "data/pubmedqa.json"):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(data):,} samples to {file_path.resolve()}...")
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Save complete.")


if __name__ == "__main__":
    subset = "pqa_labeled"
    data = load_pubmedqa(subset=subset)
    save_pubmedqa(data)
