from datasets import load_dataset
import requests
import json
from pathlib import Path


def download_hotpotqa_distractor(save_path: Path):
    """
    Download the HotpotQA dev_distractor set from CMU official site.
    """
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    print(f"Downloading HotpotQA dev_distractor from:\n{url}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with save_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✅ Download complete: {save_path.resolve()}")

    with save_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Verified JSON file — {len(data):,} records.")
    return data


def load_hotpotqa(subset="train", max_samples=None):
    subset = subset.lower()
    if subset in ["train", "validation"]:
        print(f"Loading HotpotQA ({subset}) from Hugging Face...")
        dataset = load_dataset("hotpotqa/hotpot_qa", subset)
        data = dataset[:] if not max_samples else dataset[:max_samples]
        print(f"✅ Loaded {len(data):,} HotpotQA samples from Hugging Face.")
        return data

    elif subset == "dev_distractor":
        file_path = Path("data/hotpotqa/hotpot_dev_distractor_v1.json")
        if file_path.exists():
            print(f"Found local file: {file_path}")
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = download_hotpotqa_distractor(file_path)

        if max_samples:
            data = data[:max_samples]
        print(f"Loaded {len(data)} HotpotQA samples.")
        return data

    else:
        raise ValueError(
            f"Unsupported subset '{subset}'. "
            "Use 'train', 'validation', or 'dev_distractor'."
        )


def save_hotpotqa(data, file_path="data/hotpotqa.json"):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved dataset to {file_path.resolve()}")


if __name__ == "__main__":
    subset = "dev_distractor"
    data = load_hotpotqa(subset=subset)
    save_hotpotqa(data)
