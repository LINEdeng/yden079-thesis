import sys
from pathlib import Path
from load_hotpotqa import load_hotpotqa
from load_msmarco import load_msmarco
from load_pubmedqa import load_pubmedqa


def load_dataset(name: str, subset: str = "train", max_samples: int = None):
    """
    Unified dataset loader interface.

    Args:
        name (str): dataset name, one of ['hotpotqa', 'msmarco', 'pubmedqa']
        subset (str): dataset split, e.g. 'train', 'validation', 'test'
        max_samples (int): optional limit on number of samples

    Returns:
        list[dict]: loaded dataset samples
    """
    name = name.lower()
    print(f"Loading dataset: {name} (subset={subset})")

    if name == "hotpotqa":
        return load_hotpotqa(subset=subset, max_samples=max_samples)
    elif name == "msmarco":
        return load_msmarco(subset=subset, max_samples=max_samples)
    elif name == "pubmedqa":
        return load_pubmedqa(subset=subset, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataloder.py <dataset_name> [subset] [max_samples]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    subset = sys.argv[2] if len(sys.argv) > 2 else "train"
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None

    data = load_dataset(dataset_name, subset=subset, max_samples=max_samples)
    print(f"Loaded {len(data):,} samples from {dataset_name}.")