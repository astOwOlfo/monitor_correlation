import os

import fire
from datasets import Dataset

from src import utils
from src.data import DATASET_REGISTRY, HINT_REGISTRY, base_dataset_name

"""
Data processing scripts.

Data is filtered offline for difficulty levels; filtered datasets provided under results/data folder.
`download` method will download the base dataset from source; see src/data for full dataset details.

"""


def hinted_dataset_name(
    base_dataset_fpath: str,
    hint: str | None = None,
    suffix: str | None = None,
) -> str:
    """Return the output path for a dataset with a hint applied."""
    path_base = base_dataset_fpath.removesuffix(".jsonl")
    hint_name = f"_{hint}" if hint is not None else ""
    suffix_prefix = f"_{suffix}" if suffix else ""
    return f"{path_base}{suffix_prefix}{hint_name}.jsonl"


def download(
    dataset_name: str = "leetcode",
    split: str = "train",
    overwrite: bool = False,
    suffix: str | None = None,
    **kwargs,
):
    """Download the original base dataset without filtering."""
    dataset = DATASET_REGISTRY[dataset_name]().load_dataset_from_source(split, **kwargs)
    fpath = base_dataset_name(dataset=dataset_name, split=split, suffix=suffix)

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")

    utils.save_dataset_jsonl(fpath, dataset)
    print(f"Saved base dataset to {fpath}")


def hint(
    dataset_path: str,
    hint: str,
    overwrite: bool = False,
    suffix: str | None = None,
):
    """Apply one registered hint to a given dataset."""
    assert hint in HINT_REGISTRY, f"Unknown hint '{hint}'. Available: {list(HINT_REGISTRY)}"
    fpath = hinted_dataset_name(dataset_path, hint=hint, suffix=suffix)
    print(f"Creating dataset at {fpath}")

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")

    dataset = Dataset.from_list(utils.read_jsonl_all(dataset_path))
    dataset = dataset.map(lambda x: HINT_REGISTRY[hint]()(x))
    utils.save_dataset_jsonl(fpath, dataset)
    utils.read_jsonl_all(fpath)
    print(f"Saved hinted dataset to {fpath}")


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        "download": download,
        "hint": hint,
    })
