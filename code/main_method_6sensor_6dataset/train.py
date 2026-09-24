from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SIX_SENSOR_MODEL_DIR = BASE_DIR.parent / "six_sensor_model"


def load_six_sensor_training_module():
    sys.path.insert(0, str(SIX_SENSOR_MODEL_DIR))
    spec = importlib.util.spec_from_file_location(
        "six_dataset_training_base",
        SIX_SENSOR_MODEL_DIR / "train.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = load_six_sensor_training_module()
original_load_datasets = training.load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=training.VARIANTS, default="full")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--ensemble-seeds", type=int, default=training.ENSEMBLE_SEEDS)
    parser.add_argument("--final-epochs", type=int, default=training.FINAL_EPOCHS)
    parser.add_argument("--run-tag", default="six_dataset_leave_one_out")
    parser.add_argument("--resume-dir", type=Path)
    return parser.parse_args()


def get_dataset_keys():
    return [f"xmu_{index}" for index in range(1, 7)]


def build_six_dataset_splits():
    keys = get_dataset_keys()
    splits = []
    for index, test_key in enumerate(keys):
        validation_key = keys[(index + 1) % len(keys)]
        splits.append(
            {
                "fold": index + 1,
                "train": [key for key in keys if key not in {test_key, validation_key}],
                "val": [validation_key],
                "test": [test_key],
            }
        )
    return splits


def load_six_datasets():
    return {
        key: values
        for key, values in original_load_datasets().items()
        if key in set(get_dataset_keys())
    }


def main() -> None:
    training.RESULTS_DIR = BASE_DIR / "results"
    training.parse_args = parse_args
    training.get_dataset_keys = get_dataset_keys
    training.build_standardized_splits = build_six_dataset_splits
    training.load_datasets = load_six_datasets
    training.main()


if __name__ == "__main__":
    main()
