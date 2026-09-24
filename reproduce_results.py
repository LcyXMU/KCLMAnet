from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts" / "koa8x2_sixfold"
OUTPUT_DIR = PACKAGE_DIR / "reproduced_results"
RESULT_MANIFEST = PACKAGE_DIR / "manifests" / "results_sha256.txt"
DATA_MANIFEST = PACKAGE_DIR / "manifests" / "dataset_sha256.txt"
LABELS = [
    "dirt_road",
    "cobblestone_road",
    "asphalt_road",
    "bluestone_road",
    "concrete_road",
]
METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the published KOA 8x2 metrics from saved six-fold predictions."
    )
    parser.add_argument(
        "--verify-data-sha256",
        action="store_true",
        help="Also hash all XMU1-XMU6 files. This is slower and is not needed to recompute metrics.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> int:
    require(path.is_file(), f"Missing checksum manifest: {path}")
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split("  ", maxsplit=1)
        target = PACKAGE_DIR / relative_path
        require(target.is_file(), f"Missing manifest file: {target}")
        actual = sha256(target)
        require(actual == expected, f"Checksum mismatch: {relative_path}")
        checked += 1
    return checked


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


def verify_dataset_layout() -> list[dict[str, object]]:
    inventory = []
    required_names = ("dataset_labels.csv", "dataset_mpu_left.csv", "dataset_mpu_right.csv")
    for index in range(1, 7):
        dataset_dir = PACKAGE_DIR / "data" / f"XMU {index}"
        require(dataset_dir.is_dir(), f"Missing dataset directory: {dataset_dir}")
        for name in required_names:
            path = dataset_dir / name
            require(path.is_file() and path.stat().st_size > 0, f"Missing or empty dataset file: {path}")
            inventory.append(
                {
                    "dataset": f"XMU {index}",
                    "file": name,
                    "bytes": path.stat().st_size,
                }
            )
    return inventory


def main() -> None:
    args = parse_args()
    require(ARTIFACT_DIR.is_dir(), f"Missing result artifacts: {ARTIFACT_DIR}")
    verified_result_files = verify_manifest(RESULT_MANIFEST)
    dataset_inventory = verify_dataset_layout()
    verified_data_files = verify_manifest(DATA_MANIFEST) if args.verify_data_sha256 else 0

    run_config = json.loads((ARTIFACT_DIR / "run_config.json").read_text(encoding="utf-8"))
    require(run_config["koa_planets"] == 8, "The saved run is not KOA 8x2: planets != 8")
    require(run_config["koa_iterations"] == 2, "The saved run is not KOA 8x2: iterations != 2")
    require(run_config["selected_folds"] == [1, 2, 3, 4, 5, 6], "Unexpected fold selection")

    original_fold_metrics = pd.read_csv(ARTIFACT_DIR / "fold_test_metrics.csv").sort_values("fold")
    original_summary = json.loads((ARTIFACT_DIR / "summary_metrics.json").read_text(encoding="utf-8"))
    rows = []
    pooled_frames = []
    aggregate_raw_confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    aggregate_causal_confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for fold in range(1, 7):
        fold_dir = ARTIFACT_DIR / f"fold_{fold}"
        predictions = pd.read_csv(fold_dir / "test_predictions.csv")
        require(len(predictions) > 0, f"Fold {fold} has no predictions")
        require(
            predictions["dataset_key"].nunique() == 1
            and predictions["dataset_key"].iloc[0] == f"xmu_{fold}",
            f"Fold {fold} does not test XMU {fold}",
        )
        raw_metrics = compute_metrics(predictions["y_true"], predictions["y_pred_raw"])
        causal_metrics = compute_metrics(predictions["y_true"], predictions["y_pred_causal"])
        saved_row = original_fold_metrics.loc[original_fold_metrics["fold"].eq(fold)].iloc[0]
        for prefix, metrics in (("raw", raw_metrics), ("causal", causal_metrics)):
            for metric, value in metrics.items():
                require(
                    np.isclose(value, float(saved_row[f"{prefix}_{metric}"]), rtol=0.0, atol=1e-12),
                    f"Fold {fold} mismatch for {prefix}_{metric}",
                )
        rows.append(
            {
                "fold": fold,
                "test_key": f"xmu_{fold}",
                "sample_count": len(predictions),
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{f"causal_{key}": value for key, value in causal_metrics.items()},
            }
        )
        pooled_frames.append(predictions[["y_true", "y_pred_raw", "y_pred_causal"]])
        aggregate_raw_confusion += confusion_matrix(
            predictions["y_true"], predictions["y_pred_raw"], labels=range(len(LABELS))
        )
        aggregate_causal_confusion += confusion_matrix(
            predictions["y_true"], predictions["y_pred_causal"], labels=range(len(LABELS))
        )

    fold_metrics = pd.DataFrame(rows)
    summary = {}
    for prefix in ("raw", "causal"):
        for metric in METRIC_NAMES:
            values = fold_metrics[f"{prefix}_{metric}"].to_numpy(dtype=float)
            key = f"{prefix}_{metric}"
            summary[key] = {
                "mean": float(values.mean()),
                "population_std": float(values.std(ddof=0)),
                "sample_std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            require(
                np.isclose(summary[key]["mean"], original_summary[key]["mean"], rtol=0.0, atol=1e-12),
                f"Six-fold mean mismatch for {key}",
            )
            require(
                np.isclose(
                    summary[key]["population_std"],
                    original_summary[key]["std"],
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"Six-fold population standard deviation mismatch for {key}",
            )

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled_metrics = {
        "raw": compute_metrics(pooled["y_true"], pooled["y_pred_raw"]),
        "causal": compute_metrics(pooled["y_true"], pooled["y_pred_causal"]),
        "sample_count": int(len(pooled)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics_reproduced.csv", index=False)
    pd.DataFrame(aggregate_raw_confusion, index=LABELS, columns=LABELS).to_csv(
        OUTPUT_DIR / "aggregate_confusion_matrix_raw.csv"
    )
    pd.DataFrame(aggregate_causal_confusion, index=LABELS, columns=LABELS).to_csv(
        OUTPUT_DIR / "aggregate_confusion_matrix_causal.csv"
    )
    pd.DataFrame(dataset_inventory).to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)
    (OUTPUT_DIR / "summary_metrics_reproduced.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "pooled_metrics_reproduced.json").write_text(
        json.dumps(pooled_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verification = {
        "status": "PASS",
        "mode": "saved-prediction metric recomputation; no retraining",
        "model": run_config["model_name"],
        "koa_planets": run_config["koa_planets"],
        "koa_iterations": run_config["koa_iterations"],
        "folds": run_config["selected_folds"],
        "evaluated_datasets": [f"XMU {index}" for index in range(1, 7)],
        "verified_result_files": verified_result_files,
        "verified_data_files": verified_data_files,
        "all_saved_metrics_matched": True,
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("KOA 8x2 saved-result reproduction: PASS")
    print("Mode: recomputed from saved test predictions; model training was not run")
    print("Six-fold raw metrics (mean +/- sample SD):")
    for metric in METRIC_NAMES:
        values = summary[f"raw_{metric}"]
        print(f"  {metric:9s}: {100 * values['mean']:.4f}% +/- {100 * values['sample_std']:.4f}%")
    print(f"Pooled test samples: {pooled_metrics['sample_count']}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
