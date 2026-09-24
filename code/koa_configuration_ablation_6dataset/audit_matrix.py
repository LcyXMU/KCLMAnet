from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
CONFIGURATIONS = {
    "default": 0,
    "koa_p4_i2_e8": 8,
    "koa_p2_i8_e16": 16,
    "koa_p4_i4_e16": 16,
    "koa_p8_i2_e16": 16,
    "random_e16": 16,
}
SEARCH_COLUMNS = [
    "num_heads",
    "d_model",
    "dropout_rate",
    "filters",
    "kernel_size",
    "learning_rate_log10",
    "weight_decay_log10",
    "val_accuracy",
    "trained_epochs",
]
METRIC_COLUMNS = [
    "raw_accuracy",
    "raw_precision",
    "raw_recall",
    "raw_f1_score",
    "causal_accuracy",
    "causal_precision",
    "causal_recall",
    "causal_f1_score",
]
SUMMARY_METRICS = {
    "accuracy": "raw_accuracy",
    "precision": "raw_precision",
    "recall": "raw_recall",
    "f1": "raw_f1_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--search-seed", type=int, default=0)
    return parser.parse_args()


def run_dir(args: argparse.Namespace, configuration: str) -> Path:
    return BASE_DIR / "results" / configuration / f"run_{args.run_id}_seed{args.search_seed}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_search(args: argparse.Namespace, configuration: str, fold: int) -> pd.DataFrame:
    path = run_dir(args, configuration) / f"fold_{fold}" / "koa_search.csv"
    require(path.is_file(), f"missing search log: {path}")
    return pd.read_csv(path).sort_values("evaluation").reset_index(drop=True)


def audit_completed_runs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    metrics = {}
    reference_units = None
    reference_member_seeds = None
    for configuration, expected_evaluations in CONFIGURATIONS.items():
        directory = run_dir(args, configuration)
        require((directory / "summary_metrics.json").is_file(), f"incomplete configuration: {configuration}")
        metrics_path = directory / "fold_test_metrics.csv"
        require(metrics_path.is_file(), f"missing fold metrics: {configuration}")
        frame = pd.read_csv(metrics_path).sort_values("fold").reset_index(drop=True)
        require(frame["fold"].tolist() == list(range(1, 7)), f"invalid folds: {configuration}")
        require(np.isfinite(frame[METRIC_COLUMNS].to_numpy(dtype=float)).all(), f"nonfinite metrics: {configuration}")
        units = frame[["fold", "test_key", "sample_count"]]
        if reference_units is None:
            reference_units = units
        else:
            require(units.equals(reference_units), f"test units differ: {configuration}")

        configuration_member_seeds = []
        for fold in range(1, 7):
            fold_dir = directory / f"fold_{fold}"
            require((fold_dir / "fold_result.json").is_file(), f"missing fold result: {configuration} fold {fold}")
            predictions = pd.read_csv(fold_dir / "test_predictions.csv")
            require(len(predictions) == int(frame.loc[frame["fold"].eq(fold), "sample_count"].iloc[0]), f"prediction count mismatch: {configuration} fold {fold}")
            fold_metrics = frame.loc[frame["fold"].eq(fold)].iloc[0]
            for prediction_column, prefix in (("y_pred_raw", "raw"), ("y_pred_causal", "causal")):
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    predictions["y_true"], predictions[prediction_column], average="weighted", zero_division=0
                )
                recomputed = {
                    "accuracy": accuracy_score(predictions["y_true"], predictions[prediction_column]),
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                }
                for metric, value in recomputed.items():
                    require(
                        np.isclose(float(fold_metrics[f"{prefix}_{metric}"]), float(value)),
                        f"prediction metric mismatch: {configuration} fold {fold} {prefix}_{metric}",
                    )
            members = pd.read_csv(fold_dir / "ensemble_members.csv")
            require(len(members) == 2, f"expected two members: {configuration} fold {fold}")
            require((members["training_seconds"] > 0).all(), f"invalid training time: {configuration} fold {fold}")
            configuration_member_seeds.append(tuple(members["seed"].astype(int)))
            if expected_evaluations:
                search = load_search(args, configuration, fold)
                require(len(search) == expected_evaluations, f"wrong search budget: {configuration} fold {fold}")
                require(search["evaluation"].tolist() == list(range(1, expected_evaluations + 1)), f"nonsequential evaluations: {configuration} fold {fold}")
                require(search["candidate_training_seed"].nunique() == 1, f"candidate seeds vary: {configuration} fold {fold}")
                best = json.loads((fold_dir / "best_hyperparameters.json").read_text(encoding="utf-8"))
                require(float(best["koa_search_seconds"]) > 0, f"invalid search time: {configuration} fold {fold}")
                require(
                    np.isclose(float(best["koa_validation_accuracy"]), float(search["val_accuracy"].max())),
                    f"saved best validation accuracy differs from search log: {configuration} fold {fold}",
                )
            else:
                require(not (fold_dir / "koa_search.csv").exists(), f"default unexpectedly searched: fold {fold}")
        if reference_member_seeds is None:
            reference_member_seeds = configuration_member_seeds
        else:
            require(configuration_member_seeds == reference_member_seeds, f"final member seeds differ: {configuration}")
        metrics[configuration] = frame
    return metrics


def audit_aggregate_values(args: argparse.Namespace, metrics: dict[str, pd.DataFrame]) -> None:
    suffix = f"{args.run_id}_seed{args.search_seed}"
    summary = pd.read_csv(BASE_DIR / "results" / f"ablation_summary_{suffix}.csv").set_index("configuration")
    timings = pd.read_csv(BASE_DIR / "results" / f"fold_timing_{suffix}.csv")
    selected = pd.read_csv(BASE_DIR / "results" / f"selected_hyperparameters_{suffix}.csv")
    convergence = pd.read_csv(BASE_DIR / "results" / f"convergence_trajectories_{suffix}.csv")
    expected_convergence_rows = 0
    for configuration, expected_evaluations in CONFIGURATIONS.items():
        frame = metrics[configuration]
        for output_name, source_column in SUMMARY_METRICS.items():
            values = frame[source_column].to_numpy(dtype=float)
            require(np.isclose(summary.loc[configuration, f"{output_name}_mean"], values.mean()), f"summary mean mismatch: {configuration} {output_name}")
            require(np.isclose(summary.loc[configuration, f"{output_name}_sample_std"], values.std(ddof=1)), f"summary std mismatch: {configuration} {output_name}")

        prediction_frames = [
            pd.read_csv(run_dir(args, configuration) / f"fold_{fold}" / "test_predictions.csv", usecols=["y_true", "y_pred_raw"])
            for fold in range(1, 7)
        ]
        pooled = pd.concat(prediction_frames, ignore_index=True)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            pooled["y_true"], pooled["y_pred_raw"], average="weighted", zero_division=0
        )
        pooled_values = {
            "pooled_accuracy": accuracy_score(pooled["y_true"], pooled["y_pred_raw"]),
            "pooled_precision": precision,
            "pooled_recall": recall,
            "pooled_f1_score": f1_score,
        }
        for column, value in pooled_values.items():
            require(np.isclose(summary.loc[configuration, column], value), f"pooled metric mismatch: {configuration} {column}")

        configuration_timings = timings.loc[timings["configuration"].eq(configuration)].sort_values("fold")
        require(configuration_timings["fold"].tolist() == list(range(1, 7)), f"fold timing rows mismatch: {configuration}")
        require(np.isclose(summary.loc[configuration, "search_seconds"], configuration_timings["search_seconds"].sum()), f"search time mismatch: {configuration}")
        require(np.isclose(summary.loc[configuration, "training_seconds"], configuration_timings["training_seconds"].sum()), f"training time mismatch: {configuration}")
        require(int(summary.loc[configuration, "candidate_epochs"]) == int(configuration_timings["candidate_epochs"].sum()), f"candidate epoch mismatch: {configuration}")
        require(configuration_timings["search_evaluations"].eq(expected_evaluations).all(), f"timing search budget mismatch: {configuration}")

        configuration_selected = selected.loc[selected["configuration"].eq(configuration)].sort_values("fold")
        require(configuration_selected["fold"].tolist() == list(range(1, 7)), f"selected hyperparameter rows mismatch: {configuration}")
        if expected_evaluations:
            configuration_convergence = convergence.loc[convergence["configuration"].eq(configuration)]
            require(len(configuration_convergence) == 6 * expected_evaluations, f"convergence row count mismatch: {configuration}")
            for fold in range(1, 7):
                fold_curve = configuration_convergence.loc[configuration_convergence["fold"].eq(fold)].sort_values("evaluation")
                require(fold_curve["evaluation"].tolist() == list(range(1, expected_evaluations + 1)), f"convergence evaluations mismatch: {configuration} fold {fold}")
                expected_best = fold_curve["val_accuracy"].cummax().to_numpy(dtype=float)
                require(np.allclose(fold_curve["best_val_accuracy_so_far"], expected_best), f"convergence cumulative best mismatch: {configuration} fold {fold}")
            expected_convergence_rows += 6 * expected_evaluations
    require(len(convergence) == expected_convergence_rows, "total convergence row count mismatch")


def audit_shared_candidates(args: argparse.Namespace) -> None:
    for fold in range(1, 7):
        search_4x2 = load_search(args, "koa_p4_i2_e8", fold)
        search_4x4 = load_search(args, "koa_p4_i4_e16", fold)
        require(
            search_4x2[SEARCH_COLUMNS].equals(search_4x4.iloc[:8][SEARCH_COLUMNS].reset_index(drop=True)),
            f"4x2 and 4x4 do not share their first eight evaluations: fold {fold}",
        )
        search_2x8 = load_search(args, "koa_p2_i8_e16", fold)
        require(
            search_2x8.iloc[:2][SEARCH_COLUMNS].reset_index(drop=True).equals(
                search_4x4.iloc[:2][SEARCH_COLUMNS].reset_index(drop=True)
            ),
            f"KOA configurations do not share initial two candidates: fold {fold}",
        )
        search_8x2 = load_search(args, "koa_p8_i2_e16", fold)
        random_search = load_search(args, "random_e16", fold)
        require(
            search_8x2.iloc[:8][SEARCH_COLUMNS].reset_index(drop=True).equals(
                random_search.iloc[:8][SEARCH_COLUMNS].reset_index(drop=True)
            ),
            f"population-heavy KOA and random search do not share initial eight candidates: fold {fold}",
        )


def audit_summary_artifacts(args: argparse.Namespace) -> None:
    suffix = f"{args.run_id}_seed{args.search_seed}"
    required = [
        f"ablation_summary_{suffix}.csv",
        f"paired_vs_default_{suffix}.csv",
        f"paired_preplanned_{suffix}.csv",
        f"paired_exploratory_{suffix}.csv",
        f"convergence_trajectories_{suffix}.csv",
        f"convergence_comparison_{suffix}.png",
        f"convergence_comparison_{suffix}.svg",
        f"ABLATION_RESULTS_{suffix}.md",
        f"FINAL_REPORT_CN_{suffix}.md",
        f"fold_timing_{suffix}.csv",
        f"selected_hyperparameters_{suffix}.csv",
    ]
    for name in required:
        path = BASE_DIR / "results" / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty summary artifact: {path}")
    summary = pd.read_csv(BASE_DIR / "results" / f"ablation_summary_{suffix}.csv")
    require(set(summary["configuration"]) == set(CONFIGURATIONS), "summary does not contain all configurations")
    comparisons = pd.read_csv(BASE_DIR / "results" / f"paired_preplanned_{suffix}.csv")
    require(len(comparisons) == 6 * 4, "expected six contrasts for four metrics")
    require(comparisons["fold_count"].eq(6).all(), "paired comparisons do not use all six folds")
    exploratory = pd.read_csv(BASE_DIR / "results" / f"paired_exploratory_{suffix}.csv")
    require(len(exploratory) == 3 * 4, "expected three exploratory contrasts for four metrics")
    require(exploratory["fold_count"].eq(6).all(), "exploratory comparisons do not use all six folds")


def main() -> None:
    args = parse_args()
    metrics = audit_completed_runs(args)
    audit_shared_candidates(args)
    audit_summary_artifacts(args)
    audit_aggregate_values(args, metrics)
    result = {
        "status": "PASS",
        "run_id": args.run_id,
        "search_seed": args.search_seed,
        "configurations": list(CONFIGURATIONS),
        "folds_per_configuration": 6,
        "test_sample_counts": metrics["default"]["sample_count"].astype(int).tolist(),
        "shared_candidate_checks": ["4x2_vs_4x4_first8", "2x8_vs_4x4_first2", "8x2_vs_random_first8"],
        "independent_recomputations": [
            "fold_metrics_from_predictions",
            "six_fold_mean_and_sample_std",
            "pooled_metrics",
            "search_and_training_times",
            "candidate_epochs",
            "convergence_cumulative_best",
        ],
    }
    output = BASE_DIR / "results" / f"AUDIT_{args.run_id}_seed{args.search_seed}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
