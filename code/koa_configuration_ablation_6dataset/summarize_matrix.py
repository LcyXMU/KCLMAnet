from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
CONFIG_NAMES = [
    "default",
    "koa_p4_i2_e8",
    "koa_p2_i8_e16",
    "koa_p4_i4_e16",
    "koa_p8_i2_e16",
    "random_e16",
]
METRICS = {
    "accuracy": "raw_accuracy",
    "precision": "raw_precision",
    "recall": "raw_recall",
    "f1_score": "raw_f1_score",
}
PREPLANNED_COMPARISONS = [
    ("koa_p4_i4_e16", "default", "balanced KOA vs fixed default"),
    ("koa_p4_i4_e16", "koa_p4_i2_e8", "16 vs 8 evaluations"),
    ("koa_p4_i4_e16", "random_e16", "KOA updates vs random search"),
    ("koa_p4_i4_e16", "koa_p2_i8_e16", "balanced vs iteration-heavy"),
    ("koa_p4_i4_e16", "koa_p8_i2_e16", "balanced vs population-heavy"),
    ("koa_p2_i8_e16", "koa_p8_i2_e16", "iteration-heavy vs population-heavy"),
]
EXPLORATORY_COMPARISONS = [
    ("koa_p8_i2_e16", "default", "observed-best KOA vs fixed default"),
    ("koa_p8_i2_e16", "random_e16", "observed-best KOA vs random search"),
    ("koa_p8_i2_e16", "koa_p4_i2_e8", "observed-best KOA vs original 4x2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def exact_paired_sign_pvalue(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    permuted = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(nonzero)):
        permuted.append(abs(float(np.mean(nonzero * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    running_max = 0.0
    comparison_count = len(valid)
    for rank, (index, pvalue) in enumerate(valid.items()):
        candidate = min(1.0, float(pvalue) * (comparison_count - rank))
        running_max = max(running_max, candidate)
        adjusted.loc[index] = running_max
    return adjusted


def paired_comparison_rows(
    fold_frames: dict[str, pd.DataFrame],
    comparisons_to_run: list[tuple[str, str, str]],
) -> list[dict]:
    rows = []
    for left_name, right_name, hypothesis in comparisons_to_run:
        if left_name not in fold_frames or right_name not in fold_frames:
            continue
        left = fold_frames[left_name].set_index("fold")
        right = fold_frames[right_name].set_index("fold")
        for metric_name, column in METRICS.items():
            paired = pd.concat(
                [left[column].rename("left"), right[column].rename("right")],
                axis=1,
                join="inner",
            ).dropna()
            differences = (paired["left"] - paired["right"]).to_numpy(dtype=float)
            fold_count = len(differences)
            mean_difference = float(differences.mean()) if fold_count else np.nan
            sample_std = float(differences.std(ddof=1)) if fold_count > 1 else np.nan
            standard_error = sample_std / np.sqrt(fold_count) if fold_count > 1 else np.nan
            critical_value = (
                float(student_t.ppf(0.975, df=fold_count - 1)) if fold_count > 1 else np.nan
            )
            rows.append(
                {
                    "left_configuration": left_name,
                    "right_configuration": right_name,
                    "contrast": f"{left_name} - {right_name}",
                    "hypothesis": hypothesis,
                    "metric": metric_name,
                    "fold_count": fold_count,
                    "mean_paired_difference": mean_difference,
                    "mean_paired_difference_percentage_points": 100.0 * mean_difference,
                    "paired_difference_sample_std": sample_std,
                    "ci95_low": mean_difference - critical_value * standard_error,
                    "ci95_high": mean_difference + critical_value * standard_error,
                    "paired_effect_size_dz": (
                        mean_difference / sample_std
                        if fold_count > 1 and np.isfinite(sample_std) and sample_std > 0
                        else np.nan
                    ),
                    "exact_sign_randomization_pvalue": exact_paired_sign_pvalue(differences),
                    "wins": int((differences > 1e-12).sum()),
                    "ties": int((np.abs(differences) <= 1e-12).sum()),
                    "losses": int((differences < -1e-12).sum()),
                }
            )
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons["holm_adjusted_pvalue_within_metric"] = comparisons.groupby("metric")[
            "exact_sign_randomization_pvalue"
        ].transform(holm_adjust)
    return comparisons.to_dict("records")


def search_statistics(run_dir: Path) -> dict:
    search_frames = []
    search_seconds = 0.0
    training_seconds = 0.0
    candidate_epochs = 0
    fold_rows = []
    for fold in range(1, 7):
        fold_dir = run_dir / f"fold_{fold}"
        best_path = fold_dir / "best_hyperparameters.json"
        fold_search_seconds = 0.0
        if best_path.is_file():
            best = json.loads(best_path.read_text(encoding="utf-8"))
            fold_search_seconds = float(best.get("koa_search_seconds", 0.0))
            search_seconds += fold_search_seconds
        fold_training_seconds = 0.0
        member_path = fold_dir / "ensemble_members.csv"
        if member_path.is_file():
            fold_training_seconds = float(pd.read_csv(member_path)["training_seconds"].sum())
            training_seconds += fold_training_seconds
        search_path = fold_dir / "koa_search.csv"
        fold_candidate_epochs = 0
        fold_evaluations = 0
        if not search_path.is_file():
            fold_rows.append(
                {
                    "fold": fold,
                    "search_seconds": fold_search_seconds,
                    "training_seconds": fold_training_seconds,
                    "search_evaluations": 0,
                    "candidate_epochs": 0,
                }
            )
            continue
        frame = pd.read_csv(search_path)
        if "evaluation" not in frame:
            frame["evaluation"] = np.arange(1, len(frame) + 1)
        frame["fold"] = fold
        frame["best_val_accuracy_so_far"] = frame["val_accuracy"].cummax()
        fold_candidate_epochs = int(frame["trained_epochs"].sum())
        fold_evaluations = len(frame)
        candidate_epochs += fold_candidate_epochs
        search_frames.append(frame)
        fold_rows.append(
            {
                "fold": fold,
                "search_seconds": fold_search_seconds,
                "training_seconds": fold_training_seconds,
                "search_evaluations": fold_evaluations,
                "candidate_epochs": fold_candidate_epochs,
            }
        )
    fold_statistics = pd.DataFrame(fold_rows)
    if not search_frames:
        return {
            "best_validation_accuracy_mean": np.nan,
            "default_validation_accuracy_mean": np.nan,
            "validation_gain_over_default": np.nan,
            "anytime_best_validation_accuracy": np.nan,
            "search_seconds": search_seconds,
            "training_seconds": training_seconds,
            "candidate_epochs": candidate_epochs,
            "evaluation_to_best_mean": np.nan,
            "evaluation_to_best_median": np.nan,
            "search_seconds_per_fold_mean": float(fold_statistics["search_seconds"].mean()),
            "search_seconds_per_fold_std": float(fold_statistics["search_seconds"].std(ddof=1)),
            "training_seconds_per_fold_mean": float(fold_statistics["training_seconds"].mean()),
            "training_seconds_per_fold_std": float(fold_statistics["training_seconds"].std(ddof=1)),
            "convergence": None,
            "fold_convergence": None,
            "fold_statistics": fold_statistics,
        }
    search = pd.concat(search_frames, ignore_index=True)
    final_best = search.groupby("fold")["best_val_accuracy_so_far"].last()
    initial = search.loc[search["evaluation"] == 1].set_index("fold")["val_accuracy"]
    convergence = search.groupby("evaluation", as_index=False).agg(
        mean=("best_val_accuracy_so_far", "mean"),
        std=("best_val_accuracy_so_far", "std"),
        fold_count=("fold", "nunique"),
    )
    evaluation_to_best = search.loc[
        search["best_val_accuracy_so_far"].eq(search.groupby("fold")["best_val_accuracy_so_far"].transform("max"))
    ].groupby("fold")["evaluation"].min()
    return {
        "best_validation_accuracy_mean": float(final_best.mean()),
        "default_validation_accuracy_mean": float(initial.mean()),
        "validation_gain_over_default": float((final_best - initial).mean()),
        "anytime_best_validation_accuracy": float(convergence["mean"].mean()),
        "search_seconds": search_seconds,
        "training_seconds": training_seconds,
        "candidate_epochs": candidate_epochs,
        "evaluation_to_best_mean": float(evaluation_to_best.mean()),
        "evaluation_to_best_median": float(evaluation_to_best.median()),
        "search_seconds_per_fold_mean": float(fold_statistics["search_seconds"].mean()),
        "search_seconds_per_fold_std": float(fold_statistics["search_seconds"].std(ddof=1)),
        "training_seconds_per_fold_mean": float(fold_statistics["training_seconds"].mean()),
        "training_seconds_per_fold_std": float(fold_statistics["training_seconds"].std(ddof=1)),
        "convergence": convergence,
        "fold_convergence": search[
            ["fold", "evaluation", "val_accuracy", "best_val_accuracy_so_far"]
        ].copy(),
        "fold_statistics": fold_statistics,
    }


def pooled_test_metrics(run_dir: Path) -> dict[str, float]:
    frames = []
    for fold in range(1, 7):
        path = run_dir / f"fold_{fold}" / "test_predictions.csv"
        if path.is_file():
            frames.append(pd.read_csv(path, usecols=["y_true", "y_pred_raw"]))
    if not frames:
        return {name: np.nan for name in ("accuracy", "precision", "recall", "f1_score")}
    predictions = pd.concat(frames, ignore_index=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        predictions["y_true"],
        predictions["y_pred_raw"],
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(predictions["y_true"], predictions["y_pred_raw"])),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


def selected_hyperparameter_rows(configuration: str, run_dir: Path) -> list[dict]:
    rows = []
    for fold in range(1, 7):
        path = run_dir / f"fold_{fold}" / "best_hyperparameters.json"
        if path.is_file():
            rows.append({"configuration": configuration, "fold": fold, **json.loads(path.read_text(encoding="utf-8"))})
    return rows


def plot_convergence(curves: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    for name, frame in curves.items():
        axis.plot(
            frame["evaluation"],
            100.0 * frame["mean"],
            marker="o",
            linewidth=2.0,
            label=name,
        )
    axis.set_xlabel("Candidate evaluation")
    axis.set_ylabel("Six-fold mean cumulative-best validation accuracy (%)")
    axis.set_title("Search-budget and KOA-configuration ablation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary_rows = []
    fold_frames = {}
    curves = {}
    fold_curves = []
    fold_timing_rows = []
    hyperparameter_rows = []
    missing = []
    for name in CONFIG_NAMES:
        run_dir = BASE_DIR / "results" / name / f"run_{args.run_id}_seed{args.search_seed}"
        metrics_path = run_dir / "fold_test_metrics.csv"
        if not metrics_path.is_file():
            missing.append(name)
            continue
        metrics = pd.read_csv(metrics_path).sort_values("fold")
        fold_frames[name] = metrics
        search = search_statistics(run_dir)
        pooled = pooled_test_metrics(run_dir)
        fold_timing = search["fold_statistics"].copy()
        fold_timing.insert(0, "configuration", name)
        fold_timing_rows.append(fold_timing)
        hyperparameter_rows.extend(selected_hyperparameter_rows(name, run_dir))
        if search["convergence"] is not None:
            curves[name] = search["convergence"]
            fold_curve = search["fold_convergence"].copy()
            fold_curve.insert(0, "configuration", name)
            fold_curves.append(fold_curve)
        summary_rows.append(
            {
                "configuration": name,
                "accuracy_mean": metrics["raw_accuracy"].mean(),
                "accuracy_sample_std": metrics["raw_accuracy"].std(ddof=1),
                "precision_mean": metrics["raw_precision"].mean(),
                "precision_sample_std": metrics["raw_precision"].std(ddof=1),
                "recall_mean": metrics["raw_recall"].mean(),
                "recall_sample_std": metrics["raw_recall"].std(ddof=1),
                "f1_mean": metrics["raw_f1_score"].mean(),
                "f1_sample_std": metrics["raw_f1_score"].std(ddof=1),
                "pooled_accuracy": pooled["accuracy"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1_score": pooled["f1_score"],
                "best_validation_accuracy_mean": search["best_validation_accuracy_mean"],
                "validation_gain_over_default": search["validation_gain_over_default"],
                "anytime_best_validation_accuracy": search["anytime_best_validation_accuracy"],
                "search_seconds": search["search_seconds"],
                "training_seconds": search["training_seconds"],
                "search_seconds_per_fold_mean": search["search_seconds_per_fold_mean"],
                "search_seconds_per_fold_std": search["search_seconds_per_fold_std"],
                "training_seconds_per_fold_mean": search["training_seconds_per_fold_mean"],
                "training_seconds_per_fold_std": search["training_seconds_per_fold_std"],
                "candidate_epochs": search["candidate_epochs"],
                "evaluation_to_best_mean": search["evaluation_to_best_mean"],
                "evaluation_to_best_median": search["evaluation_to_best_median"],
            }
        )
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Missing completed configurations: {', '.join(missing)}")
    if not summary_rows:
        raise SystemExit("No completed configurations found")

    summary = pd.DataFrame(summary_rows)
    comparison_rows = []
    if "default" in fold_frames:
        default = fold_frames["default"].set_index("fold")["raw_accuracy"]
        for name, frame in fold_frames.items():
            if name == "default":
                continue
            current = frame.set_index("fold")["raw_accuracy"].reindex(default.index)
            differences = (current - default).dropna().to_numpy(dtype=float)
            effect_size = (
                float(differences.mean() / differences.std(ddof=1))
                if len(differences) > 1 and differences.std(ddof=1) > 0
                else np.nan
            )
            comparison_rows.append(
                {
                    "configuration": name,
                    "paired_accuracy_difference": float(differences.mean()),
                    "paired_effect_size_dz": effect_size,
                    "exact_sign_randomization_pvalue": exact_paired_sign_pvalue(differences),
                    "fold_count": len(differences),
                }
            )
    preplanned_rows = paired_comparison_rows(fold_frames, PREPLANNED_COMPARISONS)
    exploratory_rows = paired_comparison_rows(fold_frames, EXPLORATORY_COMPARISONS)

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"ablation_summary_{args.run_id}_seed{args.search_seed}.csv"
    comparison_path = output_dir / f"paired_vs_default_{args.run_id}_seed{args.search_seed}.csv"
    preplanned_path = output_dir / f"paired_preplanned_{args.run_id}_seed{args.search_seed}.csv"
    exploratory_path = output_dir / f"paired_exploratory_{args.run_id}_seed{args.search_seed}.csv"
    convergence_path = output_dir / f"convergence_trajectories_{args.run_id}_seed{args.search_seed}.csv"
    timing_path = output_dir / f"fold_timing_{args.run_id}_seed{args.search_seed}.csv"
    hyperparameter_path = output_dir / f"selected_hyperparameters_{args.run_id}_seed{args.search_seed}.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(preplanned_rows).to_csv(preplanned_path, index=False)
    pd.DataFrame(exploratory_rows).to_csv(exploratory_path, index=False)
    if fold_curves:
        pd.concat(fold_curves, ignore_index=True).to_csv(convergence_path, index=False)
    if fold_timing_rows:
        pd.concat(fold_timing_rows, ignore_index=True).to_csv(timing_path, index=False)
    if hyperparameter_rows:
        pd.DataFrame(hyperparameter_rows).to_csv(hyperparameter_path, index=False)
    if curves:
        plot_convergence(curves, output_dir / f"convergence_comparison_{args.run_id}_seed{args.search_seed}.png")

    lines = [
        "# KOA Configuration Ablation Results",
        "",
        "Primary classification values are the unweighted mean ± sample SD across the six held-out datasets.",
        "",
        "| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Best val. (%) | Search total (h) | Search/fold (min) | Final train total (h) | Eval. to best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        best_validation = (
            f"{100.0 * row.best_validation_accuracy_mean:.2f}"
            if np.isfinite(row.best_validation_accuracy_mean)
            else "-"
        )
        evaluation_to_best = (
            f"{row.evaluation_to_best_mean:.2f}"
            if np.isfinite(row.evaluation_to_best_mean)
            else "-"
        )
        lines.append(
            f"| {row.configuration} | {100.0 * row.accuracy_mean:.2f} ± "
            f"{100.0 * row.accuracy_sample_std:.2f} | {100.0 * row.precision_mean:.2f} ± "
            f"{100.0 * row.precision_sample_std:.2f} | {100.0 * row.recall_mean:.2f} ± "
            f"{100.0 * row.recall_sample_std:.2f} | {100.0 * row.f1_mean:.2f} ± "
            f"{100.0 * row.f1_sample_std:.2f} | {best_validation} | "
            f"{row.search_seconds / 3600.0:.2f} | "
            f"{row.search_seconds_per_fold_mean / 60.0:.2f} ± "
            f"{row.search_seconds_per_fold_std / 60.0:.2f} | "
            f"{row.training_seconds / 3600.0:.2f} | "
            f"{evaluation_to_best} |"
        )
    lines.extend(
        [
            "",
            "## Pooled predictions",
            "",
            "These secondary metrics concatenate all six held-out prediction sets; inferential comparisons remain paired by fold.",
            "",
            "| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.configuration} | {100.0 * row.pooled_accuracy:.2f} | "
            f"{100.0 * row.pooled_precision:.2f} | {100.0 * row.pooled_recall:.2f} | "
            f"{100.0 * row.pooled_f1_score:.2f} |"
        )
    if preplanned_rows:
        lines.extend(
            [
                "",
                "## Preplanned paired fold comparisons",
                "",
                "Positive differences favor the left configuration. Exact two-sided sign-randomization p-values are Holm-adjusted separately within each metric. With only six folds, inferential results should be interpreted together with effect sizes and confidence intervals.",
                "",
                "| Contrast | Metric | Difference (pp) | 95% CI (pp) | dz | W/T/L | Exact p | Holm p |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in preplanned_rows:
            lines.append(
                f"| {row['contrast']} | {row['metric']} | "
                f"{row['mean_paired_difference_percentage_points']:.3f} | "
                f"[{100.0 * row['ci95_low']:.3f}, {100.0 * row['ci95_high']:.3f}] | "
                f"{row['paired_effect_size_dz']:.3f} | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | "
                f"{row['exact_sign_randomization_pvalue']:.4f} | "
                f"{row['holm_adjusted_pvalue_within_metric']:.4f} |"
            )
    if exploratory_rows:
        lines.extend(
            [
                "",
                "## Exploratory paired fold comparisons",
                "",
                "These contrasts were added after observing that 8×2 had the highest mean test accuracy. They are descriptive and must not be presented as preregistered confirmatory tests.",
                "",
                "| Contrast | Metric | Difference (pp) | 95% CI (pp) | dz | W/T/L | Exact p | Holm p |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in exploratory_rows:
            lines.append(
                f"| {row['contrast']} | {row['metric']} | "
                f"{row['mean_paired_difference_percentage_points']:.3f} | "
                f"[{100.0 * row['ci95_low']:.3f}, {100.0 * row['ci95_high']:.3f}] | "
                f"{row['paired_effect_size_dz']:.3f} | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | "
                f"{row['exact_sign_randomization_pvalue']:.4f} | "
                f"{row['holm_adjusted_pvalue_within_metric']:.4f} |"
            )
    if missing:
        lines.extend(["", f"Incomplete configurations: {', '.join(missing)}."])
    (output_dir / f"ABLATION_RESULTS_{args.run_id}_seed{args.search_seed}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
