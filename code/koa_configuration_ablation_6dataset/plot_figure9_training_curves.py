from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "koa_p8_i2_e16"
    / "run_20260809_koa_config_ablation_deterministic_v2_seed0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--all-epochs", action="store_true")
    return parser.parse_args()


def load_histories(run_dir: Path) -> list[pd.DataFrame]:
    history_paths = sorted(run_dir.glob("fold_*/member_*_history.csv"))
    if not history_paths:
        raise FileNotFoundError(f"No member histories found under {run_dir}")

    histories = []
    for path in history_paths:
        frame = pd.read_csv(path)
        frame["epoch"] = frame["epoch"].astype(int) + 1
        frame["run"] = f"{path.parent.name}/{path.stem.removesuffix('_history')}"
        histories.append(frame)
    return histories


def epoch_summary(histories: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    combined = pd.concat(
        [history[["epoch", metric]].assign(run=history["run"]) for history in histories],
        ignore_index=True,
    )
    return (
        combined.groupby("epoch")[metric]
        .agg(
            median="median",
            lower_quartile=lambda values: values.quantile(0.25),
            upper_quartile=lambda values: values.quantile(0.75),
            run_count="count",
        )
        .reset_index()
    )


def plot_metric(
    axis: plt.Axes,
    histories: list[pd.DataFrame],
    train_metric: str,
    validation_metric: str,
    ylabel: str,
) -> None:
    colors = {train_metric: "#1f77b4", validation_metric: "#d95f02"}
    labels = {train_metric: "Training", validation_metric: "Validation"}

    for history in histories:
        for metric in (train_metric, validation_metric):
            axis.plot(
                history["epoch"],
                history[metric],
                color=colors[metric],
                linewidth=0.7,
                alpha=0.14,
            )

    for metric in (train_metric, validation_metric):
        summary = epoch_summary(histories, metric)
        epochs = summary["epoch"].to_numpy()
        lower = summary["lower_quartile"].to_numpy()
        upper = summary["upper_quartile"].to_numpy()
        median = summary["median"].to_numpy()
        axis.fill_between(epochs, lower, upper, color=colors[metric], alpha=0.16)
        axis.plot(
            epochs,
            median,
            color=colors[metric],
            linewidth=2.0,
            linestyle="-" if metric == train_metric else "--",
            label=f"{labels[metric]} median (IQR)",
        )

    axis.set_xlabel("Epoch", fontsize=15)
    axis.set_ylabel(ylabel, fontsize=15)
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    axis.legend(frameon=False, loc="best", fontsize=12)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir.parent.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = load_histories(run_dir)
    common_epoch_count = min(int(history["epoch"].max()) for history in histories)
    plotted_histories = (
        histories
        if args.all_epochs
        else [history.loc[history["epoch"] <= common_epoch_count].copy() for history in histories]
    )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    plot_metric(axes[0], plotted_histories, "loss", "val_loss", "Loss")
    plot_metric(
        axes[1],
        plotted_histories,
        "accuracy",
        "val_accuracy",
        "Accuracy",
    )
    axes[1].set_ylim(0.75, 1.01)

    stem = output_dir / "figure9_koa8x2_sixfold_training_curves"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(f"Plotted {len(histories)} histories from {run_dir}")
    if not args.all_epochs:
        print(f"Restricted aggregation to the {common_epoch_count} epochs shared by all histories")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
