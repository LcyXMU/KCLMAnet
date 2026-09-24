from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
SIX_DATASET_ENTRY = BASE_DIR.parent / "main_method_6sensor_6dataset" / "train.py"
DEFAULT_RUN_DIR = (
    BASE_DIR
    / "results"
    / "koa_p8_i2_e16"
    / "run_20260809_koa_config_ablation_deterministic_v2_seed0"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "figure12_koa8x2_attention"
CLASS_NAMES = ["Dirt road", "Cobblestone road", "Asphalt road", "Bluestone road", "Concrete road"]
PANEL_NAMES = ["b", "c", "d", "e", "f"]


def load_six_dataset_module():
    spec = importlib.util.spec_from_file_location("figure12_six_dataset", SIX_DATASET_ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIX_DATASET = load_six_dataset_module()
TRAINING = SIX_DATASET.training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def find_attention_layer(model: tf.keras.Model):
    attention_class = TRAINING.training_utils.MultiHeadAttention
    layers = [layer for layer in model.layers if isinstance(layer, attention_class)]
    if len(layers) != 1:
        raise RuntimeError(f"Expected one attention layer, found {len(layers)}")
    return layers[0]


def build_sequence_model(model: tf.keras.Model, attention_layer) -> tf.keras.Model:
    input_tensors = attention_layer._inbound_nodes[0].input_tensors
    sequence_tensor = input_tensors[0] if isinstance(input_tensors, (list, tuple)) else input_tensors
    return tf.keras.Model(inputs=model.inputs, outputs=sequence_tensor)


def attention_weights(attention_layer, sequence: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(sequence)[0]
    query = attention_layer.split_heads(attention_layer.wq(sequence), batch_size)
    key = attention_layer.split_heads(attention_layer.wk(sequence), batch_size)
    logits = tf.matmul(query, key, transpose_b=True)
    logits /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
    return tf.nn.softmax(logits, axis=-1)


def predict_mean_head_attention(
    model: tf.keras.Model,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    attention_layer = find_attention_layer(model)
    sequence_model = build_sequence_model(model, attention_layer)
    matrices = []
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        sequence = sequence_model(inputs[start:stop], training=False)
        weights = attention_weights(attention_layer, sequence)
        matrices.append(tf.reduce_mean(weights, axis=1).numpy())
    return np.concatenate(matrices, axis=0)


def assert_metadata_matches(actual: pd.DataFrame, expected_path: Path) -> None:
    expected = pd.read_csv(expected_path)
    columns = ["dataset_key", "window_index", "start_row", "end_row"]
    if len(actual) != len(expected) or not actual[columns].reset_index(drop=True).equals(
        expected[columns].reset_index(drop=True)
    ):
        raise RuntimeError(f"Reconstructed test metadata does not match {expected_path}")


def extract_class_attention(run_dir: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    training = TRAINING
    six_dataset = SIX_DATASET
    training.enable_gpu_memory_growth()
    datasets = six_dataset.load_six_datasets()
    splits = {int(split["fold"]): split for split in six_dataset.build_six_dataset_splits()}
    class_sums = np.zeros((len(CLASS_NAMES), 24, 24), dtype=np.float64)
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for fold in range(1, 7):
        fold_dir = run_dir / f"fold_{fold}"
        split = splits[fold]
        scaler = training.fit_train_only_scaler(datasets, split["train"])
        inputs, outputs, metadata = training.build_split_arrays(
            datasets,
            split["test"],
            scaler,
            training.EVAL_STRIDE,
            0.0,
        )
        assert_metadata_matches(metadata, fold_dir / "test_metadata.csv")
        hyperparameters = json.loads((fold_dir / "best_hyperparameters.json").read_text(encoding="utf-8"))
        member_average = np.zeros((len(inputs), 24, 24), dtype=np.float64)

        for member in (1, 2):
            model = training.build_model(hyperparameters, True)
            model.load_weights(fold_dir / f"member_{member}_best.weights.h5")
            member_average += predict_mean_head_attention(model, inputs, batch_size) / 2.0
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        labels = outputs.argmax(axis=1)
        for class_index in range(len(CLASS_NAMES)):
            selected = member_average[labels == class_index]
            class_sums[class_index] += selected.sum(axis=0)
            class_counts[class_index] += len(selected)
        print(f"fold={fold} test_samples={len(inputs)}", flush=True)

        del inputs, outputs, metadata, member_average, scaler
        gc.collect()

    if np.any(class_counts == 0):
        raise RuntimeError(f"Missing test samples for classes: {class_counts.tolist()}")
    return class_sums / class_counts[:, None, None], class_counts


def save_summary_data(output_dir: Path, matrices: np.ndarray, counts: np.ndarray) -> None:
    np.savez_compressed(
        output_dir / "figure12_attention_summary.npz",
        class_attention_matrices=matrices,
        class_sample_counts=counts,
        class_names=np.asarray(CLASS_NAMES),
    )
    curve_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        pd.DataFrame(matrices[class_index]).to_csv(
            output_dir / f"class_{class_index + 1}_{class_name.replace(' ', '_').lower()}_attention_matrix.csv",
            index=False,
        )
        for time_step, value in enumerate(matrices[class_index].mean(axis=0), start=1):
            curve_rows.append(
                {
                    "class": class_name,
                    "time_step": time_step,
                    "mean_attention_weight": float(value),
                    "test_sample_count": int(counts[class_index]),
                }
            )
    pd.DataFrame(curve_rows).to_csv(output_dir / "class_temporal_attention_profiles.csv", index=False)


def load_or_extract(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    summary_path = args.output_dir / "figure12_attention_summary.npz"
    if summary_path.is_file() and not args.recompute:
        payload = np.load(summary_path)
        return payload["class_attention_matrices"], payload["class_sample_counts"]
    matrices, counts = extract_class_attention(args.run_dir.resolve(), args.batch_size)
    save_summary_data(args.output_dir, matrices, counts)
    return matrices, counts


def style_curve_axis(axis: plt.Axes, matrices: np.ndarray) -> None:
    time_steps = np.arange(1, matrices.shape[-1] + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for class_name, matrix, color in zip(CLASS_NAMES, matrices, colors):
        axis.plot(time_steps, matrix.mean(axis=0), color=color, linewidth=1.8, label=class_name)
    axis.set_xlabel("Time step", fontsize=13)
    axis.set_ylabel("Mean attention weight", fontsize=13)
    axis.set_xlim(1, matrices.shape[-1])
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    axis.legend(frameon=False, fontsize=9, ncol=2, loc="best")


def style_heatmap_axis(axis: plt.Axes, matrix: np.ndarray, vmin: float, vmax: float):
    image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ticks = np.arange(0, matrix.shape[0], 4)
    axis.set_xticks(ticks, ticks + 1)
    axis.set_yticks(ticks, ticks + 1)
    axis.set_xlabel("Key time step", fontsize=12)
    axis.set_ylabel("Query time step", fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    return image


def plot_combined(output_dir: Path, matrices: np.ndarray) -> None:
    vmin = float(matrices.min())
    vmax = float(matrices.max())
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.5), constrained_layout=True)
    style_curve_axis(axes[0, 0], matrices)
    axes[0, 0].set_title("(a) Temporal attention profiles", fontsize=13)
    heatmap_axes = []
    image = None
    for class_index, axis in enumerate(axes.flat[1:]):
        image = style_heatmap_axis(axis, matrices[class_index], vmin, vmax)
        axis.set_title(f"({PANEL_NAMES[class_index]}) {CLASS_NAMES[class_index]}", fontsize=13)
        heatmap_axes.append(axis)
    colorbar = figure.colorbar(image, ax=heatmap_axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Attention weight", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    save_figure(figure, output_dir / "figure12_combined")


def plot_individual_panels(output_dir: Path, matrices: np.ndarray) -> None:
    vmin = float(matrices.min())
    vmax = float(matrices.max())

    figure, axis = plt.subplots(figsize=(6.3, 4.3), constrained_layout=True)
    style_curve_axis(axis, matrices)
    save_figure(figure, output_dir / "figure12_a_temporal_attention_profiles")

    for class_index, class_name in enumerate(CLASS_NAMES):
        figure, axis = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
        image = style_heatmap_axis(axis, matrices[class_index], vmin, vmax)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.03)
        colorbar.set_label("Attention weight", fontsize=12)
        colorbar.ax.tick_params(labelsize=10)
        safe_name = class_name.replace(" ", "_").lower()
        save_figure(figure, output_dir / f"figure12_{PANEL_NAMES[class_index]}_{safe_name}_attention_map")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
        }
    )
    matrices, counts = load_or_extract(args)
    plot_combined(args.output_dir, matrices)
    plot_individual_panels(args.output_dir, matrices)
    print(f"class_sample_counts={dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
