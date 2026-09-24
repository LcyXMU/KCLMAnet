from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    LeakyReLU,
    Reshape,
    SpatialDropout2D,
    TimeDistributed,
)
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from config import (
    BASE_DIR,
    BATCH_SIZE,
    CAUSAL_SMOOTHING_WIDTHS,
    CONVOLUTIONAL_STEPS,
    DATASETS_DIR,
    DEFAULT_HYPERPARAMETERS,
    ENSEMBLE_SEEDS,
    EVAL_STRIDE,
    FINAL_EPOCHS,
    FINAL_PATIENCE,
    KOA_ITERATIONS,
    KOA_PLANETS,
    MODEL_NAME,
    OBJECTIVE_EPOCHS,
    OBJECTIVE_PATIENCE,
    PARAMETER_BOUNDS,
    RESULTS_DIR,
    SENSOR_AXES,
    SENSOR_LOCATIONS,
    SURFACE_TYPE_LABELS,
    TRAIN_LABEL_PURITY,
    TRAIN_STRIDE,
    WINDOW_ROWS,
)


OPTIMIZED_CODE_DIR = BASE_DIR.parent / "training_utils"


def load_training_utilities():
    local_config = sys.modules.get("config")
    config_spec = importlib.util.spec_from_file_location(
        "six_sensor_base_config",
        OPTIMIZED_CODE_DIR / "config.py",
    )
    base_config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(base_config)
    training_spec = importlib.util.spec_from_file_location(
        "six_sensor_training_utils",
        OPTIMIZED_CODE_DIR / "train_optimized.py",
    )
    module = importlib.util.module_from_spec(training_spec)
    sys.modules[training_spec.name] = module
    sys.modules["config"] = base_config
    try:
        training_spec.loader.exec_module(module)
    finally:
        if local_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = local_config
    return module


training_utils = load_training_utilities()
ORIGINAL_SANITIZE_HYPERPARAMETERS = training_utils.sanitize_hyperparameters


VARIANTS = {
    "full": {"attention": True, "koa": True, "name": MODEL_NAME},
    "cnn_lstm": {"attention": False, "koa": False, "name": "Six-sensor CNN-LSTM"},
    "cnn_lstm_attention": {
        "attention": True,
        "koa": False,
        "name": "Six-sensor CNN-LSTM with Multi-Head Attention",
    },
    "cnn_lstm_koa": {"attention": False, "koa": True, "name": "Six-sensor KOA-CNN-LSTM"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--ensemble-seeds", type=int, default=ENSEMBLE_SEEDS)
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS)
    parser.add_argument("--run-tag", default="six_sensor")
    parser.add_argument("--resume-dir", type=Path)
    return parser.parse_args()


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def enable_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def get_dataset_keys() -> List[str]:
    return [f"xmu_{index}" for index in range(1, 7)]


def build_standardized_splits() -> List[Dict[str, object]]:
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


def build_segment_catalog(datasets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, int]]:
    catalog = {}
    for dataset_key, dataset in datasets.items():
        label_indices = dataset["labels"].argmax(axis=1)
        boundaries = np.concatenate(
            ([0], np.flatnonzero(np.diff(label_indices) != 0) + 1, [len(label_indices)])
        )
        for run_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < WINDOW_ROWS:
                continue
            group_id = f"{dataset_key}_run_{run_index:02d}"
            catalog[group_id] = {
                "dataset_key": dataset_key,
                "start": int(start),
                "end": int(end),
                "label": int(label_indices[start]),
                "eval_windows": int((end - start) // WINDOW_ROWS),
            }
    return catalog


def build_segment_group_splits(catalog: Dict[str, Dict[str, int]]) -> List[Dict[str, object]]:
    fold_count = len(get_dataset_keys())
    fold_groups = [[] for _ in range(fold_count)]
    fold_class_load = np.zeros((fold_count, len(SURFACE_TYPE_LABELS)), dtype=int)
    fold_total_load = np.zeros(fold_count, dtype=int)
    for label_index in range(len(SURFACE_TYPE_LABELS)):
        groups = [
            (group_id, values)
            for group_id, values in catalog.items()
            if values["label"] == label_index
        ]
        groups.sort(key=lambda item: (-item[1]["eval_windows"], item[0]))
        for group_id, values in groups:
            target = min(
                range(fold_count),
                key=lambda fold: (fold_class_load[fold, label_index], fold_total_load[fold], fold),
            )
            fold_groups[target].append(group_id)
            fold_class_load[target, label_index] += values["eval_windows"]
            fold_total_load[target] += values["eval_windows"]
    all_groups = set(catalog)
    splits = []
    for fold in range(fold_count):
        test_groups = sorted(fold_groups[fold])
        val_groups = sorted(fold_groups[(fold + 1) % fold_count])
        train_groups = sorted(all_groups - set(test_groups) - set(val_groups))
        splits.append(
            {
                "fold": fold + 1,
                "unit": "segment_group",
                "train": train_groups,
                "val": val_groups,
                "test": test_groups,
            }
        )
    return splits


def sensor_columns(position: str) -> List[str]:
    return [f"{axis}_{position}" for axis in SENSOR_AXES]


def load_datasets() -> Dict[str, Dict[str, np.ndarray]]:
    datasets = {}
    for dataset_key in get_dataset_keys():
        index = int(dataset_key.rsplit("_", maxsplit=1)[1])
        dataset_dir = DATASETS_DIR / f"XMU {index}"
        side_frames = {
            side: pd.read_csv(dataset_dir / f"dataset_mpu_{side}.csv", float_precision="high")
            for side in ("left", "right")
        }
        left_timestamps = side_frames["left"]["timestamp"].to_numpy(dtype=np.float64)
        right_timestamps = side_frames["right"]["timestamp"].to_numpy(dtype=np.float64)
        if len(left_timestamps) != len(right_timestamps) or not np.allclose(
            left_timestamps,
            right_timestamps,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"XMU {index} left/right sensor streams are not synchronized")
        sensor_values = []
        for _, side, position in SENSOR_LOCATIONS:
            sensor_values.append(side_frames[side][sensor_columns(position)].to_numpy(dtype=np.float32))
        values = np.stack(sensor_values, axis=1)
        labels = pd.read_csv(
            dataset_dir / "dataset_labels.csv",
            usecols=SURFACE_TYPE_LABELS,
            float_precision="high",
        )[SURFACE_TYPE_LABELS].to_numpy(dtype=np.float32)
        if len(values) != len(labels):
            raise ValueError(f"XMU {index} sensor and label row counts differ")
        datasets[dataset_key] = {"values": values, "labels": labels}
    return datasets


@dataclass
class TrainOnlyScaler:
    lower: np.ndarray
    upper: np.ndarray
    scaler: StandardScaler

    def transform(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        clipped = np.clip(flat, self.lower, self.upper)
        transformed = self.scaler.transform(clipped).astype(np.float32)
        return transformed.reshape(values.shape)


def fit_train_only_scaler(
    datasets: Dict[str, Dict[str, np.ndarray]],
    train_keys: Sequence[str],
) -> TrainOnlyScaler:
    values = np.concatenate([datasets[key]["values"].reshape(len(datasets[key]["values"]), -1) for key in train_keys])
    lower = np.quantile(values, 0.005, axis=0)
    upper = np.quantile(values, 0.995, axis=0)
    clipped = np.clip(values, lower, upper)
    return TrainOnlyScaler(lower=lower, upper=upper, scaler=StandardScaler().fit(clipped))


def fit_segment_group_scaler(
    datasets: Dict[str, Dict[str, np.ndarray]],
    catalog: Dict[str, Dict[str, int]],
    group_ids: Sequence[str],
) -> TrainOnlyScaler:
    values = []
    for group_id in group_ids:
        group = catalog[group_id]
        segment = datasets[group["dataset_key"]]["values"][group["start"] : group["end"]]
        values.append(segment.reshape(len(segment), -1))
    combined = np.concatenate(values)
    lower = np.quantile(combined, 0.005, axis=0)
    upper = np.quantile(combined, 0.995, axis=0)
    return TrainOnlyScaler(
        lower=lower,
        upper=upper,
        scaler=StandardScaler().fit(np.clip(combined, lower, upper)),
    )


def window_starts(row_count: int, stride: int) -> Iterable[int]:
    return range(0, row_count - WINDOW_ROWS + 1, stride)


def build_split_arrays(
    datasets: Dict[str, Dict[str, np.ndarray]],
    split_keys: Sequence[str],
    scaler: TrainOnlyScaler,
    stride: int,
    minimum_label_purity: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    macro_steps = WINDOW_ROWS // CONVOLUTIONAL_STEPS
    input_shape = (macro_steps, len(SENSOR_LOCATIONS), CONVOLUTIONAL_STEPS, len(SENSOR_AXES))
    all_inputs = []
    all_outputs = []
    metadata = []
    for dataset_key in split_keys:
        labels = datasets[dataset_key]["labels"]
        values = scaler.transform(datasets[dataset_key]["values"])
        for window_index, start in enumerate(window_starts(len(values), stride)):
            end = start + WINDOW_ROWS
            label_distribution = labels[start:end].mean(axis=0)
            label_index = int(label_distribution.argmax())
            purity = float(label_distribution[label_index])
            if purity < minimum_label_purity:
                continue
            window = values[start:end].reshape(
                macro_steps,
                CONVOLUTIONAL_STEPS,
                len(SENSOR_LOCATIONS),
                len(SENSOR_AXES),
            )
            all_inputs.append(np.transpose(window, (0, 2, 1, 3)))
            one_hot = np.zeros(len(SURFACE_TYPE_LABELS), dtype=np.float32)
            one_hot[label_index] = 1.0
            all_outputs.append(one_hot)
            metadata.append(
                {
                    "dataset_key": dataset_key,
                    "window_index": window_index,
                    "start_row": start,
                    "end_row": end - 1,
                    "label_purity": purity,
                }
            )
    return np.asarray(all_inputs, dtype=np.float32).reshape((-1, *input_shape)), np.asarray(all_outputs), pd.DataFrame(metadata)


def build_segment_group_arrays(
    datasets: Dict[str, Dict[str, np.ndarray]],
    catalog: Dict[str, Dict[str, int]],
    group_ids: Sequence[str],
    scaler: TrainOnlyScaler,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    macro_steps = WINDOW_ROWS // CONVOLUTIONAL_STEPS
    input_shape = (macro_steps, len(SENSOR_LOCATIONS), CONVOLUTIONAL_STEPS, len(SENSOR_AXES))
    all_inputs, all_outputs, metadata = [], [], []
    for group_id in group_ids:
        group = catalog[group_id]
        dataset = datasets[group["dataset_key"]]
        values = scaler.transform(dataset["values"][group["start"] : group["end"]])
        for window_index, local_start in enumerate(window_starts(len(values), stride)):
            local_end = local_start + WINDOW_ROWS
            window = values[local_start:local_end].reshape(
                macro_steps,
                CONVOLUTIONAL_STEPS,
                len(SENSOR_LOCATIONS),
                len(SENSOR_AXES),
            )
            all_inputs.append(np.transpose(window, (0, 2, 1, 3)))
            one_hot = np.zeros(len(SURFACE_TYPE_LABELS), dtype=np.float32)
            one_hot[group["label"]] = 1.0
            all_outputs.append(one_hot)
            metadata.append(
                {
                    "dataset_key": group["dataset_key"],
                    "group_id": group_id,
                    "window_index": window_index,
                    "start_row": group["start"] + local_start,
                    "end_row": group["start"] + local_end - 1,
                    "label_purity": 1.0,
                }
            )
    return np.asarray(all_inputs, dtype=np.float32).reshape((-1, *input_shape)), np.asarray(all_outputs), pd.DataFrame(metadata)


class SixSensorAugmentation(Layer):
    def __init__(self, noise_std: float = 0.015, gain_range: float = 0.08):
        super().__init__()
        self.noise_std = noise_std
        self.gain_range = gain_range

    def call(self, inputs, training=None):
        if not training:
            return inputs
        shape = tf.shape(inputs)
        gain = tf.random.uniform(
            (shape[0], 1, shape[2], 1, shape[4]),
            minval=1.0 - self.gain_range,
            maxval=1.0 + self.gain_range,
        )
        return inputs * gain + tf.random.normal(shape, stddev=self.noise_std)


class SixSensorStatistics(Layer):
    def call(self, inputs):
        values = tf.transpose(inputs, (0, 2, 1, 3, 4))
        values = tf.reshape(values, (tf.shape(values)[0], tf.shape(values)[1], -1, tf.shape(values)[-1]))
        mean = tf.reduce_mean(values, axis=2)
        centered = values - mean[:, :, None, :]
        standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(centered), axis=2) + 1e-6)
        root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(values), axis=2) + 1e-6)
        peak = tf.reduce_max(tf.abs(values), axis=2)
        statistics = tf.concat([mean, standard_deviation, root_mean_square, peak], axis=-1)
        return tf.reshape(
            statistics,
            (tf.shape(statistics)[0], len(SENSOR_LOCATIONS) * len(SENSOR_AXES) * 4),
        )


class LocalAveragePooling(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=3)


def build_model(hyperparameters: Dict[str, float], use_attention: bool) -> Model:
    num_heads = int(hyperparameters["num_heads"])
    d_model = int(hyperparameters["d_model"])
    filters = int(hyperparameters["filters"])
    kernel_size = int(hyperparameters["kernel_size"])
    dropout_rate = float(hyperparameters["dropout_rate"])
    input_shape = (
        WINDOW_ROWS // CONVOLUTIONAL_STEPS,
        len(SENSOR_LOCATIONS),
        CONVOLUTIONAL_STEPS,
        len(SENSOR_AXES),
    )

    inputs = Input(shape=input_shape)
    x = SixSensorAugmentation()(inputs)
    x = TimeDistributed(
        Conv2D(filters, (1, kernel_size), padding="same", use_bias=False)
    )(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = TimeDistributed(SpatialDropout2D(dropout_rate))(x)
    shortcut = x
    x = TimeDistributed(
        Conv2D(filters, (1, kernel_size), padding="same", use_bias=False)
    )(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = TimeDistributed(
        Conv2D(filters, (1, kernel_size), padding="same", use_bias=False)
    )(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([x, shortcut])
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = LocalAveragePooling()(x)
    x = Reshape((input_shape[0], len(SENSOR_LOCATIONS) * filters))(x)
    x = Dense(d_model, activation="elu")(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(d_model // 2, return_sequences=True, dropout=dropout_rate * 0.5))(x)
    x = LayerNormalization()(x)
    sequence = LSTM(d_model, return_sequences=True, dropout=dropout_rate * 0.5)(x)
    sequence = LayerNormalization()(sequence)
    if use_attention:
        attention = training_utils.MultiHeadAttention(num_heads=num_heads, d_model=d_model)(
            sequence,
            sequence,
            sequence,
        )
        attention = Dropout(dropout_rate)(attention)
        sequence = Add()([sequence, attention])
    x = LayerNormalization()(sequence)
    x = LSTM(d_model // 2, dropout=dropout_rate * 0.5)(x)

    statistics = SixSensorStatistics()(inputs)
    statistics = Dense(128, activation="elu")(statistics)
    statistics = LayerNormalization()(statistics)
    statistics = Dropout(dropout_rate)(statistics)
    x = Concatenate()([x, statistics])
    x = Dense(d_model, activation="elu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(len(SURFACE_TYPE_LABELS), activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=10.0 ** float(hyperparameters["learning_rate_log10"]),
            weight_decay=10.0 ** float(hyperparameters["weight_decay_log10"]),
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"],
    )
    return model


def sanitize_without_attention(values: Dict[str, float]) -> Dict[str, float]:
    cleaned = dict(values)
    cleaned["num_heads"] = 1
    for key in ("d_model", "filters", "kernel_size"):
        low, high = training_utils.PARAMETER_BOUNDS[key]
        cleaned[key] = int(np.clip(round(cleaned[key]), low, high))
    for key in ("dropout_rate", "learning_rate_log10", "weight_decay_log10"):
        low, high = training_utils.PARAMETER_BOUNDS[key]
        cleaned[key] = float(np.clip(cleaned[key], low, high))
    cleaned["d_model"] = int(round(cleaned["d_model"] / 2) * 2)
    cleaned["d_model"] = int(
        np.clip(
            cleaned["d_model"],
            training_utils.PARAMETER_BOUNDS["d_model"][0],
            training_utils.PARAMETER_BOUNDS["d_model"][1],
        )
    )
    return cleaned


def configure_training_utilities(use_attention: bool) -> None:
    training_utils.BATCH_SIZE = BATCH_SIZE
    training_utils.FINAL_PATIENCE = FINAL_PATIENCE
    training_utils.OBJECTIVE_EPOCHS = OBJECTIVE_EPOCHS
    training_utils.OBJECTIVE_PATIENCE = OBJECTIVE_PATIENCE
    training_utils.KOA_PLANETS = KOA_PLANETS
    training_utils.KOA_ITERATIONS = KOA_ITERATIONS
    training_utils.SURFACE_TYPE_LABELS = SURFACE_TYPE_LABELS
    training_utils.DEFAULT_HYPERPARAMETERS = dict(DEFAULT_HYPERPARAMETERS)
    training_utils.PARAMETER_BOUNDS = dict(PARAMETER_BOUNDS)
    training_utils.sanitize_hyperparameters = ORIGINAL_SANITIZE_HYPERPARAMETERS
    if not use_attention:
        training_utils.PARAMETER_BOUNDS.pop("num_heads")
        training_utils.DEFAULT_HYPERPARAMETERS["num_heads"] = 1
        training_utils.sanitize_hyperparameters = sanitize_without_attention
    training_utils.build_model = lambda hyperparameters: build_model(hyperparameters, use_attention)


def causal_smooth(frame: pd.DataFrame, probabilities: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return probabilities.copy()
    smoothed = np.empty_like(probabilities)
    for _, indices in frame.groupby("dataset_key", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        smoothed[positions] = (
            pd.DataFrame(probabilities[positions]).rolling(width, min_periods=1).mean().to_numpy()
        )
    return smoothed


def choose_causal_width(
    frame: pd.DataFrame,
    outputs: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[int, pd.DataFrame]:
    y_true = outputs.argmax(axis=1)
    rows = []
    for width in CAUSAL_SMOOTHING_WIDTHS:
        predictions = causal_smooth(frame, probabilities, width).argmax(axis=1)
        rows.append({"width": width, "validation_accuracy": float(np.mean(y_true == predictions))})
    scores = pd.DataFrame(rows)
    best = scores["validation_accuracy"].max()
    selected = int(scores.loc[scores["validation_accuracy"].eq(best), "width"].min())
    return selected, scores


@dataclass
class FoldResult:
    fold: int
    test_key: str
    sample_count: int
    raw_metrics: Dict[str, float]
    causal_metrics: Dict[str, float]
    causal_width: int


def run_fold(
    split: Dict[str, object],
    datasets: Dict[str, Dict[str, np.ndarray]],
    run_dir: Path,
    args: argparse.Namespace,
    use_koa: bool,
    segment_catalog: Optional[Dict[str, Dict[str, int]]] = None,
) -> FoldResult:
    fold = int(split["fold"])
    fold_seed = 20260802 + fold * 100
    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Fold {fold} ===", flush=True)
    print(f"train={split['train']} val={split['val']} test={split['test']}", flush=True)

    if split.get("unit") == "segment_group":
        scaler = fit_segment_group_scaler(datasets, segment_catalog, split["train"])
        input_train, output_train, meta_train = build_segment_group_arrays(
            datasets, segment_catalog, split["train"], scaler, TRAIN_STRIDE
        )
        input_val, output_val, meta_val = build_segment_group_arrays(
            datasets, segment_catalog, split["val"], scaler, EVAL_STRIDE
        )
        input_test, output_test, meta_test = build_segment_group_arrays(
            datasets, segment_catalog, split["test"], scaler, EVAL_STRIDE
        )
    else:
        scaler = fit_train_only_scaler(datasets, split["train"])
        input_train, output_train, meta_train = build_split_arrays(
            datasets, split["train"], scaler, TRAIN_STRIDE, TRAIN_LABEL_PURITY
        )
        input_val, output_val, meta_val = build_split_arrays(
            datasets, split["val"], scaler, EVAL_STRIDE, 0.0
        )
        input_test, output_test, meta_test = build_split_arrays(
            datasets, split["test"], scaler, EVAL_STRIDE, 0.0
        )
    print(
        f"samples train={len(input_train)} val={len(input_val)} test={len(input_test)} shape={input_train.shape[1:]}",
        flush=True,
    )
    meta_train.to_csv(fold_dir / "train_metadata.csv", index=False)
    meta_val.to_csv(fold_dir / "val_metadata.csv", index=False)
    meta_test.to_csv(fold_dir / "test_metadata.csv", index=False)

    hyperparameters_path = fold_dir / "best_hyperparameters.json"
    if hyperparameters_path.is_file():
        hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    elif use_koa:
        search_started = time.perf_counter()
        hyperparameters = training_utils.optimize_with_koa(
            input_train,
            output_train,
            input_val,
            output_val,
            fold_seed,
            fold_dir,
        )
        hyperparameters["koa_search_seconds"] = time.perf_counter() - search_started
        save_json(hyperparameters_path, hyperparameters)
    else:
        hyperparameters = training_utils.sanitize_hyperparameters(DEFAULT_HYPERPARAMETERS)
        hyperparameters["koa_validation_accuracy"] = None
        hyperparameters["koa_search_seconds"] = 0.0
        save_json(hyperparameters_path, hyperparameters)
    print(f"hyperparameters={hyperparameters}", flush=True)

    validation_probabilities = []
    test_probabilities = []
    member_rows = []
    for member_index in range(1, args.ensemble_seeds + 1):
        member_result_path = fold_dir / f"member_{member_index}_result.json"
        member_validation_path = fold_dir / f"member_{member_index}_validation_probabilities.npy"
        member_test_path = fold_dir / f"member_{member_index}_test_probabilities.npy"
        if member_result_path.is_file() and member_validation_path.is_file() and member_test_path.is_file():
            member_rows.append(json.loads(member_result_path.read_text(encoding="utf-8")))
            validation_probabilities.append(np.load(member_validation_path))
            test_probabilities.append(np.load(member_test_path))
            continue
        member_seed = fold_seed + member_index
        train_started = time.perf_counter()
        model, history, weights_path = training_utils.train_ensemble_member(
            hyperparameters,
            input_train,
            output_train,
            input_val,
            output_val,
            fold_dir,
            member_seed,
            member_index,
            args.final_epochs,
        )
        training_seconds = time.perf_counter() - train_started
        member_validation_probabilities = model.predict(input_val, batch_size=BATCH_SIZE, verbose=0)
        member_test_probabilities = model.predict(input_test, batch_size=BATCH_SIZE, verbose=0)
        np.save(member_validation_path, member_validation_probabilities)
        np.save(member_test_path, member_test_probabilities)
        member_result = {
            "member": member_index,
            "seed": member_seed,
            "best_epoch": int(np.argmax(history["val_accuracy"]) + 1),
            "best_val_accuracy": float(max(history["val_accuracy"])),
            "training_seconds": training_seconds,
            "weights": str(weights_path),
        }
        save_json(member_result_path, member_result)
        member_rows.append(member_result)
        validation_probabilities.append(member_validation_probabilities)
        test_probabilities.append(member_test_probabilities)
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    probabilities_val = np.mean(validation_probabilities, axis=0)
    probabilities_test = np.mean(test_probabilities, axis=0)
    causal_width, causal_scores = choose_causal_width(meta_val, output_val, probabilities_val)
    causal_scores.to_csv(fold_dir / "validation_causal_smoothing_selection.csv", index=False)
    causal_test_probabilities = causal_smooth(meta_test, probabilities_test, causal_width)
    y_true = output_test.argmax(axis=1)
    y_raw = probabilities_test.argmax(axis=1)
    y_causal = causal_test_probabilities.argmax(axis=1)
    raw_metrics = training_utils.compute_metrics(y_true, y_raw)
    causal_metrics = training_utils.compute_metrics(y_true, y_causal)

    prediction_frame = meta_test.copy()
    prediction_frame["y_true"] = y_true
    prediction_frame["y_pred_raw"] = y_raw
    prediction_frame["y_pred_causal"] = y_causal
    for index, label in enumerate(SURFACE_TYPE_LABELS):
        prediction_frame[f"prob_raw_{label}"] = probabilities_test[:, index]
        prediction_frame[f"prob_causal_{label}"] = causal_test_probabilities[:, index]
    prediction_frame.to_csv(fold_dir / "test_predictions.csv", index=False)
    matrix_raw = confusion_matrix(y_true, y_raw, labels=range(len(SURFACE_TYPE_LABELS)))
    matrix_causal = confusion_matrix(y_true, y_causal, labels=range(len(SURFACE_TYPE_LABELS)))
    pd.DataFrame(matrix_raw, index=SURFACE_TYPE_LABELS, columns=SURFACE_TYPE_LABELS).to_csv(
        fold_dir / "confusion_matrix_raw.csv"
    )
    pd.DataFrame(matrix_causal, index=SURFACE_TYPE_LABELS, columns=SURFACE_TYPE_LABELS).to_csv(
        fold_dir / "confusion_matrix_causal.csv"
    )
    pd.DataFrame(member_rows).to_csv(fold_dir / "ensemble_members.csv", index=False)
    payload = {
        "fold": fold,
        "train_keys": split["train"],
        "val_keys": split["val"],
        "test_keys": split["test"],
        "hyperparameters": hyperparameters,
        "ensemble_members": member_rows,
        "causal_smoothing_width_selected_on_validation": causal_width,
        "metrics_test_raw": raw_metrics,
        "metrics_test_causal": causal_metrics,
        "test_sample_count": len(y_true),
    }
    save_json(fold_dir / "fold_result.json", payload)
    print(
        f"fold {fold} raw_accuracy={raw_metrics['accuracy']:.4f} "
        f"causal_accuracy={causal_metrics['accuracy']:.4f} causal_width={causal_width}",
        flush=True,
    )
    return FoldResult(
        fold=fold,
        test_key="segment_groups" if split.get("unit") == "segment_group" else split["test"][0],
        sample_count=len(y_true),
        raw_metrics=raw_metrics,
        causal_metrics=causal_metrics,
        causal_width=causal_width,
    )


def load_completed_fold(fold_dir: Path) -> FoldResult:
    payload = json.loads((fold_dir / "fold_result.json").read_text(encoding="utf-8"))
    return FoldResult(
        fold=int(payload["fold"]),
        test_key=payload["test_keys"][0],
        sample_count=int(payload["test_sample_count"]),
        raw_metrics=payload["metrics_test_raw"],
        causal_metrics=payload["metrics_test_causal"],
        causal_width=int(payload["causal_smoothing_width_selected_on_validation"]),
    )


def summarize_results(results: List[FoldResult], run_dir: Path) -> None:
    rows = []
    for result in sorted(results, key=lambda item: item.fold):
        rows.append(
            {
                "fold": result.fold,
                "test_key": result.test_key,
                "sample_count": result.sample_count,
                "causal_smoothing_width": result.causal_width,
                **{f"raw_{key}": value for key, value in result.raw_metrics.items()},
                **{f"causal_{key}": value for key, value in result.causal_metrics.items()},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "fold_test_metrics.csv", index=False)
    summary = {}
    for prefix in ("raw", "causal"):
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            values = frame[f"{prefix}_{metric}"].to_numpy(dtype=float)
            summary[f"{prefix}_{metric}"] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    save_json(run_dir / "summary_metrics.json", summary)


def main() -> None:
    args = parse_args()
    variant = VARIANTS[args.variant]
    configure_training_utilities(variant["attention"])
    enable_gpu_memory_growth()
    if args.resume_dir is None:
        run_dir = RESULTS_DIR / args.variant / datetime.now().strftime(f"run_%Y%m%d_%H%M%S_{args.run_tag}")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.resume_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    print("loading synchronized six-sensor datasets...", flush=True)
    datasets = load_datasets()
    segment_catalog = None
    splits = build_standardized_splits()
    selected = [split for split in splits if split["fold"] in set(args.folds)]
    run_config = {
        "model_name": variant["name"],
        "variant": args.variant,
        "attention_enabled": variant["attention"],
        "koa_enabled": variant["koa"],
        "datasets_dir": str(DATASETS_DIR),
        "sensor_locations": [name for name, _, _ in SENSOR_LOCATIONS],
        "sensor_axes": SENSOR_AXES,
        "input_shape": [
            WINDOW_ROWS // CONVOLUTIONAL_STEPS,
            len(SENSOR_LOCATIONS),
            CONVOLUTIONAL_STEPS,
            len(SENSOR_AXES),
        ],
        "window_rows": WINDOW_ROWS,
        "train_stride": TRAIN_STRIDE,
        "eval_stride": EVAL_STRIDE,
        "batch_size": BATCH_SIZE,
        "ensemble_seeds": args.ensemble_seeds,
        "final_epochs": args.final_epochs,
        "final_patience": FINAL_PATIENCE,
        "koa_planets": KOA_PLANETS,
        "koa_iterations": KOA_ITERATIONS,
        "objective_epochs": OBJECTIVE_EPOCHS,
        "parameter_bounds": training_utils.PARAMETER_BOUNDS,
        "default_hyperparameters": training_utils.DEFAULT_HYPERPARAMETERS,
        "primary_ablation_metric": "raw window-level ensemble prediction without smoothing",
        "causal_smoothing_widths": CAUSAL_SMOOTHING_WIDTHS,
        "selected_folds": args.folds,
    }
    if not (run_dir / "run_config.json").is_file():
        save_json(run_dir / "run_config.json", run_config)
    if not (run_dir / "standardized_splits.json").is_file():
        save_json(run_dir / "standardized_splits.json", {"splits": splits})
    print(f"results={run_dir}", flush=True)
    results = []
    for split in selected:
        fold_dir = run_dir / f"fold_{split['fold']}"
        if (fold_dir / "fold_result.json").is_file():
            print(f"reusing completed fold {split['fold']}", flush=True)
            results.append(load_completed_fold(fold_dir))
        else:
            results.append(
                run_fold(split, datasets, run_dir, args, variant["koa"], segment_catalog)
            )
    summarize_results(results, run_dir)
    print(f"finished. results saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
