from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import (
    BackupAndRestore,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    LeakyReLU,
    Reshape,
    SpatialDropout1D,
    TimeDistributed,
)
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from config import (
    BASE_DIR,
    BATCH_SIZE,
    CONVOLUTIONAL_STEPS,
    DATASETS_DIR,
    DEFAULT_HYPERPARAMETERS,
    ENSEMBLE_SEEDS,
    EVAL_STRIDE,
    FIELDS,
    FINAL_EPOCHS,
    FINAL_PATIENCE,
    KOA_ITERATIONS,
    KOA_PLANETS,
    MODEL_NAME,
    OBJECTIVE_EPOCHS,
    OBJECTIVE_PATIENCE,
    PARAMETER_BOUNDS,
    RESULTS_DIR,
    SIDES,
    SMOOTHING_WIDTHS,
    SURFACE_TYPE_LABELS,
    TRAIN_LABEL_PURITY,
    TRAIN_STRIDE,
    WINDOW_ROWS,
)


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--skip-koa", action="store_true")
    parser.add_argument("--ensemble-seeds", type=int, default=ENSEMBLE_SEEDS)
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS)
    parser.add_argument("--run-tag", default="optimized")
    parser.add_argument("--resume-dir", type=Path)
    return parser.parse_args()


def enable_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def get_dataset_keys() -> List[str]:
    return [f"xmu_{index}" for index in range(1, 7)]


def build_standardized_splits() -> List[Dict[str, List[str]]]:
    keys = get_dataset_keys()
    splits = []
    for index, test_key in enumerate(keys):
        validation_key = keys[(index + 1) % len(keys)]
        train_keys = [key for key in keys if key not in {test_key, validation_key}]
        splits.append(
            {
                "fold": index + 1,
                "train": train_keys,
                "val": [validation_key],
                "test": [test_key],
            }
        )
    return splits


def load_datasets() -> Dict[str, Dict[str, pd.DataFrame]]:
    datasets = {}
    for index in range(1, 7):
        dataset_dir = DATASETS_DIR / f"XMU {index}"
        left = pd.read_csv(dataset_dir / "dataset_mpu_left.csv", usecols=FIELDS, float_precision="high")
        right = pd.read_csv(dataset_dir / "dataset_mpu_right.csv", usecols=FIELDS, float_precision="high")
        labels = pd.read_csv(
            dataset_dir / "dataset_labels.csv",
            usecols=SURFACE_TYPE_LABELS,
            float_precision="high",
        )
        datasets[f"xmu_{index}"] = {
            "left": left[FIELDS],
            "right": right[FIELDS],
            "labels": labels[SURFACE_TYPE_LABELS],
        }
    return datasets


@dataclass
class TrainOnlyScaler:
    lower: np.ndarray
    upper: np.ndarray
    scaler: StandardScaler

    def transform(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lower, self.upper)
        return self.scaler.transform(clipped).astype(np.float32)


def fit_train_only_scaler(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    train_keys: Sequence[str],
) -> TrainOnlyScaler:
    values = np.concatenate(
        [datasets[key][side].to_numpy(dtype=np.float32) for key in train_keys for side in SIDES],
        axis=0,
    )
    lower = np.quantile(values, 0.005, axis=0)
    upper = np.quantile(values, 0.995, axis=0)
    clipped = np.clip(values, lower, upper)
    scaler = StandardScaler().fit(clipped)
    return TrainOnlyScaler(lower=lower, upper=upper, scaler=scaler)


def window_starts(row_count: int, stride: int) -> Iterable[int]:
    return range(0, row_count - WINDOW_ROWS + 1, stride)


def build_split_arrays(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    split_keys: Sequence[str],
    scaler: TrainOnlyScaler,
    stride: int,
    minimum_label_purity: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    input_shape = (WINDOW_ROWS // CONVOLUTIONAL_STEPS, CONVOLUTIONAL_STEPS, len(FIELDS))
    all_inputs: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    for dataset_key in split_keys:
        labels = datasets[dataset_key]["labels"].to_numpy(dtype=np.float32)
        for side in SIDES:
            values = scaler.transform(datasets[dataset_key][side].to_numpy(dtype=np.float32))
            for window_index, start in enumerate(window_starts(len(values), stride)):
                end = start + WINDOW_ROWS
                label_distribution = labels[start:end].mean(axis=0)
                label_index = int(label_distribution.argmax())
                purity = float(label_distribution[label_index])
                if purity < minimum_label_purity:
                    continue
                all_inputs.append(values[start:end].reshape(input_shape))
                one_hot = np.zeros(len(SURFACE_TYPE_LABELS), dtype=np.float32)
                one_hot[label_index] = 1.0
                all_outputs.append(one_hot)
                metadata.append(
                    {
                        "dataset_key": dataset_key,
                        "side": side,
                        "window_index": window_index,
                        "start_row": start,
                        "end_row": end - 1,
                        "label_purity": purity,
                    }
                )

    return np.asarray(all_inputs, dtype=np.float32), np.asarray(all_outputs, dtype=np.float32), pd.DataFrame(metadata)


class SensorAugmentation(Layer):
    def __init__(self, noise_std: float = 0.015, gain_range: float = 0.08):
        super().__init__()
        self.noise_std = noise_std
        self.gain_range = gain_range

    def call(self, inputs, training=None):
        if not training:
            return inputs
        batch_size = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1]
        gain = tf.random.uniform(
            (batch_size, 1, 1, channels),
            minval=1.0 - self.gain_range,
            maxval=1.0 + self.gain_range,
        )
        noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_std)
        return inputs * gain + noise


class WindowStatistics(Layer):
    def call(self, inputs):
        flattened = tf.reshape(inputs, (tf.shape(inputs)[0], -1, tf.shape(inputs)[-1]))
        mean = tf.reduce_mean(flattened, axis=1)
        centered = flattened - mean[:, None, :]
        standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(centered), axis=1) + 1e-6)
        root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(flattened), axis=1) + 1e-6)
        peak = tf.reduce_max(tf.abs(flattened), axis=1)
        return tf.concat([mean, standard_deviation, root_mean_square, peak], axis=-1)


class MultiHeadAttention(Layer):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.output_projection = Dense(d_model)

    def split_heads(self, values, batch_size):
        values = tf.reshape(values, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(values, perm=[0, 2, 1, 3])

    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)
        logits = tf.matmul(query, key, transpose_b=True)
        logits /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        weights = tf.nn.softmax(logits, axis=-1)
        attention = tf.matmul(weights, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))
        return self.output_projection(attention)


def build_model(hyperparameters: Dict[str, float]) -> Model:
    num_heads = int(hyperparameters["num_heads"])
    d_model = int(hyperparameters["d_model"])
    filters = int(hyperparameters["filters"])
    kernel_size = int(hyperparameters["kernel_size"])
    dropout_rate = float(hyperparameters["dropout_rate"])
    learning_rate = 10.0 ** float(hyperparameters["learning_rate_log10"])
    weight_decay = 10.0 ** float(hyperparameters["weight_decay_log10"])
    input_shape = (WINDOW_ROWS // CONVOLUTIONAL_STEPS, CONVOLUTIONAL_STEPS, len(FIELDS))

    inputs = Input(shape=input_shape)
    augmented = SensorAugmentation()(inputs)
    x = TimeDistributed(Conv1D(filters, kernel_size, padding="same", use_bias=False))(augmented)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = TimeDistributed(SpatialDropout1D(dropout_rate))(x)
    shortcut = x
    x = TimeDistributed(Conv1D(filters, kernel_size, padding="same", use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = TimeDistributed(Conv1D(filters, kernel_size, padding="same", use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([x, shortcut])
    x = TimeDistributed(LeakyReLU(negative_slope=0.1))(x)
    x = TimeDistributed(GlobalAveragePooling1D())(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)

    x = Bidirectional(LSTM(d_model // 2, return_sequences=True, dropout=dropout_rate * 0.5))(x)
    x = LayerNormalization()(x)
    sequence = LSTM(d_model, return_sequences=True, dropout=dropout_rate * 0.5)(x)
    sequence = LayerNormalization()(sequence)
    attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model)(sequence, sequence, sequence)
    attention = Dropout(dropout_rate)(attention)
    x = Add()([sequence, attention])
    x = LayerNormalization()(x)
    x = LSTM(d_model // 2, dropout=dropout_rate * 0.5)(x)

    statistics = WindowStatistics()(inputs)
    statistics = Dense(64, activation="elu")(statistics)
    statistics = LayerNormalization()(statistics)
    statistics = Dropout(dropout_rate)(statistics)
    x = Concatenate()([x, statistics])
    x = Dense(d_model // 2, activation="elu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(len(SURFACE_TYPE_LABELS), activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=1.0,
    )
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def sanitize_hyperparameters(values: Dict[str, float]) -> Dict[str, float]:
    cleaned = dict(values)
    for key in ["num_heads", "d_model", "filters", "kernel_size"]:
        low, high = PARAMETER_BOUNDS[key]
        cleaned[key] = int(np.clip(round(cleaned[key]), low, high))
    for key in ["dropout_rate", "learning_rate_log10", "weight_decay_log10"]:
        low, high = PARAMETER_BOUNDS[key]
        cleaned[key] = float(np.clip(cleaned[key], low, high))
    multiple = math.lcm(2, cleaned["num_heads"])
    cleaned["d_model"] = int(round(cleaned["d_model"] / multiple) * multiple)
    cleaned["d_model"] = int(np.clip(cleaned["d_model"], PARAMETER_BOUNDS["d_model"][0], PARAMETER_BOUNDS["d_model"][1]))
    while cleaned["d_model"] % multiple != 0:
        cleaned["d_model"] -= 1
    return cleaned


def random_hyperparameters(rng: np.random.Generator) -> Dict[str, float]:
    values = {}
    for key, (low, high) in PARAMETER_BOUNDS.items():
        if isinstance(low, int) and isinstance(high, int):
            values[key] = int(rng.integers(low, high + 1))
        else:
            values[key] = float(rng.uniform(low, high))
    return sanitize_hyperparameters(values)


def class_weights_from_outputs(outputs: np.ndarray) -> Dict[int, float]:
    counts = outputs.sum(axis=0)
    weights = np.sqrt(counts.sum() / (len(counts) * np.maximum(counts, 1.0)))
    weights /= weights.mean()
    weights = np.clip(weights, 0.7, 1.5)
    return {index: float(weight) for index, weight in enumerate(weights)}


def build_training_dataset(
    inputs: np.ndarray,
    outputs: np.ndarray,
    seed: int,
) -> tf.data.Dataset:
    class_weights = class_weights_from_outputs(outputs)
    labels = outputs.argmax(axis=1)
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs, sample_weights))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.shuffle(len(inputs), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(1)


def evaluate_candidate(
    hyperparameters: Dict[str, float],
    input_train: np.ndarray,
    output_train: np.ndarray,
    input_val: np.ndarray,
    output_val: np.ndarray,
    fold_seed: int,
) -> Tuple[float, int]:
    tf.keras.backend.clear_session()
    set_global_seed(fold_seed)
    model = build_model(hyperparameters)
    training_dataset = build_training_dataset(input_train, output_train, fold_seed)
    history = model.fit(
        training_dataset,
        validation_data=(input_val, output_val),
        epochs=OBJECTIVE_EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", mode="max", patience=OBJECTIVE_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            TerminateOnNaN(),
        ],
        verbose=0,
    )
    best_accuracy = float(max(history.history["val_accuracy"]))
    trained_epochs = len(history.history["val_accuracy"])
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return 1.0 - best_accuracy, trained_epochs


def optimize_with_koa(
    input_train: np.ndarray,
    output_train: np.ndarray,
    input_val: np.ndarray,
    output_val: np.ndarray,
    fold_seed: int,
    output_dir: Path,
) -> Dict[str, float]:
    rng = np.random.default_rng(fold_seed)
    planets = [sanitize_hyperparameters(DEFAULT_HYPERPARAMETERS)]
    planets.extend(random_hyperparameters(rng) for _ in range(KOA_PLANETS - 1))
    velocities = []
    for _ in range(KOA_PLANETS):
        velocities.append(
            {
                "num_heads": float(rng.uniform(-1, 1)),
                "d_model": float(rng.uniform(-16, 16)),
                "dropout_rate": float(rng.uniform(-0.04, 0.04)),
                "filters": float(rng.uniform(-12, 12)),
                "kernel_size": float(rng.uniform(-1, 1)),
                "learning_rate_log10": float(rng.uniform(-0.15, 0.15)),
                "weight_decay_log10": float(rng.uniform(-0.2, 0.2)),
            }
        )

    best_planet = None
    best_fitness = float("inf")
    search_rows = []
    search_index = 0

    for iteration in range(KOA_ITERATIONS):
        print(f"  KOA iteration {iteration + 1}/{KOA_ITERATIONS}", flush=True)
        for planet_index, planet in enumerate(planets):
            planet = sanitize_hyperparameters(planet)
            fitness, trained_epochs = evaluate_candidate(
                planet,
                input_train,
                output_train,
                input_val,
                output_val,
                fold_seed + search_index,
            )
            search_index += 1
            search_rows.append(
                {
                    "iteration": iteration + 1,
                    "planet": planet_index + 1,
                    **planet,
                    "val_accuracy": 1.0 - fitness,
                    "trained_epochs": trained_epochs,
                }
            )
            pd.DataFrame(search_rows).to_csv(output_dir / "koa_search.csv", index=False)
            print(f"    planet {planet_index + 1}: val_accuracy={1.0 - fitness:.4f}", flush=True)
            if fitness < best_fitness:
                best_fitness = fitness
                best_planet = dict(planet)
            planets[planet_index] = planet

        for planet_index, planet in enumerate(planets):
            for key in PARAMETER_BOUNDS:
                attraction = float(rng.uniform(0.2, 1.0))
                velocities[planet_index][key] += attraction * (best_planet[key] - planet[key])
                planet[key] += velocities[planet_index][key]
            planets[planet_index] = sanitize_hyperparameters(planet)

    best_planet = sanitize_hyperparameters(best_planet)
    best_planet["koa_validation_accuracy"] = 1.0 - best_fitness
    return best_planet


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def train_ensemble_member(
    hyperparameters: Dict[str, float],
    input_train: np.ndarray,
    output_train: np.ndarray,
    input_val: np.ndarray,
    output_val: np.ndarray,
    fold_dir: Path,
    seed: int,
    member_index: int,
    final_epochs: int,
) -> Tuple[Model, Dict[str, List[float]], Path]:
    tf.keras.backend.clear_session()
    set_global_seed(seed)
    model = build_model(hyperparameters)
    training_dataset = build_training_dataset(input_train, output_train, seed)
    weights_path = fold_dir / f"member_{member_index}_best.weights.h5"
    history_path = fold_dir / f"member_{member_index}_history.csv"
    backup_dir = fold_dir / f"member_{member_index}_training_backup"
    callbacks = [
        BackupAndRestore(backup_dir=str(backup_dir), delete_checkpoint=True),
        ModelCheckpoint(
            str(weights_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=FINAL_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        CSVLogger(str(history_path), append=True),
        TerminateOnNaN(),
    ]
    history = model.fit(
        training_dataset,
        validation_data=(input_val, output_val),
        epochs=final_epochs,
        callbacks=callbacks,
        verbose=0,
    )
    model.load_weights(weights_path)
    return model, history.history, weights_path


def fuse_sides(
    metadata: pd.DataFrame,
    outputs: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = metadata.copy()
    frame["y_true"] = outputs.argmax(axis=1)
    for index, label in enumerate(SURFACE_TYPE_LABELS):
        frame[f"prob_{label}"] = probabilities[:, index]
    probability_columns = [f"prob_{label}" for label in SURFACE_TYPE_LABELS]
    group_columns = ["dataset_key", "start_row", "end_row"]
    label_counts = frame.groupby(group_columns)["y_true"].nunique()
    if int(label_counts.max()) != 1:
        raise ValueError("Left/right labels are not synchronized")
    fused = frame.groupby(group_columns, as_index=False).agg(
        {"y_true": "first", "label_purity": "first", **{column: "mean" for column in probability_columns}}
    )
    fused = fused.sort_values(["dataset_key", "start_row"]).reset_index(drop=True)
    return fused, fused["y_true"].to_numpy(dtype=int), fused[probability_columns].to_numpy(dtype=np.float32)


def smooth_probabilities(frame: pd.DataFrame, probabilities: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return probabilities.copy()
    smoothed = np.empty_like(probabilities)
    for _, indices in frame.groupby("dataset_key", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        values = pd.DataFrame(probabilities[positions]).rolling(width, center=True, min_periods=1).mean().to_numpy()
        smoothed[positions] = values
    return smoothed


def choose_smoothing_width(frame: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[int, pd.DataFrame]:
    rows = []
    for width in SMOOTHING_WIDTHS:
        predictions = smooth_probabilities(frame, probabilities, width).argmax(axis=1)
        rows.append({"width": width, "validation_accuracy": accuracy_score(y_true, predictions)})
    scores = pd.DataFrame(rows)
    best_accuracy = scores["validation_accuracy"].max()
    selected = int(scores.loc[scores["validation_accuracy"].eq(best_accuracy), "width"].min())
    return selected, scores


@dataclass
class FoldResult:
    fold: int
    test_key: str
    metrics_side: Dict[str, float]
    metrics_fused: Dict[str, float]
    metrics_primary: Dict[str, float]
    confusion_matrix: List[List[int]]
    smoothing_width: int
    sample_count: int


def run_fold(
    split: Dict[str, List[str]],
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    run_dir: Path,
    args: argparse.Namespace,
) -> FoldResult:
    fold = int(split["fold"])
    fold_seed = 20260802 + fold * 100
    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Fold {fold} ===", flush=True)
    print(f"train={split['train']} val={split['val']} test={split['test']}", flush=True)

    scaler = fit_train_only_scaler(datasets, split["train"])
    input_train, output_train, meta_train = build_split_arrays(
        datasets,
        split["train"],
        scaler,
        TRAIN_STRIDE,
        TRAIN_LABEL_PURITY,
    )
    input_val, output_val, meta_val = build_split_arrays(datasets, split["val"], scaler, EVAL_STRIDE, 0.0)
    input_test, output_test, meta_test = build_split_arrays(datasets, split["test"], scaler, EVAL_STRIDE, 0.0)
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
        print("reusing completed KOA search", flush=True)
    elif args.skip_koa:
        hyperparameters = sanitize_hyperparameters(DEFAULT_HYPERPARAMETERS)
        hyperparameters["koa_validation_accuracy"] = None
    else:
        hyperparameters = optimize_with_koa(
            input_train,
            output_train,
            input_val,
            output_val,
            fold_seed,
            fold_dir,
        )
    save_json(hyperparameters_path, hyperparameters)
    print(f"hyperparameters={hyperparameters}", flush=True)

    validation_probabilities = []
    test_probabilities = []
    member_rows = []
    for member_index in range(1, args.ensemble_seeds + 1):
        member_seed = fold_seed + member_index
        member_result_path = fold_dir / f"member_{member_index}_result.json"
        member_validation_path = fold_dir / f"member_{member_index}_validation_probabilities.npy"
        member_test_path = fold_dir / f"member_{member_index}_test_probabilities.npy"
        if member_result_path.is_file() and member_validation_path.is_file() and member_test_path.is_file():
            print(f"  reusing completed ensemble member {member_index}/{args.ensemble_seeds}", flush=True)
            member_rows.append(json.loads(member_result_path.read_text(encoding="utf-8")))
            validation_probabilities.append(np.load(member_validation_path))
            test_probabilities.append(np.load(member_test_path))
            continue
        print(f"  ensemble member {member_index}/{args.ensemble_seeds}", flush=True)
        model, history, weights_path = train_ensemble_member(
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
        member_validation_probabilities = model.predict(input_val, batch_size=BATCH_SIZE, verbose=0)
        member_test_probabilities = model.predict(input_test, batch_size=BATCH_SIZE, verbose=0)
        validation_probabilities.append(member_validation_probabilities)
        test_probabilities.append(member_test_probabilities)
        np.save(member_validation_path, member_validation_probabilities)
        np.save(member_test_path, member_test_probabilities)
        best_epoch = int(np.argmax(history["val_accuracy"]) + 1)
        member_result = {
            "member": member_index,
            "seed": member_seed,
            "best_epoch": best_epoch,
            "best_val_accuracy": float(max(history["val_accuracy"])),
            "weights": str(weights_path),
        }
        member_rows.append(member_result)
        save_json(member_result_path, member_result)
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    pd.DataFrame(member_rows).to_csv(fold_dir / "ensemble_members.csv", index=False)
    probabilities_val = np.mean(validation_probabilities, axis=0)
    probabilities_test = np.mean(test_probabilities, axis=0)

    fused_val, y_val_true, fused_probabilities_val = fuse_sides(meta_val, output_val, probabilities_val)
    smoothing_width, smoothing_scores = choose_smoothing_width(fused_val, y_val_true, fused_probabilities_val)
    smoothing_scores.to_csv(fold_dir / "validation_smoothing_selection.csv", index=False)
    fused_test, y_test_true, fused_probabilities_test = fuse_sides(meta_test, output_test, probabilities_test)
    primary_probabilities = smooth_probabilities(fused_test, fused_probabilities_test, smoothing_width)

    y_side_true = output_test.argmax(axis=1)
    y_side_pred = probabilities_test.argmax(axis=1)
    y_fused_pred = fused_probabilities_test.argmax(axis=1)
    y_primary_pred = primary_probabilities.argmax(axis=1)
    metrics_side = compute_metrics(y_side_true, y_side_pred)
    metrics_fused = compute_metrics(y_test_true, y_fused_pred)
    metrics_primary = compute_metrics(y_test_true, y_primary_pred)
    matrix = confusion_matrix(y_test_true, y_primary_pred, labels=list(range(len(SURFACE_TYPE_LABELS))))

    prediction_frame = fused_test.copy()
    prediction_frame["y_pred_fused"] = y_fused_pred
    prediction_frame["y_pred_primary"] = y_primary_pred
    for index, label in enumerate(SURFACE_TYPE_LABELS):
        prediction_frame[f"prob_fused_{label}"] = fused_probabilities_test[:, index]
        prediction_frame[f"prob_primary_{label}"] = primary_probabilities[:, index]
    prediction_frame.to_csv(fold_dir / "test_predictions_fused.csv", index=False)
    pd.DataFrame(matrix, index=SURFACE_TYPE_LABELS, columns=SURFACE_TYPE_LABELS).to_csv(
        fold_dir / "confusion_matrix_primary.csv"
    )

    payload = {
        "fold": fold,
        "train_keys": split["train"],
        "val_keys": split["val"],
        "test_keys": split["test"],
        "hyperparameters": hyperparameters,
        "ensemble_members": member_rows,
        "smoothing_width_selected_on_validation": smoothing_width,
        "metrics_test_side_level": metrics_side,
        "metrics_test_left_right_fused": metrics_fused,
        "metrics_test_primary_fused_smoothed": metrics_primary,
        "primary_test_sample_count": len(y_test_true),
    }
    save_json(fold_dir / "fold_result.json", payload)
    print(
        f"fold {fold} primary_accuracy={metrics_primary['accuracy']:.4f} "
        f"fused_accuracy={metrics_fused['accuracy']:.4f} smoothing={smoothing_width}",
        flush=True,
    )

    del validation_probabilities, test_probabilities
    gc.collect()
    return FoldResult(
        fold=fold,
        test_key=split["test"][0],
        metrics_side=metrics_side,
        metrics_fused=metrics_fused,
        metrics_primary=metrics_primary,
        confusion_matrix=matrix.tolist(),
        smoothing_width=smoothing_width,
        sample_count=len(y_test_true),
    )


def summarize_results(results: List[FoldResult], run_dir: Path) -> None:
    rows = []
    aggregate = np.zeros((len(SURFACE_TYPE_LABELS), len(SURFACE_TYPE_LABELS)), dtype=int)
    for result in results:
        aggregate += np.asarray(result.confusion_matrix, dtype=int)
        rows.append(
            {
                "fold": result.fold,
                "test_key": result.test_key,
                "sample_count": result.sample_count,
                "smoothing_width": result.smoothing_width,
                "side_accuracy": result.metrics_side["accuracy"],
                "fused_accuracy": result.metrics_fused["accuracy"],
                "accuracy": result.metrics_primary["accuracy"],
                "precision": result.metrics_primary["precision"],
                "recall": result.metrics_primary["recall"],
                "f1_score": result.metrics_primary["f1_score"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("fold")
    frame.to_csv(run_dir / "fold_test_metrics.csv", index=False)
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1_score", "side_accuracy", "fused_accuracy"]:
        summary[metric] = {
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=0)),
            "min": float(frame[metric].min()),
            "max": float(frame[metric].max()),
        }
    save_json(run_dir / "summary_metrics.json", summary)
    pd.DataFrame(aggregate, index=SURFACE_TYPE_LABELS, columns=SURFACE_TYPE_LABELS).to_csv(
        run_dir / "aggregate_confusion_matrix.csv"
    )


def load_completed_fold(fold_dir: Path) -> FoldResult:
    payload = json.loads((fold_dir / "fold_result.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(fold_dir / "confusion_matrix_primary.csv", index_col=0).to_numpy(dtype=int)
    return FoldResult(
        fold=int(payload["fold"]),
        test_key=payload["test_keys"][0],
        metrics_side=payload["metrics_test_side_level"],
        metrics_fused=payload["metrics_test_left_right_fused"],
        metrics_primary=payload["metrics_test_primary_fused_smoothed"],
        confusion_matrix=matrix.tolist(),
        smoothing_width=int(payload["smoothing_width_selected_on_validation"]),
        sample_count=int(payload["primary_test_sample_count"]),
    )


def main() -> None:
    args = parse_args()
    enable_gpu_memory_growth()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume_dir is None:
        run_name = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_{args.run_tag}")
        run_dir = RESULTS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.resume_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    splits = build_standardized_splits()
    selected_splits = [split for split in splits if split["fold"] in set(args.folds)]
    run_config = {
        "model_name": MODEL_NAME,
        "datasets_dir": str(DATASETS_DIR),
        "fields": FIELDS,
        "window_rows": WINDOW_ROWS,
        "input_shape": [WINDOW_ROWS // CONVOLUTIONAL_STEPS, CONVOLUTIONAL_STEPS, len(FIELDS)],
        "train_stride": TRAIN_STRIDE,
        "eval_stride": EVAL_STRIDE,
        "train_label_purity": TRAIN_LABEL_PURITY,
        "batch_size": BATCH_SIZE,
        "final_epochs": args.final_epochs,
        "final_patience": FINAL_PATIENCE,
        "ensemble_seeds": args.ensemble_seeds,
        "koa_enabled": not args.skip_koa,
        "objective_epochs": OBJECTIVE_EPOCHS,
        "objective_patience": OBJECTIVE_PATIENCE,
        "koa_planets": KOA_PLANETS,
        "koa_iterations": KOA_ITERATIONS,
        "parameter_bounds": PARAMETER_BOUNDS,
        "default_hyperparameters": DEFAULT_HYPERPARAMETERS,
        "smoothing_widths": SMOOTHING_WIDTHS,
        "selected_folds": args.folds,
    }
    if not (run_dir / "run_config.json").is_file():
        save_json(run_dir / "run_config.json", run_config)
    if not (run_dir / "standardized_splits.json").is_file():
        save_json(run_dir / "standardized_splits.json", {"splits": splits})
    print(f"results={run_dir}", flush=True)
    print("loading datasets...", flush=True)
    datasets = load_datasets()
    results = []
    for split in selected_splits:
        fold_dir = run_dir / f"fold_{split['fold']}"
        if (fold_dir / "fold_result.json").is_file():
            print(f"reusing completed fold {split['fold']}", flush=True)
            results.append(load_completed_fold(fold_dir))
        else:
            results.append(run_fold(split, datasets, run_dir, args))
    summarize_results(results, run_dir)
    print(f"finished. results saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
