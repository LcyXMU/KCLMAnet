from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

import train_configuration as experiment


def digest(arrays) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.asarray(array).tobytes())
    return hasher.hexdigest()


def main() -> None:
    args = experiment.ACTIVE_ARGS
    experiment.configure_experiment(args)
    training = experiment.training
    utilities = experiment.training_utils
    seed = 20261103
    training.tf.keras.backend.clear_session()
    utilities.set_global_seed(seed)
    model = training.build_model(utilities.DEFAULT_HYPERPARAMETERS, True)
    sample = np.ones((2, *model.input_shape[1:]), dtype=np.float32)
    augmentation = next(layer for layer in model.layers if isinstance(layer, training.SixSensorAugmentation))
    augmented = augmentation(sample, training=True).numpy()
    training_outputs = [model(sample, training=True).numpy() for _ in range(3)]
    weight_rows = []
    for variable in model.weights:
        values = variable.numpy()
        weight_rows.append({
            "path": variable.path,
            "sha256": digest([values]),
            "mean": float(values.mean()),
            "std": float(values.std()),
        })
    print(json.dumps({
        "weights_sha256": digest(model.get_weights()),
        "weights": weight_rows,
        "augmentation_sha256": digest([augmented]),
        "augmentation_mean": float(augmented.mean()),
        "augmentation_std": float(augmented.std()),
        "training_outputs_sha256": digest(training_outputs),
        "training_outputs": [values.tolist() for values in training_outputs],
    }, indent=2))


if __name__ == "__main__":
    main()
