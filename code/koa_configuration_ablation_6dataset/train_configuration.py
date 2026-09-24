from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SIX_DATASET_DIR = BASE_DIR.parent / "main_method_6sensor_6dataset"
CANDIDATE_SEED_OFFSET = 50000

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def load_six_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "koa_configuration_six_dataset_base",
        SIX_DATASET_DIR / "train.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


six_dataset = load_six_dataset_module()
training = six_dataset.training
training_utils = training.training_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full"], default="full")
    parser.add_argument("--algorithm", choices=["default", "koa", "random"], required=True)
    parser.add_argument("--koa-planets", type=int, default=4)
    parser.add_argument("--koa-iterations", type=int, default=2)
    parser.add_argument("--search-evaluations", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--ensemble-seeds", type=int, default=training.ENSEMBLE_SEEDS)
    parser.add_argument("--final-epochs", type=int, default=training.FINAL_EPOCHS)
    parser.add_argument("--run-tag", default="koa_configuration_ablation")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.koa_planets < 2:
        parser.error("--koa-planets must be at least 2")
    if args.koa_iterations < 1:
        parser.error("--koa-iterations must be at least 1")
    if args.search_evaluations < 1:
        parser.error("--search-evaluations must be positive")
    return args


def candidate_training_seed(fold_seed: int) -> int:
    return fold_seed + CANDIDATE_SEED_OFFSET


def initialize_velocities(rng: np.random.Generator, planet_count: int) -> list[Dict[str, float]]:
    rows = []
    for _ in range(planet_count):
        rows.append(
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
    return rows


def fair_koa_search(
    input_train: np.ndarray,
    output_train: np.ndarray,
    input_val: np.ndarray,
    output_val: np.ndarray,
    fold_seed: int,
    output_dir: Path,
) -> Dict[str, float]:
    search_seed = fold_seed + int(ACTIVE_ARGS.search_seed) * 1_000_003
    rng = np.random.default_rng(search_seed)
    planet_count = int(ACTIVE_ARGS.koa_planets)
    iteration_count = int(ACTIVE_ARGS.koa_iterations)
    planets = [training_utils.sanitize_hyperparameters(training_utils.DEFAULT_HYPERPARAMETERS)]
    planets.extend(training_utils.random_hyperparameters(rng) for _ in range(planet_count - 1))
    velocities = initialize_velocities(rng, planet_count)
    best_planet = None
    best_fitness = float("inf")
    search_rows = []
    evaluation = 0
    fixed_candidate_seed = candidate_training_seed(fold_seed)

    for iteration in range(iteration_count):
        print(f"  KOA iteration {iteration + 1}/{iteration_count}", flush=True)
        for planet_index, current_planet in enumerate(planets):
            current_planet = training_utils.sanitize_hyperparameters(current_planet)
            fitness, trained_epochs = training_utils.evaluate_candidate(
                current_planet,
                input_train,
                output_train,
                input_val,
                output_val,
                fixed_candidate_seed,
            )
            evaluation += 1
            search_rows.append(
                {
                    "algorithm": "koa",
                    "search_seed": ACTIVE_ARGS.search_seed,
                    "candidate_training_seed": fixed_candidate_seed,
                    "evaluation": evaluation,
                    "iteration": iteration + 1,
                    "planet": planet_index + 1,
                    **current_planet,
                    "val_accuracy": 1.0 - fitness,
                    "trained_epochs": trained_epochs,
                }
            )
            pd.DataFrame(search_rows).to_csv(output_dir / "koa_search.csv", index=False)
            print(f"    planet {planet_index + 1}: val_accuracy={1.0 - fitness:.4f}", flush=True)
            if fitness < best_fitness:
                best_fitness = fitness
                best_planet = dict(current_planet)
            planets[planet_index] = current_planet

        for planet_index, current_planet in enumerate(planets):
            for key in training_utils.PARAMETER_BOUNDS:
                attraction = float(rng.uniform(0.2, 1.0))
                velocities[planet_index][key] += attraction * (best_planet[key] - current_planet[key])
                current_planet[key] += velocities[planet_index][key]
            planets[planet_index] = training_utils.sanitize_hyperparameters(current_planet)

    best_planet = training_utils.sanitize_hyperparameters(best_planet)
    best_planet["koa_validation_accuracy"] = 1.0 - best_fitness
    return best_planet


def random_search(
    input_train: np.ndarray,
    output_train: np.ndarray,
    input_val: np.ndarray,
    output_val: np.ndarray,
    fold_seed: int,
    output_dir: Path,
) -> Dict[str, float]:
    search_seed = fold_seed + int(ACTIVE_ARGS.search_seed) * 1_000_003
    rng = np.random.default_rng(search_seed)
    evaluation_count = int(ACTIVE_ARGS.search_evaluations)
    candidates = [training_utils.sanitize_hyperparameters(training_utils.DEFAULT_HYPERPARAMETERS)]
    candidates.extend(training_utils.random_hyperparameters(rng) for _ in range(evaluation_count - 1))
    fixed_candidate_seed = candidate_training_seed(fold_seed)
    best_candidate = None
    best_fitness = float("inf")
    search_rows = []

    for evaluation, candidate in enumerate(candidates, start=1):
        fitness, trained_epochs = training_utils.evaluate_candidate(
            candidate,
            input_train,
            output_train,
            input_val,
            output_val,
            fixed_candidate_seed,
        )
        search_rows.append(
            {
                "algorithm": "random",
                "search_seed": ACTIVE_ARGS.search_seed,
                "candidate_training_seed": fixed_candidate_seed,
                "evaluation": evaluation,
                "iteration": 1,
                "planet": evaluation,
                **candidate,
                "val_accuracy": 1.0 - fitness,
                "trained_epochs": trained_epochs,
            }
        )
        pd.DataFrame(search_rows).to_csv(output_dir / "koa_search.csv", index=False)
        print(f"  random candidate {evaluation}/{evaluation_count}: val_accuracy={1.0 - fitness:.4f}", flush=True)
        if fitness < best_fitness:
            best_fitness = fitness
            best_candidate = dict(candidate)

    best_candidate = training_utils.sanitize_hyperparameters(best_candidate)
    best_candidate["koa_validation_accuracy"] = 1.0 - best_fitness
    return best_candidate


def add_ablation_metadata(path: Path, payload: Dict) -> None:
    if path.name == "run_config.json":
        payload = dict(payload)
        payload.update(
            {
                "search_algorithm": ACTIVE_ARGS.algorithm,
                "search_seed": ACTIVE_ARGS.search_seed,
                "candidate_seed_policy": "fixed within each fold",
                "candidate_seed_offset": CANDIDATE_SEED_OFFSET,
                "deterministic_ops": True,
                "tf_deterministic_ops": os.environ["TF_DETERMINISTIC_OPS"],
                "tf_cudnn_deterministic": os.environ["TF_CUDNN_DETERMINISTIC"],
                "tf_enable_onednn_opts": os.environ["TF_ENABLE_ONEDNN_OPTS"],
                "koa_planets": ACTIVE_ARGS.koa_planets,
                "koa_iterations": ACTIVE_ARGS.koa_iterations,
                "search_evaluations": (
                    ACTIVE_ARGS.search_evaluations
                    if ACTIVE_ARGS.algorithm == "random"
                    else ACTIVE_ARGS.koa_planets * ACTIVE_ARGS.koa_iterations
                    if ACTIVE_ARGS.algorithm == "koa"
                    else 0
                ),
            }
        )
    ORIGINAL_SAVE_JSON(path, payload)


def configure_experiment(args: argparse.Namespace) -> None:
    training.tf.config.experimental.enable_op_determinism()
    training.RESULTS_DIR = BASE_DIR / "results"
    training.KOA_PLANETS = args.koa_planets
    training.KOA_ITERATIONS = args.koa_iterations
    training.parse_args = lambda: args
    training.get_dataset_keys = six_dataset.get_dataset_keys
    training.build_standardized_splits = six_dataset.build_six_dataset_splits
    training.load_datasets = six_dataset.load_six_datasets
    training.save_json = add_ablation_metadata
    training_utils.optimize_with_koa = fair_koa_search if args.algorithm == "koa" else random_search
    if args.algorithm == "default":
        training.VARIANTS["full"] = dict(training.VARIANTS["full"], koa=False)


def describe(args: argparse.Namespace) -> None:
    evaluations = (
        args.koa_planets * args.koa_iterations
        if args.algorithm == "koa"
        else args.search_evaluations
        if args.algorithm == "random"
        else 0
    )
    payload = {
        "algorithm": args.algorithm,
        "koa_planets": args.koa_planets,
        "koa_iterations": args.koa_iterations,
        "search_evaluations_per_fold": evaluations,
        "search_seed": args.search_seed,
        "candidate_seed_policy": "fixed within each fold",
        "deterministic_ops": True,
        "folds": args.folds,
        "ensemble_seeds": args.ensemble_seeds,
        "final_epochs": args.final_epochs,
        "resume_dir": str(args.resume_dir) if args.resume_dir else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


ORIGINAL_SAVE_JSON = training.save_json
ACTIVE_ARGS = parse_args()


def main() -> None:
    describe(ACTIVE_ARGS)
    if ACTIVE_ARGS.dry_run:
        return
    configure_experiment(ACTIVE_ARGS)
    training.main()


if __name__ == "__main__":
    main()
