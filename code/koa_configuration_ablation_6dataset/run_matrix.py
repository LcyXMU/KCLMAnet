from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MATRIX = {
    "default": ["--algorithm", "default"],
    "koa_p4_i2_e8": ["--algorithm", "koa", "--koa-planets", "4", "--koa-iterations", "2"],
    "koa_p2_i8_e16": ["--algorithm", "koa", "--koa-planets", "2", "--koa-iterations", "8"],
    "koa_p4_i4_e16": ["--algorithm", "koa", "--koa-planets", "4", "--koa-iterations", "4"],
    "koa_p8_i2_e16": ["--algorithm", "koa", "--koa-planets", "8", "--koa-iterations", "2"],
    "random_e16": ["--algorithm", "random", "--search-evaluations", "16"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=MATRIX, default=list(MATRIX))
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--ensemble-seeds", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=160)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, name: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(BASE_DIR / "train_configuration.py"),
        *MATRIX[name],
        "--search-seed",
        str(args.search_seed),
        "--folds",
        *[str(fold) for fold in args.folds],
        "--ensemble-seeds",
        str(args.ensemble_seeds),
        "--final-epochs",
        str(args.final_epochs),
        "--resume-dir",
        str(run_dir),
    ]


def main() -> None:
    args = parse_args()
    manifest = {
        "run_id": args.run_id,
        "search_seed": args.search_seed,
        "folds": args.folds,
        "ensemble_seeds": args.ensemble_seeds,
        "final_epochs": args.final_epochs,
        "configs": args.configs,
    }
    manifest_path = BASE_DIR / "results" / f"matrix_{args.run_id}.json"
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    for name in args.configs:
        run_dir = BASE_DIR / "results" / name / f"run_{args.run_id}_seed{args.search_seed}"
        command = build_command(args, name, run_dir)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "summary_metrics.json").is_file():
            print(f"skip completed: {run_dir}", flush=True)
            continue
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        log_path = run_dir / "training.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONHASHSEED": "0",
                    "TF_DETERMINISTIC_OPS": "1",
                    "TF_CUDNN_DETERMINISTIC": "1",
                    "TF_ENABLE_ONEDNN_OPTS": "0",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        if return_code != 0:
            raise SystemExit(f"{name} failed with exit code {return_code}; see {log_path}")


if __name__ == "__main__":
    main()
