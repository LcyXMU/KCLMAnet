from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_matrix import MATRIX


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--interval-seconds", type=int, default=300)
    return parser.parse_args()


def process_matches(pid: int | None, run_id: str) -> bool:
    if pid is None:
        return False
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        return False
    command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return "run_matrix.py" in command and run_id in command


def completed(args: argparse.Namespace) -> bool:
    return all(
        (
            BASE_DIR
            / "results"
            / name
            / f"run_{args.run_id}_seed{args.search_seed}"
            / "summary_metrics.json"
        ).is_file()
        for name in MATRIX
    )


def status(args: argparse.Namespace) -> dict:
    configurations = {}
    for name in MATRIX:
        run_dir = BASE_DIR / "results" / name / f"run_{args.run_id}_seed{args.search_seed}"
        folds = {}
        for fold in range(1, 7):
            fold_dir = run_dir / f"fold_{fold}"
            histories = {}
            for history_path in sorted(fold_dir.glob("member_*_history.csv")):
                line_count = sum(1 for _ in history_path.open(encoding="utf-8"))
                histories[history_path.stem] = max(0, line_count - 1)
            folds[str(fold)] = {
                "complete": (fold_dir / "fold_result.json").is_file(),
                "search_evaluations": sum(
                    1 for _ in (fold_dir / "koa_search.csv").open(encoding="utf-8")
                ) - 1
                if (fold_dir / "koa_search.csv").is_file()
                else 0,
                "member_epochs": histories,
            }
        configurations[name] = {
            "complete": (run_dir / "summary_metrics.json").is_file(),
            "folds": folds,
        }
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "configurations": configurations}


def append_status(args: argparse.Namespace) -> None:
    path = BASE_DIR / "results" / f"supervisor_{args.run_id}_seed{args.search_seed}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(status(args), ensure_ascii=False) + "\n")


def wait_for_process(args: argparse.Namespace, pid: int) -> None:
    while process_matches(pid, args.run_id):
        append_status(args)
        time.sleep(args.interval_seconds)


def run_or_resume(args: argparse.Namespace) -> int:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "TF_DETERMINISTIC_OPS": "1",
            "TF_CUDNN_DETERMINISTIC": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
    )
    log_path = BASE_DIR / "results" / f"matrix_{args.run_id}_seed{args.search_seed}.log"
    command = [
        sys.executable,
        str(BASE_DIR / "run_matrix.py"),
        "--run-id",
        args.run_id,
        "--search-seed",
        str(args.search_seed),
    ]
    with log_path.open("a", encoding="utf-8") as stream:
        return subprocess.call(command, cwd=BASE_DIR, env=environment, stdout=stream, stderr=subprocess.STDOUT)


def summarize(args: argparse.Namespace) -> None:
    subprocess.check_call(
        [
            sys.executable,
            str(BASE_DIR / "summarize_matrix.py"),
            "--run-id",
            args.run_id,
            "--search-seed",
            str(args.search_seed),
        ],
        cwd=BASE_DIR,
    )
    subprocess.check_call(
        [
            sys.executable,
            str(BASE_DIR / "audit_matrix.py"),
            "--run-id",
            args.run_id,
            "--search-seed",
            str(args.search_seed),
        ],
        cwd=BASE_DIR,
    )


def main() -> None:
    args = parse_args()
    if args.watch_pid:
        wait_for_process(args, args.watch_pid)
    while not completed(args):
        append_status(args)
        return_code = run_or_resume(args)
        if return_code != 0:
            time.sleep(args.interval_seconds)
    append_status(args)
    summarize(args)


if __name__ == "__main__":
    main()
