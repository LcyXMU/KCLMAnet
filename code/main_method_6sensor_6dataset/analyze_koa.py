from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "results" / "full" / (
    "run_20260806_091757_six_dataset_full_koa_ensemble"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_seconds(seconds: float) -> str:
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_best_parameter_table(run_dir: Path, folds: list[int]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        values = load_json(run_dir / f"fold_{fold}" / "best_hyperparameters.json")
        rows.append(
            {
                "fold": fold,
                "num_heads": values["num_heads"],
                "d_model": values["d_model"],
                "dropout_rate": values["dropout_rate"],
                "filters": values["filters"],
                "kernel_size": values["kernel_size"],
                "learning_rate_log10": values["learning_rate_log10"],
                "learning_rate": 10.0 ** values["learning_rate_log10"],
                "weight_decay_log10": values["weight_decay_log10"],
                "weight_decay": 10.0 ** values["weight_decay_log10"],
                "koa_validation_accuracy": values["koa_validation_accuracy"],
                "koa_search_seconds": values["koa_search_seconds"],
            }
        )
    return pd.DataFrame(rows)


def build_convergence_tables(
    run_dir: Path, folds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for fold in folds:
        frame = pd.read_csv(run_dir / f"fold_{fold}" / "koa_search.csv")
        frame.insert(0, "fold", fold)
        frame.insert(1, "evaluation", np.arange(1, len(frame) + 1))
        frame["fitness"] = 1.0 - frame["val_accuracy"]
        frame["best_val_accuracy_so_far"] = frame["val_accuracy"].cummax()
        frame["best_fitness_so_far"] = frame["fitness"].cummin()
        frames.append(frame)
    convergence = pd.concat(frames, ignore_index=True)
    summary = (
        convergence.groupby("evaluation", as_index=False)
        .agg(
            iteration=("iteration", "first"),
            candidate_val_accuracy_mean=("val_accuracy", "mean"),
            candidate_val_accuracy_std=("val_accuracy", "std"),
            cumulative_best_accuracy_mean=("best_val_accuracy_so_far", "mean"),
            cumulative_best_accuracy_std=("best_val_accuracy_so_far", "std"),
            cumulative_best_accuracy_min=("best_val_accuracy_so_far", "min"),
            cumulative_best_accuracy_max=("best_val_accuracy_so_far", "max"),
        )
    )
    iteration_rows = []
    for (fold, iteration), group in convergence.groupby(["fold", "iteration"], sort=True):
        last_evaluation = int(group["evaluation"].max())
        best_accuracy = float(
            convergence.loc[
                (convergence["fold"] == fold)
                & (convergence["evaluation"] <= last_evaluation),
                "val_accuracy",
            ].max()
        )
        iteration_rows.append(
            {
                "fold": int(fold),
                "iteration": int(iteration),
                "evaluations_completed": last_evaluation,
                "best_val_accuracy_so_far": best_accuracy,
                "best_fitness_so_far": 1.0 - best_accuracy,
            }
        )
    return convergence, summary, pd.DataFrame(iteration_rows)


def build_member_table(run_dir: Path, folds: list[int]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        frame = pd.read_csv(run_dir / f"fold_{fold}" / "ensemble_members.csv")
        frame.insert(0, "fold", fold)
        frames.append(frame.drop(columns=["weights"], errors="ignore"))
    return pd.concat(frames, ignore_index=True)


def plot_convergence(convergence: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.6), dpi=180)
    for fold, frame in convergence.groupby("fold"):
        axis.step(
            frame["evaluation"],
            100.0 * frame["best_val_accuracy_so_far"],
            where="post",
            linewidth=1.2,
            alpha=0.55,
            label=f"Fold {fold}",
        )
    x_values = summary["evaluation"].to_numpy(dtype=float)
    mean_values = 100.0 * summary["cumulative_best_accuracy_mean"].to_numpy(dtype=float)
    std_values = 100.0 * summary["cumulative_best_accuracy_std"].fillna(0.0).to_numpy(dtype=float)
    axis.plot(x_values, mean_values, color="black", marker="o", linewidth=2.6, label="Six-fold mean")
    axis.fill_between(
        x_values,
        mean_values - std_values,
        mean_values + std_values,
        color="black",
        alpha=0.12,
        label="Mean ± SD",
    )
    axis.axvline(4.5, color="gray", linestyle="--", linewidth=1.0)
    axis.text(2.5, axis.get_ylim()[0], "Iteration 1", ha="center", va="bottom", fontsize=9)
    axis.text(6.5, axis.get_ylim()[0], "Iteration 2", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(range(1, 9))
    axis.set_xlabel("Candidate evaluation")
    axis.set_ylabel("Best validation accuracy so far (%)")
    axis.set_title("KOA hyperparameter-search trajectory")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def write_report(
    run_dir: Path,
    run_config: dict,
    best_parameters: pd.DataFrame,
    convergence: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    members: pd.DataFrame,
) -> None:
    initial = convergence.loc[convergence["evaluation"] == 1, "val_accuracy"]
    final = convergence.groupby("fold")["best_val_accuracy_so_far"].last()
    gains = final.to_numpy() - initial.to_numpy()
    total_search_seconds = float(best_parameters["koa_search_seconds"].sum())
    total_training_seconds = float(members["training_seconds"].sum())
    total_candidate_epochs = int(convergence["trained_epochs"].sum())
    best_evaluations = (
        convergence.loc[
            convergence.groupby("fold")["best_val_accuracy_so_far"].idxmax(),
            ["fold", "evaluation"],
        ]
        .set_index("fold")["evaluation"]
        .astype(int)
    )
    parameter_rows = []
    for row in best_parameters.itertuples(index=False):
        parameter_rows.append(
            f"| {row.fold} | {row.num_heads} | {row.d_model} | {row.dropout_rate:.4f} | "
            f"{row.filters} | {row.kernel_size} | {row.learning_rate:.3e} | "
            f"{row.weight_decay:.3e} | {100.0 * row.koa_validation_accuracy:.4f} | "
            f"{format_seconds(row.koa_search_seconds)} |"
        )
    convergence_rows = []
    for row in convergence_summary.itertuples(index=False):
        convergence_rows.append(
            f"| {int(row.evaluation)} | {int(row.iteration)} | "
            f"{100.0 * row.candidate_val_accuracy_mean:.4f} ± "
            f"{100.0 * row.candidate_val_accuracy_std:.4f} | "
            f"{100.0 * row.cumulative_best_accuracy_mean:.4f} ± "
            f"{100.0 * row.cumulative_best_accuracy_std:.4f} |"
        )
    member_rows = []
    for row in members.itertuples(index=False):
        member_rows.append(
            f"| {row.fold} | {row.member} | {row.seed} | {row.best_epoch} | "
            f"{100.0 * row.best_val_accuracy:.4f} | {format_seconds(row.training_seconds)} |"
        )
    best_evaluation_text = ", ".join(
        f"Fold {fold}={evaluation}" for fold, evaluation in best_evaluations.items()
    )
    report = f"""# 算法参数与 KOA 收敛过程

## 1. 实验与训练设置

- 模型：{run_config['model_name']}。
- 输入：6 个同步传感器，每个传感器 6 个轴，模型输入形状为 `{tuple(run_config['input_shape'])}`。
- 窗口与步长：窗口 {run_config['window_rows']} 行；训练步长 {run_config['train_stride']}；验证和测试步长 {run_config['eval_stride']}。
- 交叉验证：6 个数据集按数据集级留一法划分；每折 4 个训练集、1 个验证集、1 个测试集。
- 最终训练：batch size={run_config['batch_size']}，最大 {run_config['final_epochs']} epochs，验证准确率早停 patience={run_config['final_patience']}。
- 集成：每折 {run_config['ensemble_seeds']} 个随机种子，输出概率取算术平均。
- 优化器：AdamW，梯度裁剪 `clipnorm=1.0`；学习率和权重衰减由 KOA 按折选择。
- 损失函数：categorical cross-entropy，label smoothing=0.02；类别权重采用平方根反频率并截断到 [0.7, 1.5]。
- 学习率调度：验证损失连续 5 epochs 未改善时乘 0.5，最小学习率 1e-5。
- 数据增强：训练期间加入标准差 0.015 的高斯噪声和 ±8% 随机增益。

## 2. KOA 搜索设置

- 适应度函数：`fitness = 1 - best validation accuracy`，因此适应度越小越好；测试集不参与超参数搜索。
- 种群与迭代：4 个 planets、2 次 iterations，每折 8 次候选评估，六折合计 48 次评估。
- 候选短训练：每个候选最多 {run_config['objective_epochs']} epochs，验证准确率早停 patience=3；验证损失调度 patience=2。
- 初始化：第 1 个 planet 使用默认参数，其余 3 个在搜索边界内随机初始化。
- 更新：每个参数按 `velocity += attraction × (global_best - current)` 更新，其中 attraction 从 [0.2, 1.0] 均匀采样。
- 初始速度范围：heads [-1,1]、d_model [-16,16]、dropout [-0.04,0.04]、filters [-12,12]、kernel [-1,1]、log10(LR) [-0.15,0.15]、log10(WD) [-0.2,0.2]。
- 离散约束：heads、d_model、filters 和 kernel 四舍五入并截断；d_model 调整为 `lcm(2, heads)` 的整数倍。
- 搜索空间：heads [2,6]；d_model [64,192]；dropout [0.10,0.35]；filters [48,160]；kernel [3,7]；log10(LR) [-4,-3]；log10(WD) [-6,-3.5]。
- 默认点：heads=4、d_model=128、dropout=0.20、filters=96、kernel=5、LR=3.162e-4、WD=3.162e-5。

## 3. 各折 KOA 最优参数

| Fold | Heads | d_model | Dropout | Filters | Kernel | Learning rate | Weight decay | KOA val acc. (%) | Search time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(parameter_rows)}

六折 KOA 搜索总耗时为 **{format_seconds(total_search_seconds)}**，候选模型累计短训练 **{total_candidate_epochs} epochs**。

## 4. 收敛过程

| Evaluation | Iteration | Candidate val acc. mean ± SD (%) | Cumulative best mean ± SD (%) |
|---:|---:|---:|---:|
{chr(10).join(convergence_rows)}

- 默认候选点的六折平均验证准确率为 **{100.0 * initial.mean():.4f}%**，搜索结束时累计最优均值为 **{100.0 * final.mean():.4f}%**。
- KOA 相对每折默认候选点的平均提升为 **{100.0 * gains.mean():.4f} 个百分点**；{int(np.count_nonzero(gains > 1e-12))}/6 折获得严格提升。
- 各折首次达到最终最优值的候选评估位置：{best_evaluation_text}。
- 收敛图见 `koa_convergence.png` 和矢量版 `koa_convergence.svg`；完整逐候选轨迹见 `koa_convergence.csv`。
- 当前仅设置 2 次迭代，因此曲线可以体现“搜索轨迹和累计最优值”，但不足以证明严格的渐近收敛。论文中建议表述为 optimization trajectory，而不要过度声称 convergence guarantee。
- 当前实现为不同候选使用 `fold_seed + evaluation_index`，所以曲线同时包含超参数差异与随机初始化差异。若要单独量化 KOA 的稳定增益，建议后续让同一折的候选共享固定种子，或每个候选重复多个种子后取均值。

## 5. 最终训练随机种子与轮次

| Fold | Member | Seed | Best epoch | Best val acc. (%) | Training time |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(member_rows)}

12 个成员的最终训练总耗时为 **{format_seconds(total_training_seconds)}**。成员最优 epoch 为按验证准确率选择的 1-based epoch。

## 6. 论文报告建议

- 正文给出 KOA 的种群数、迭代数、适应度函数和搜索空间；各折最优参数可放入附录或补充材料。
- 收敛图使用验证集累计最优准确率，不应使用测试准确率，以避免测试信息泄漏。
- 若审稿人要求更有说服力的收敛分析，应固定数据划分并增加 KOA 迭代数和独立搜索随机种子，再报告均值与标准差；这会产生一组新的实验，不能与本次 2-iteration 日志混写。
"""
    (run_dir / "ALGORITHM_PARAMETERS_AND_CONVERGENCE.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_config = load_json(run_dir / "run_config.json")
    folds = [int(fold) for fold in run_config["selected_folds"]]
    best_parameters = build_best_parameter_table(run_dir, folds)
    convergence, convergence_summary, iteration_convergence = build_convergence_tables(run_dir, folds)
    members = build_member_table(run_dir, folds)
    best_parameters.to_csv(run_dir / "best_hyperparameters_by_fold.csv", index=False)
    convergence.to_csv(run_dir / "koa_convergence.csv", index=False)
    convergence_summary.to_csv(run_dir / "koa_convergence_summary.csv", index=False)
    iteration_convergence.to_csv(run_dir / "koa_iteration_convergence.csv", index=False)
    members.to_csv(run_dir / "ensemble_training_parameters.csv", index=False)
    plot_convergence(convergence, convergence_summary, run_dir / "koa_convergence.png")
    write_report(run_dir, run_config, best_parameters, convergence, convergence_summary, members)
    print(f"Analysis written to {run_dir}")


if __name__ == "__main__":
    main()
