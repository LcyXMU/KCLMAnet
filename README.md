# KCLMAnet reproducibility package

This folder collects the KOA-CNN-LSTM with Multi-Head Attention source code, and the self-collected XMU1-XMU6 datasets.

## One-command reproduction

```bash
./run_reproduction.sh
```

To additionally hash all approximately 1.0 GB of dataset files:

```bash
./run_reproduction.sh --verify-data-sha256
```

Expected primary six-fold arithmetic means are:

| Metric | Mean (%) |
|---|---:|
| Accuracy | 97.4554 |
| Weighted Precision | 97.5842 |
| Weighted Recall | 97.4554 |
| Weighted F1-score | 97.4073 |

Generated files are written to `reproduced_results/`. A successful run creates `verification_report.json` with `status` equal to `PASS`.

## Directory layout

```text
.
├── artifacts/koa8x2_sixfold/       # Saved weights, predictions, logs and fold results
├── code/                            # Source snapshot needed by the KOA 8x2 method
├── data/XMU 1 ... data/XMU 6/       # Self-collected datasets
├── manifests/                       # SHA-256 integrity records
├── reproduce_results.py             # Numeric reproduction and verification
└── run_reproduction.sh              # One-command entry point
```

## Experimental protocol represented by the artifacts

- Model: KOA-CNN-LSTM with Multi-Head Attention
- KOA population: 8 candidate planets
- KOA iterations: 2
- Candidate evaluations: 16 per fold
- Evaluation: six-fold leave-one-dataset-out on XMU1-XMU6
- Validation: the dataset following the test dataset in cyclic order
- Training: the remaining four datasets
- Input: six synchronized sensor positions and six IMU axes
- Window length: 600 samples
- Training stride: 150 samples
- Validation/test stride: 600 samples
- Final ensemble: two independently seeded members per fold

## Optional full retraining

Full retraining is intentionally not started by the one-command script. It requires the recorded TensorFlow environment and substantial GPU time. From this package, it can be started explicitly with:

```bash
cd code/koa_configuration_ablation_6dataset
/home/lcy/anaconda3/envs/Intelligentvehicle-gpu/bin/python train_configuration.py \
  --variant full \
  --algorithm koa \
  --koa-planets 8 \
  --koa-iterations 2 \
  --search-seed 0 \
  --folds 1 2 3 4 5 6 \
  --ensemble-seeds 2 \
  --final-epochs 160 \
  --run-tag koa8x2_full_retraining
```

Deterministic settings and seeds are recorded in the source and `artifacts/koa8x2_sixfold/run_config.json`. Saved-prediction reproduction is bit-for-bit verifiable. A new GPU training run can be deterministic in the recorded local environment, but exact floating-point identity is not guaranteed after changing the GPU, CUDA/cuDNN, TensorFlow, driver, or hardware.

## Publishing note

This directory is about 1.1 GB and contains large CSV and HDF5 files. For GitHub, store large data/model files with Git LFS or publish them as a versioned release/data archive, then retain the checksum manifests. Confirm that the dataset and model-weight licenses permit redistribution before making the repository public.
