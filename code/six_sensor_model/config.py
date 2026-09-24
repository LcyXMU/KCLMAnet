from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DATASETS_DIR = BASE_DIR.parent.parent / "data"

MODEL_NAME = "KOA-CNN-LSTM with Multi-Head Attention"

SURFACE_TYPE_LABELS = [
    "dirt_road",
    "cobblestone_road",
    "asphalt_road",
    "bluestone_road",
    "concrete_road",
]

SENSOR_LOCATIONS = [
    ("dashboard_left", "left", "dashboard"),
    ("dashboard_right", "right", "dashboard"),
    ("above_suspension_left", "left", "above_suspension"),
    ("above_suspension_right", "right", "above_suspension"),
    ("below_suspension_left", "left", "below_suspension"),
    ("below_suspension_right", "right", "below_suspension"),
]

SENSOR_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

WINDOW_ROWS = 600
CONVOLUTIONAL_STEPS = 25
TRAIN_STRIDE = 150
EVAL_STRIDE = 600
TRAIN_LABEL_PURITY = 0.9

BATCH_SIZE = 64
FINAL_EPOCHS = 160
FINAL_PATIENCE = 20
ENSEMBLE_SEEDS = 2

OBJECTIVE_EPOCHS = 8
OBJECTIVE_PATIENCE = 3
KOA_PLANETS = 4
KOA_ITERATIONS = 2

CAUSAL_SMOOTHING_WIDTHS = [1, 3, 5, 7, 11]

DEFAULT_HYPERPARAMETERS = {
    "num_heads": 4,
    "d_model": 128,
    "dropout_rate": 0.2,
    "filters": 96,
    "kernel_size": 5,
    "learning_rate_log10": -3.5,
    "weight_decay_log10": -4.5,
}

PARAMETER_BOUNDS = {
    "num_heads": (2, 6),
    "d_model": (64, 192),
    "dropout_rate": (0.1, 0.35),
    "filters": (48, 160),
    "kernel_size": (3, 7),
    "learning_rate_log10": (-4.0, -3.0),
    "weight_decay_log10": (-6.0, -3.5),
}
