"""Shared configuration for the EEG motor-imagery pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# PhysioNet EEG Motor Movement/Imagery Dataset
PHYSIONET_DB = "eegmmidb"
PHYSIONET_VERSION = "1.0.0"

DEFAULT_SUBJECTS = list(range(1, 21))  # S001–S020
MOTOR_RUNS = list(range(3, 15))  # skip baseline runs 1–2

SFREQ = 160.0
BANDPASS = (8.0, 30.0)
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0  # 4 s windows → 641 samples at 160 Hz
N_CHANNELS = 64

# 5-class mapping: rest / real vs imagined × hands vs feet
CLASS_NAMES = [
    "rest",
    "real_hands",
    "real_feet",
    "imagined_hands",
    "imagined_feet",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Run groups
REAL_HAND_LR_RUNS = {3, 7, 11}  # T1 left fist, T2 right fist
IMAG_HAND_LR_RUNS = {4, 8, 12}
REAL_HANDS_FEET_RUNS = {5, 9, 13}  # T1 both fists, T2 both feet
IMAG_HANDS_FEET_RUNS = {6, 10, 14}

CSP_COMPONENTS = 6
RANDOM_STATE = 42
WITHIN_TEST_SIZE = 0.2
