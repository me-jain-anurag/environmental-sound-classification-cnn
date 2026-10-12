"""Central configuration for the Environmental Sound Classification (ESC-50) CNN project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- dataset -------------------------------------------------------------
DATA_DIR = ROOT / "data" / "ESC-50"          # git clone https://github.com/karolpiczak/ESC-50 data/ESC-50
AUDIO_DIR = DATA_DIR / "audio"
META_CSV = DATA_DIR / "meta" / "esc50.csv"
FEATURE_FILE = ROOT / "features" / "esc50_logmel.npz"
RUNS_DIR = ROOT / "runs"

# ---- audio front-end -----------------------------------------------------
SR = 22050            # resample 44.1 kHz clips to 22.05 kHz (keeps content up to ~11 kHz)
DURATION = 5.0        # every ESC-50 clip is 5 s long
N_FFT = 2048          # ~93 ms analysis window
HOP = 512             # ~23 ms hop  ->  216 frames per clip
N_MELS = 128          # mel frequency bands
N_FRAMES = 216
TOP_DB = 80.0         # dynamic range kept by the log-mel transform

# ---- training defaults ---------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
