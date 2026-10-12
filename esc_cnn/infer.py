"""Inference helpers shared by demo.py and predict.py (no dataset download needed)."""
import json
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import librosa

from . import config as C
from .features import log_mel, normalise

WINDOW = int(C.SR * C.DURATION)          # 5 s, the length the network was trained on
HOP = WINDOW // 2                        # longer recordings: 5 s windows with 50 % overlap
MAX_WINDOWS = 24                         # i.e. the first ~60 s of a long file
ESC_NAME = re.compile(r"^\d-\d+-[A-Z]-(\d+)\.wav$")   # ESC-50 file name pattern: {fold}-{id}-{take}-{target}.wav


def load_classifier(run="esc50"):
    """Return (keras model, list of class names) for a trained run in runs/<run>/."""
    import keras
    folder = C.RUNS_DIR / run
    model = keras.models.load_model(folder / "best_model.keras", compile=False)
    names = json.load(open(folder / "classes.json"))
    return model, names


def windows(path):
    """Split an audio file into 5 s log-mel inputs (shorter clips are zero-padded)."""
    y, _ = librosa.load(str(path), sr=C.SR, mono=True)
    if len(y) <= WINDOW:
        segs = [np.pad(y, (0, WINDOW - len(y)))]
    else:
        starts = list(range(0, len(y) - WINDOW + 1, HOP))[:MAX_WINDOWS]
        segs = [y[s:s + WINDOW] for s in starts]
    X = np.stack([normalise(log_mel(s.astype(np.float32))) for s in segs])[..., np.newaxis]
    return X, y


def classify(model, path):
    """Class probabilities for one file (averaged over its 5 s windows) and its waveform."""
    X, y = windows(path)
    prob = model.predict(X, verbose=0).mean(axis=0)
    return prob, y


def true_label(path, names):
    """Ground-truth class if the file keeps its original ESC-50 name, else None."""
    m = ESC_NAME.match(os.path.basename(str(path)))
    if m and len(names) == 50:
        return names[int(m.group(1))]
    return None
