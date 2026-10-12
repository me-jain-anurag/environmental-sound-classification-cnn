"""Audio loading and log-mel spectrogram features."""
import numpy as np
import librosa

from . import config as C


def load_audio(path, sr=C.SR, duration=C.DURATION):
    """Load a clip as mono float32, resampled to `sr` and padded/cut to exactly `duration` seconds."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    target = int(round(sr * duration))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def log_mel(y, sr=C.SR):
    """Log-scaled mel spectrogram in dB, shape (N_MELS, N_FRAMES), values in [-TOP_DB, 0]."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP,
                                       n_mels=C.N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=C.TOP_DB)
    return S_db[:, :C.N_FRAMES].astype(np.float32)


def normalise(S_db):
    """Map [-80, 0] dB to roughly [-1, 1] so the network sees zero-centred inputs."""
    return (S_db + C.TOP_DB / 2) / (C.TOP_DB / 2)


def file_to_input(path):
    """Full pipeline for one file -> network input of shape (N_MELS, N_FRAMES, 1)."""
    return normalise(log_mel(load_audio(path)))[..., np.newaxis]
