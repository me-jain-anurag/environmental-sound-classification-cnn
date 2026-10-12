"""Step 1 - convert all 2,000 ESC-50 clips into log-mel spectrograms and cache them.

Usage:  uv run extract_features.py      (downloads ESC-50 first if data/ESC-50 is missing)
Output: features/esc50_logmel.npz  (X: 2000 x 128 x 216, labels, folds, categories)
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from esc_cnn import config as C
from esc_cnn.download import ensure_dataset
from esc_cnn.features import load_audio, log_mel


def one(fname):
    return log_mel(load_audio(C.AUDIO_DIR / fname))


def main():
    ensure_dataset()
    meta = pd.read_csv(C.META_CSV)
    print(f"ESC-50 metadata: {len(meta)} clips, {meta['target'].nunique()} classes, "
          f"{meta['fold'].nunique()} folds")
    print(f"Front-end: sr={C.SR} Hz, n_fft={C.N_FFT}, hop={C.HOP}, n_mels={C.N_MELS}")
    t0 = time.time()
    X = Parallel(n_jobs=-1, verbose=0)(delayed(one)(f) for f in meta["filename"])
    X = np.stack(X).astype(np.float32)
    print(f"Extracted log-mel spectrograms: {X.shape}  in {time.time() - t0:.1f} s")
    print(f"dB range: min={X.min():.1f}  max={X.max():.1f}  mean={X.mean():.1f}")

    C.FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.FEATURE_FILE, X=X,
                        y=meta["target"].to_numpy(np.int64),
                        fold=meta["fold"].to_numpy(np.int64),
                        esc10=meta["esc10"].to_numpy(bool),
                        filename=meta["filename"].to_numpy(str),
                        category=meta["category"].to_numpy(str))
    print(f"Saved -> {C.FEATURE_FILE.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
