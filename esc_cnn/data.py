"""Loading cached features, fold-based splits and on-the-fly augmentation (tf.data)."""
import numpy as np
import tensorflow as tf

from . import config as C
from .features import normalise


def load_features(path=C.FEATURE_FILE, subset="esc50"):
    """Return dict with X (N, 128, 216, 1) normalised, y, fold, category, filename, class_names.

    subset="esc10" keeps only the 10-class ESC-10 subset and re-indexes its labels 0..9.
    """
    d = np.load(path, allow_pickle=False)
    X, y, fold = d["X"], d["y"], d["fold"]
    cat, fname = d["category"], d["filename"]
    if subset == "esc10":
        keep = d["esc10"]
        X, y, fold, cat, fname = X[keep], y[keep], fold[keep], cat[keep], fname[keep]
        old = sorted(set(y.tolist()))
        remap = {o: i for i, o in enumerate(old)}
        y = np.array([remap[v] for v in y], dtype=np.int64)
    n_cls = int(y.max()) + 1
    class_names = [""] * n_cls
    for lab, c in zip(y, cat):
        class_names[lab] = str(c)
    return {"X": normalise(X)[..., np.newaxis].astype(np.float32), "y": y, "fold": fold,
            "category": cat, "filename": fname, "class_names": class_names}


def split(data, folds):
    m = np.isin(data["fold"], folds)
    return data["X"][m], data["y"][m]


# ---------------------------------------------------------------- augmentation
def _spec_augment(x):
    """Random circular time-shift + one frequency mask + one time mask (SpecAugment)."""
    x = tf.roll(x, shift=tf.random.uniform([], 0, C.N_FRAMES, dtype=tf.int32), axis=1)
    fill = tf.reduce_mean(x)
    f = tf.random.uniform([], 0, 20, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, C.N_MELS - f, dtype=tf.int32)
    fmask = tf.concat([tf.ones([f0]), tf.zeros([f]), tf.ones([C.N_MELS - f0 - f])], 0)
    t = tf.random.uniform([], 0, 30, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, C.N_FRAMES - t, dtype=tf.int32)
    tmask = tf.concat([tf.ones([t0]), tf.zeros([t]), tf.ones([C.N_FRAMES - t0 - t])], 0)
    mask = fmask[:, None, None] * tmask[None, :, None]
    return x * mask + fill * (1.0 - mask)


def _mixup(xb, yb, alpha):
    """Blend each example with a shuffled partner: x = l*x1 + (1-l)*x2, same for one-hot labels."""
    n = tf.shape(xb)[0]
    g1 = tf.random.gamma([n], alpha)
    g2 = tf.random.gamma([n], alpha)
    lam = g1 / (g1 + g2)
    lam = tf.maximum(lam, 1.0 - lam)                       # keep the original clip dominant
    idx = tf.random.shuffle(tf.range(n))
    lx = tf.reshape(lam, [-1, 1, 1, 1])
    ly = tf.reshape(lam, [-1, 1])
    return lx * xb + (1 - lx) * tf.gather(xb, idx), ly * yb + (1 - ly) * tf.gather(yb, idx)


def make_dataset(X, y, num_classes, batch_size=C.BATCH_SIZE, train=False, mixup_alpha=C.MIXUP_ALPHA):
    yo = tf.one_hot(y, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((X, yo))
    if train:
        ds = ds.shuffle(len(X), seed=C.SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b: (_spec_augment(a), b), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        if mixup_alpha and mixup_alpha > 0:
            ds = ds.map(lambda a, b: _mixup(a, b, mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)
