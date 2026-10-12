"""Step 2 - train the CNN on ESC-50 using the official folds.

Default protocol: train on folds 1-3, validate on fold 4 (early stopping / checkpoint),
report final accuracy on the untouched fold 5.

Usage:
    uv run train.py                       # ESC-50, 50 classes
    uv run train.py --subset esc10        # ESC-10 subset, 10 classes
    uv run train.py --epochs 1            # quick smoke test
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("COLUMNS", "100")   # wide enough for the full model.summary() table

import numpy as np
import keras
import tensorflow as tf

from esc_cnn import config as C
from esc_cnn.data import load_features, split, make_dataset
from esc_cnn.model import build_cnn


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=["esc50", "esc10"], default="esc50")
    p.add_argument("--train-folds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out", default=None, help="run folder (default runs/<subset>)")
    return p.parse_args()


def main():
    a = parse()
    keras.utils.set_random_seed(C.SEED)
    out = C.RUNS_DIR / (a.out or a.subset)
    out.mkdir(parents=True, exist_ok=True)
    if not C.FEATURE_FILE.exists():
        import extract_features
        extract_features.main()

    data = load_features(subset=a.subset)
    n_cls = len(data["class_names"])
    Xtr, ytr = split(data, a.train_folds)
    Xva, yva = split(data, [a.val_fold])
    Xte, yte = split(data, [a.test_fold])
    print(f"Subset: {a.subset.upper()}  |  classes: {n_cls}  |  input shape: {Xtr.shape[1:]}")
    print(f"Train folds {a.train_folds}: {len(Xtr)} clips  |  Val fold {a.val_fold}: {len(Xva)}"
          f"  |  Test fold {a.test_fold}: {len(Xte)}")

    train_ds = make_dataset(Xtr, ytr, n_cls, a.batch_size, train=True)
    val_ds = make_dataset(Xva, yva, n_cls, a.batch_size)

    model = build_cnn(num_classes=n_cls)
    model.compile(optimizer=keras.optimizers.Adam(a.lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=C.LABEL_SMOOTHING),
                  metrics=["accuracy"])
    model.summary(line_length=96)

    cbs = [
        keras.callbacks.ModelCheckpoint(out / "best_model.keras", monitor="val_accuracy",
                                        save_best_only=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=a.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(out / "history.csv"),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs,
                     shuffle=False, verbose=2)   # tf.data pipeline already shuffles
    train_min = (time.time() - t0) / 60

    best = keras.models.load_model(out / "best_model.keras")
    te_loss, te_acc = best.evaluate(make_dataset(Xte, yte, n_cls, a.batch_size), verbose=0)
    va_loss, va_acc = best.evaluate(val_ds, verbose=0)
    best_epoch = int(np.argmax(hist.history["val_accuracy"])) + 1
    res = {"subset": a.subset, "classes": n_cls, "train_clips": int(len(Xtr)), "val_clips": int(len(Xva)),
           "test_clips": int(len(Xte)), "epochs_run": len(hist.history["loss"]), "best_epoch": best_epoch,
           "val_accuracy": round(float(va_acc), 4), "test_accuracy": round(float(te_acc), 4),
           "test_loss": round(float(te_loss), 4), "params": int(best.count_params()),
           "train_minutes": round(train_min, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "classes.json").write_text(json.dumps(data["class_names"], indent=1))
    print("\n================ FINAL RESULT ================")
    print(f"Best epoch          : {best_epoch}")
    print(f"Validation accuracy : {va_acc * 100:.2f} %  (fold {a.val_fold})")
    print(f"Test accuracy       : {te_acc * 100:.2f} %  (fold {a.test_fold}, never seen in training)")
    print(f"Training time       : {train_min:.1f} min on CPU")
    print(f"Saved model -> {(out / 'best_model.keras').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
