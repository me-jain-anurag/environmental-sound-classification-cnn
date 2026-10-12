"""Step 3 - evaluate the trained CNN on the held-out test fold and produce plots.

Usage:  uv run evaluate.py                 (uses runs/esc50/best_model.keras, test fold 5)
        uv run evaluate.py --subset esc10
Writes runs/<subset>/eval/: metrics.json, classification_report.txt,
        confusion_matrix.png, per_class_accuracy.png, training_curves.png, top_confusions.csv
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             top_k_accuracy_score)

from esc_cnn import config as C
from esc_cnn.data import load_features, split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["esc50", "esc10"], default="esc50")
    ap.add_argument("--test-fold", type=int, default=5)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if not C.FEATURE_FILE.exists():
        import extract_features
        extract_features.main()
    run = C.RUNS_DIR / (a.run or a.subset)
    out = run / "eval"
    out.mkdir(parents=True, exist_ok=True)

    data = load_features(subset=a.subset)
    names = data["class_names"]
    n_cls = len(names)
    Xte, yte = split(data, [a.test_fold])
    model = keras.models.load_model(run / "best_model.keras", compile=False)
    prob = model.predict(Xte, batch_size=64, verbose=0)
    pred = prob.argmax(1)

    acc = accuracy_score(yte, pred)
    top3 = top_k_accuracy_score(yte, prob, k=3, labels=np.arange(n_cls))
    f1 = f1_score(yte, pred, average="macro")
    print(f"Model      : {run.relative_to(C.ROOT)}/best_model.keras")
    print(f"Test fold  : {a.test_fold}  ({len(yte)} clips, {n_cls} classes)")
    print(f"Accuracy   : {acc * 100:.2f} %")
    print(f"Top-3 acc. : {top3 * 100:.2f} %")
    print(f"Macro F1   : {f1:.4f}")

    rep = classification_report(yte, pred, labels=np.arange(n_cls), target_names=names, digits=3, zero_division=0)
    (out / "classification_report.txt").write_text(rep)
    json.dump({"accuracy": round(acc, 4), "top3_accuracy": round(top3, 4), "macro_f1": round(f1, 4),
               "test_clips": int(len(yte))}, open(out / "metrics.json", "w"), indent=2)

    cm = confusion_matrix(yte, pred, labels=np.arange(n_cls))
    per_cls = cm.diagonal() / cm.sum(1)
    order = np.argsort(per_cls)
    print("\nHardest classes :", ", ".join(f"{names[i]} ({per_cls[i]:.0%})" for i in order[:5]))
    print("Easiest classes :", ", ".join(f"{names[i]} ({per_cls[i]:.0%})" for i in order[::-1][:5]))

    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], names[i], names[j]) for i in range(n_cls) for j in range(n_cls) if off[i, j]),
                   reverse=True)[:10]
    pd.DataFrame(pairs, columns=["count", "true_class", "predicted_as"]).to_csv(out / "top_confusions.csv", index=False)
    print("Most frequent confusions (true -> predicted):")
    for c, t, p_ in pairs[:6]:
        print(f"   {t:>16s} -> {p_:<16s} x{c}")

    # ---- confusion matrix
    big = n_cls > 12
    fig, ax = plt.subplots(figsize=(11, 10) if big else (6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls), names, rotation=90, fontsize=6.5 if big else 9)
    ax.set_yticks(range(n_cls), names, fontsize=6.5 if big else 9)
    if not big:
        for i in range(n_cls):
            for j in range(n_cls):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix - test fold {a.test_fold} (accuracy {acc:.1%})")
    fig.colorbar(im, fraction=0.035, pad=0.02)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=200); plt.close(fig)

    # ---- per-class accuracy
    fig, ax = plt.subplots(figsize=(7, 10) if big else (6.5, 4))
    ax.barh(range(n_cls), per_cls[order] * 100, color=["#c0504d" if v < 0.5 else "#4f81bd" for v in per_cls[order]])
    ax.set_yticks(range(n_cls), [names[i] for i in order], fontsize=7 if big else 9)
    ax.axvline(acc * 100, color="k", ls="--", lw=1, label=f"overall {acc:.1%}")
    ax.set_xlabel("Test accuracy (%)"); ax.set_xlim(0, 100); ax.legend(loc="lower right")
    ax.set_title("Per-class accuracy on the test fold")
    fig.tight_layout(); fig.savefig(out / "per_class_accuracy.png", dpi=200); plt.close(fig)

    # ---- training curves
    h = pd.read_csv(run / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = h["epoch"] + 1
    ax[0].plot(ep, h["accuracy"] * 100, label="train (augmented)")
    ax[0].plot(ep, h["val_accuracy"] * 100, label="validation (fold 4)")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("%"); ax[0].legend()
    ax[1].plot(ep, h["loss"], label="train"); ax[1].plot(ep, h["val_loss"], label="validation")
    ax[1].set_title("Loss (cross-entropy)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for x in ax: x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=200); plt.close(fig)
    print(f"\nSaved report and plots -> {out.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
