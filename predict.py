"""Classify audio files with the trained CNN and print the top-3 classes.

Usage:
    uv run predict.py path/to/sound.wav [more files ...]
    uv run predict.py samples/*.wav
    uv run predict.py --demo 8          # 8 random clips from the unseen test fold (needs the dataset)

Recordings longer than 5 s are split into 5 s windows and the predictions are averaged.
"""
import argparse
import os

from esc_cnn import config as C
from esc_cnn.infer import classify, load_classifier, true_label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--demo", type=int, default=0, help="pick N random clips from test fold 5 (downloads ESC-50)")
    ap.add_argument("--run", default="esc50", help="trained run in runs/ (esc50 or esc10)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    files = list(a.files)
    if a.demo:
        import pandas as pd
        from esc_cnn.download import ensure_dataset
        ensure_dataset()
        meta = pd.read_csv(C.META_CSV)
        test = meta[meta["fold"] == 5].sample(a.demo, random_state=a.seed)
        files += [str(C.AUDIO_DIR / f) for f in test["filename"]]
    if not files:
        ap.error("give one or more audio files, or --demo N")

    model, names = load_classifier(a.run)
    for f in files:
        prob, _ = classify(model, f)
        top = prob.argsort()[::-1][:3]
        true = true_label(f, names)
        mark = "   " if true is None else ("OK " if names[top[0]] == true else "XX ")
        print(f"{mark}{os.path.basename(f):<22s} true: {(true or '?'):<16s} predicted: "
              + " | ".join(f"{names[k]} {prob[k]:.2f}" for k in top))


if __name__ == "__main__":
    main()
