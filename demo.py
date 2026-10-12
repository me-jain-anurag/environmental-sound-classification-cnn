"""One-command demo of the environmental sound classifier.

    uv run demo.py                  classify the 10 sample clips in samples/ (no dataset download)
    uv run demo.py my_sound.wav     classify your own recordings (wav / flac / ogg / mp3, any length)
    uv run demo.py --show           ... and open the result picture (demo_output.png)
    uv run demo.py --full           download ESC-50 (~600 MB) and reproduce the test-fold evaluation
"""
import argparse
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from esc_cnn import config as C

SAMPLES = C.ROOT / "samples"
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}


def open_file(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)                                   # noqa: S606 (local file)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:                                     # viewer missing: just report the path
        print(f"(could not open the picture automatically: {exc})")


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from esc_cnn.features import log_mel

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.55 * n + 0.4), gridspec_kw={"width_ratios": [1.6, 1]})
    if n == 1:
        axes = [axes]
    for (name, true, prob, y, names), (ax_s, ax_p) in zip(results, axes):
        librosa.display.specshow(log_mel(y[: int(C.SR * C.DURATION)]), sr=C.SR, hop_length=C.HOP,
                                 x_axis="time", y_axis="mel", ax=ax_s, cmap="magma")
        ax_s.set_ylabel(""); ax_s.set_xlabel(""); ax_s.tick_params(labelsize=7)
        ax_s.set_title(name if true is None else f"{name}   (true: {true})", fontsize=8, loc="left")
        top = prob.argsort()[::-1][:3]
        ok = true is None or names[top[0]] == true
        colors = ["#3a7a3a" if ok else "#a8443a", "#9db4c8", "#9db4c8"]
        ax_p.barh(range(3), prob[top], color=colors)
        ax_p.set_yticks(range(3), [names[k].replace("_", " ") for k in top], fontsize=8)
        ax_p.invert_yaxis(); ax_p.set_xlim(0, 1.12); ax_p.tick_params(axis="x", labelsize=7)
        for i, k in enumerate(top):
            ax_p.text(prob[k] + 0.02, i, f"{prob[k]:.2f}", va="center", fontsize=7)
    fig.suptitle("Environmental sound classification - log-mel spectrogram and top-3 CNN predictions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def classify_files(files, show):
    from esc_cnn.infer import classify, load_classifier, true_label

    print("Loading the trained CNN (runs/esc50/best_model.keras) ...")
    model, names = load_classifier("esc50")
    print(f"{'':3}{'file':<26}{'true class':<18}{'prediction (probability)':<34}2nd / 3rd guess")
    print("-" * 104)
    results, correct, known = [], 0, 0
    t0 = time.time()
    for f in files:
        prob, y = classify(model, f)
        top = prob.argsort()[::-1][:3]
        true = true_label(f, names)
        mark = "  "
        if true is not None:
            known += 1
            hit = names[top[0]] == true
            correct += hit
            mark = "OK" if hit else "XX"
        pred = f"{names[top[0]]} ({prob[top[0]]:.2f})"
        print(f"{mark:<3}{os.path.basename(str(f))[:25]:<26}{(true or '-'):<18}{pred:<34}"
              f"{names[top[1]]} / {names[top[2]]}")
        results.append((os.path.basename(str(f)), true, prob, y, names))
    print("-" * 104)
    if known:
        print(f"{correct} of {known} clips classified correctly  (OK = correct, XX = wrong)")
    print(f"Inference time: {time.time() - t0:.1f} s for {len(files)} file(s)")
    out = C.ROOT / "demo_output.png"
    plot(results, out)
    print(f"Picture with spectrograms and top-3 scores saved to {out.name}")
    if show:
        open_file(out)


def full_reproduction():
    from esc_cnn.download import ensure_dataset
    import extract_features
    import evaluate

    ensure_dataset()
    if not C.FEATURE_FILE.exists():
        extract_features.main()
    print()
    evaluate.main([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="audio files to classify (default: samples/*.wav)")
    ap.add_argument("--show", action="store_true", help="open demo_output.png when done")
    ap.add_argument("--full", action="store_true", help="download ESC-50 and reproduce the test-fold result")
    a = ap.parse_args()

    print(f"Sound classifier demo | Python {platform.python_version()} on {platform.system()}")
    if a.full:
        full_reproduction()
        return
    files = a.files or sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in AUDIO_EXT)
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        sys.exit(f"File not found: {missing[0]}")
    classify_files(files, a.show)


if __name__ == "__main__":
    main()
