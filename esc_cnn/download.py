"""Cross-platform download of the ESC-50 dataset (~600 MB) into data/ESC-50."""
import shutil
import sys
import urllib.request
import zipfile

from . import config as C

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def dataset_ready():
    return C.META_CSV.exists() and C.AUDIO_DIR.is_dir() and sum(1 for _ in C.AUDIO_DIR.glob("*.wav")) == 2000


def ensure_dataset():
    """Download and unpack ESC-50 unless it is already present."""
    if dataset_ready():
        return
    data_dir = C.DATA_DIR.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ESC-50-master.zip"
    print(f"Downloading ESC-50 (about 600 MB) from {URL}")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct}%)")
        else:
            sys.stdout.write(f"\r  {done / 1e6:7.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(URL, zip_path, progress)
    print("\nUnpacking ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    if C.DATA_DIR.exists():
        shutil.rmtree(C.DATA_DIR)
    (data_dir / "ESC-50-master").rename(C.DATA_DIR)
    zip_path.unlink()
    if not dataset_ready():
        raise SystemExit("ESC-50 download looks incomplete - delete the data/ folder and try again.")
    print(f"ESC-50 ready: 2000 clips in {C.AUDIO_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    ensure_dataset()
