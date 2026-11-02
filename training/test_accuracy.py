"""
Re-computes the Kaggle notebook's test-set results on your laptop, using the trained model in
backend/model/. It rebuilds EXACTLY the same 15% test split as train_kaggle.ipynb (same file order,
same random_state=42), so the numbers match what the notebook would have printed.

Run (from the project folder):   get_accuracy.bat
or:   backend\\.venv\\Scripts\\python training\\test_accuracy.py

Downloads the original TrashNet (43 MB) from GitHub the first time, into training/data/.
"""

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn is missing. Run:  backend\\.venv\\Scripts\\pip install scikit-learn")

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DATA = ROOT / "training" / "data"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]   # same as the notebook
TO_BIN = {"cardboard": "paper", "glass": "glass", "metal": "metal", "paper": "paper",
          "plastic": "plastics", "trash": "landfill"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42


def get_trashnet():
    root = DATA / "dataset-resized"
    if root.exists():
        return root
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "dataset-resized.zip"
    if not zpath.exists():
        print("Downloading TrashNet (43 MB) ...")
        urllib.request.urlretrieve(TRASHNET_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    zpath.unlink()
    return root


def build_samples(root):
    """Same order as the notebook: classes in CLASSES order, files sorted by path."""
    samples = []
    for i, c in enumerate(CLASSES):
        files = sorted(p for p in (root / c).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"))
        samples += [(str(p), i) for p in files]
    return samples


def split(samples):
    """Identical to the notebook's cell 3."""
    labels = [l for _, l in samples]
    train, rest = train_test_split(samples, test_size=0.30, stratify=labels, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=[l for _, l in rest], random_state=SEED)
    return train, val, test


def main():
    import torch
    from model import WasteClassifier

    root = get_trashnet()
    samples = build_samples(root)
    _, _, test = split(samples)
    print(f"TrashNet images: {len(samples)}   test images: {len(test)}")

    clf = WasteClassifier()
    if clf.demo:
        sys.exit(f"Model not loaded: {clf.load_error}")
    if clf.classes != CLASSES:
        sys.exit(f"classes.json order {clf.classes} differs from the notebook's {CLASSES}")

    y_true, y_pred = [], []
    with torch.no_grad():
        for n, (path, label) in enumerate(test, 1):
            x = clf.transform(clf.open_image(Path(path).read_bytes())).unsqueeze(0)
            y_pred.append(int(clf.model(x).argmax(1)))
            y_true.append(label)
            if n % 50 == 0:
                print(f"  {n}/{len(test)}")

    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    bin_acc = sum(TO_BIN[CLASSES[t]] == TO_BIN[CLASSES[p]] for t, p in zip(y_true, y_pred)) / n

    lines = [
        "",
        "=" * 50,
        f"Test accuracy (6 classes):  {acc:.1%}   ({sum(t == p for t, p in zip(y_true, y_pred))}/{n})",
        f"Correct bin chosen:         {bin_acc:.1%}",
        "=" * 50,
        "Per class (correct / total):",
    ]
    for i, c in enumerate(CLASSES):
        tot = sum(1 for t in y_true if t == i)
        ok = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        lines.append(f"  {c:10s} {ok:3d} / {tot:3d}   ({ok / tot:.0%})")
    text = "\n".join(lines)
    print(text)

    out = ROOT / "training" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_accuracy.txt").write_text(text, encoding="utf-8")
    (out / "test_accuracy.json").write_text(json.dumps({"accuracy": acc, "bin_accuracy": bin_acc, "n": n}), encoding="utf-8")
    print(f"\nSaved to {out / 'test_accuracy.txt'}")


if __name__ == "__main__":
    main()
