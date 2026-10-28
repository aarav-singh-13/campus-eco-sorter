"""
Week 1 - Model evaluation.

Expects a test folder in ImageFolder layout, one sub-folder per MODEL label
(the names in backend/data/classes.json -> model_output_order), e.g. for TrashNet:
    test/
      cardboard/  glass/  metal/  paper/  plastic/  trash/
You only need this if you evaluate on your laptop; the Kaggle notebook already
reports the same metrics.

Usage (from the project root):
    python training/evaluate.py --data path/to/test
    python training/evaluate.py --data path/to/my_phone_photos --tag phone

Outputs (in training/results/):
    report_<tag>.txt          accuracy + per-class precision / recall / F1
    confusion_<tag>.png       confusion matrix
    misclassified_<tag>.csv   every wrong image with true / predicted / confidence
Also prints average inference time per image (CPU) and model file size.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402
from torchvision import datasets  # noqa: E402

from model import WasteClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Test folder in ImageFolder layout")
    ap.add_argument("--tag", default="test", help="Name used in output files, e.g. trashnet / campus")
    ap.add_argument("--threshold", type=float, default=0.60, help="Low-confidence threshold to report")
    args = ap.parse_args()

    clf = WasteClassifier()
    if clf.demo:
        sys.exit(f"Model not loaded: {clf.load_error}")

    ds = datasets.ImageFolder(args.data, transform=clf.transform)
    # Map the test folder names onto the model's class order.
    folder_to_model = {}
    for name, idx in ds.class_to_idx.items():
        if name not in clf.classes:
            sys.exit(f"Folder '{name}' is not in backend/data/classes.json: {clf.classes}")
        folder_to_model[idx] = clf.classes.index(name)

    y_true, y_pred, rows = [], [], []
    low_conf = 0
    times = []
    clf.model.eval()
    with torch.no_grad():
        for i, (x, folder_idx) in enumerate(ds):
            t0 = time.perf_counter()
            probs = torch.softmax(clf.model(x.unsqueeze(0)), dim=1)[0]
            times.append(time.perf_counter() - t0)
            conf, pred = probs.max(0)
            true = folder_to_model[folder_idx]
            y_true.append(true)
            y_pred.append(int(pred))
            if conf < args.threshold:
                low_conf += 1
            if int(pred) != true:
                rows.append([ds.samples[i][0], clf.classes[true], clf.classes[int(pred)], f"{conf:.3f}"])

    out = ROOT / "training" / "results"
    out.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(clf.classes)))

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=labels, target_names=clf.classes, digits=3, zero_division=0)
    model_file = ROOT / "backend" / "model" / clf.model_file
    summary = (
        f"Dataset: {args.data}  ({len(y_true)} images)\n"
        f"Accuracy: {acc:.3%}\n"
        f"Below {args.threshold:.0%} confidence: {low_conf} images ({low_conf / len(y_true):.1%})\n"
        f"Avg CPU inference: {1000 * sum(times) / len(times):.1f} ms/image\n"
        f"Model size: {model_file.stat().st_size / 1e6:.1f} MB\n\n{report}"
    )
    (out / f"report_{args.tag}.txt").write_text(summary, encoding="utf-8")
    print(summary)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=clf.classes).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion matrix - {args.tag}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out / f"confusion_{args.tag}.png", dpi=150)

    with open(out / f"misclassified_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "predicted", "confidence"])
        w.writerows(rows)

    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
