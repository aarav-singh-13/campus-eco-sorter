"""
Convert the trained PyTorch model to ONNX so the website can run WITHOUT PyTorch
(PyTorch is ~860 MB - too big for Vercel; onnxruntime is ~60 MB).

Run:  export_onnx.bat     (or: backend\\.venv\\Scripts\\python training\\export_onnx.py)
Writes backend/model/mobilenetv3_waste.onnx, then checks it gives the same answers as PyTorch.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from model import IMG_CHECK_SIZES, ONNX_FILE, WasteClassifier, preprocess_numpy  # noqa: E402


def export(net, path):
    dummy = torch.randn(1, 3, 224, 224)
    common = dict(input_names=["input"], output_names=["logits"], opset_version=17)
    try:  # classic exporter: single self-contained .onnx file
        torch.onnx.export(net, dummy, str(path), dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False, **common)
        return
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(net, dummy, str(path), dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          **common)
        return
    except Exception as exc:
        print("Classic exporter failed, trying the new one:", exc)
    torch.onnx.export(net, (dummy,), str(path), dynamo=True, external_data=False, **common)


def main():
    clf = WasteClassifier(backend="torch")
    if clf.demo:
        sys.exit(f"Model not loaded: {clf.load_error}")
    net = clf.model.eval()

    export(net, ONNX_FILE)
    extra = ONNX_FILE.with_name(ONNX_FILE.name + ".data")
    if extra.exists():
        sys.exit(f"Export wrote a separate {extra.name} file - tell Claude, this needs a different setting.")
    print(f"Saved {ONNX_FILE}  ({ONNX_FILE.stat().st_size / 1e6:.1f} MB)")

    # ---- check ONNX == PyTorch ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    # same network output for the same input
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = net(x).numpy()
    out = sess.run(None, {name: x.numpy()})[0]
    print(f"Max difference in model output: {np.abs(ref - out).max():.2e}  (should be < 1e-3)")

    # same preprocessing + prediction on images of different shapes
    rng = np.random.default_rng(0)
    agree = 0
    imgs = []
    data = ROOT / "training" / "data" / "dataset-resized"
    if data.exists():  # real TrashNet photos if get_accuracy.bat downloaded them
        imgs = [Image.open(p).convert("RGB") for p in sorted(data.rglob("*.jpg"))[::60][:40]]
    imgs += [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for w, h in IMG_CHECK_SIZES]
    max_pix = 0.0
    for img in imgs:
        t = clf.transform(img).unsqueeze(0)
        n = preprocess_numpy(img)
        max_pix = max(max_pix, float(np.abs(t.numpy() - n).max()))
        with torch.no_grad():
            p_torch = int(net(t).argmax(1))
        p_onnx = int(sess.run(None, {name: n})[0].argmax(1)[0])
        agree += p_torch == p_onnx
    print(f"Max preprocessing difference: {max_pix:.2e}")
    print(f"Same prediction on {agree}/{len(imgs)} test images")
    if agree == len(imgs):
        print("\nAll good. Now run push_to_github.bat and redeploy on Vercel.")
    else:
        print("\nSome predictions differ - paste this output to Claude before deploying.")


if __name__ == "__main__":
    main()
