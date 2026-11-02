"""
Model loading and inference for the Campus Eco & Waste Sorting Identifier.

Supports these kinds of saved model in backend/model/:
  0. ONNX file (training/export_onnx.py)           -> mobilenetv3_waste.onnx
     Used first when onnxruntime is installed. Needs no PyTorch, so it fits on Vercel.
  1. TorchScript file  (torch.jit.save)            -> e.g. mobilenetv3_waste.pt
  2. state_dict / checkpoint dict (torch.save)       -> e.g. mobilenetv3_waste.pth
  3. A whole pickled nn.Module (torch.save(model))  -> e.g. mobilenetv3_waste.pth

If no model file is found (or neither onnxruntime nor torch is installed) the server runs in
DEMO MODE: it returns a repeatable fake prediction so the website can be
built and tested before the trained weights are dropped in.
"""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CLASSES_FILE = BASE_DIR / "data" / "classes.json"

# Which torchvision architecture you fine-tuned: "large" or "small".
MODEL_ARCH = os.getenv("MODEL_ARCH", "large").lower()
# Optional explicit path, otherwise the first .pt/.pth file in backend/model/ is used.
MODEL_PATH = os.getenv("MODEL_PATH", "")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms

    TORCH_AVAILABLE = True
except ImportError:  # Vercel runs without torch (ONNX), or demo mode
    TORCH_AVAILABLE = False

try:
    import numpy as np
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ONNX_FILE = MODEL_DIR / "mobilenetv3_waste.onnx"
# image shapes used by training/export_onnx.py to check preprocessing (portrait, landscape, square, odd)
IMG_CHECK_SIZES = [(512, 384), (384, 512), (300, 300), (1001, 333), (257, 999)]


def load_class_config():
    """(model labels in output-neuron order, {model label: website category})."""
    with open(CLASSES_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    labels = cfg["model_output_order"]
    mapping = cfg.get("label_to_category") or {l: l for l in labels}
    return labels, mapping


def build_transform():
    """Must match the validation/test transform used during training."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_mobilenet(num_classes, arch=MODEL_ARCH):
    """Recreate the fine-tuned architecture so a state_dict can be loaded into it."""
    if arch == "small":
        net = torchvision.models.mobilenet_v3_small(weights=None)
    else:
        net = torchvision.models.mobilenet_v3_large(weights=None)
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_features, num_classes)
    return net


def preprocess_numpy(img):
    """Same as build_transform() - Resize(256), CenterCrop(224), ToTensor, Normalize - without torch."""
    w, h = img.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    left = int(round((nw - 224) / 2.0))
    top = int(round((nh - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def find_model_file():
    if MODEL_PATH:
        p = Path(MODEL_PATH)
        return p if p.exists() else None
    for pattern in ("*.pt", "*.pth"):
        files = sorted(p for p in MODEL_DIR.glob(pattern) if p.is_file())
        if files:
            return files[0]
    # A .pth is really a zip file. If it got unzipped by accident (a folder with data.pkl
    # inside), accept the folder too.
    for d in sorted(p for p in MODEL_DIR.iterdir() if p.is_dir()):
        if (d / "data.pkl").exists() or (d / "constants.pkl").exists():
            return d
    return None


def _source(path):
    """File path as-is; an unzipped model folder is zipped back up in memory."""
    if path.is_file():
        return str(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(path.rglob("*")):
            if f.is_file():
                z.write(f, f"{path.name}/{f.relative_to(path).as_posix()}")
    buf.seek(0)
    return buf


def _strip_prefix(state_dict, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


class WasteClassifier:
    def __init__(self, backend="auto"):
        """backend: "auto" (ONNX if available, else PyTorch) or "torch" (force PyTorch)."""
        self.classes, self.label_to_category = load_class_config()
        # Website categories the model can actually predict (e.g. no compost/e-waste with TrashNet only)
        self.categories = list(dict.fromkeys(self.label_to_category[l] for l in self.classes))
        self.model = None
        self.session = None
        self.model_file = None
        self.demo = True
        self.load_error = None

        if backend == "auto" and ONNX_AVAILABLE and ONNX_FILE.exists():
            try:
                self.session = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.model_file = ONNX_FILE.name
                self.demo = False
                return
            except Exception as exc:
                self.load_error = f"Could not load {ONNX_FILE.name}: {exc}"

        if not TORCH_AVAILABLE:
            self.load_error = self.load_error or "No ONNX model/onnxruntime and no PyTorch - running in demo mode."
            return

        self.transform = build_transform()
        path = find_model_file()
        if path is None:
            self.load_error = f"No .pt/.pth file found in {MODEL_DIR} - running in demo mode."
            return

        try:
            self.model = self._load(path)
            self.model.eval()
            self.model_file = path.name
            self.demo = False
        except Exception as exc:  # show the reason on /api/health instead of crashing
            self.load_error = f"Could not load {path.name}: {exc}"
            self.model = None

    def _load(self, path):
        # 1) TorchScript
        try:
            return torch.jit.load(_source(path), map_location="cpu")
        except Exception:
            pass

        # 2/3) torch.save output. weights_only=False is needed for whole pickled
        # models - only load model files you trained yourselves.
        obj = torch.load(_source(path), map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            return obj

        if isinstance(obj, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
            net = build_mobilenet(len(self.classes))
            net.load_state_dict(_strip_prefix(obj))
            return net

        raise ValueError("Unrecognised model file format")

    @staticmethod
    def open_image(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # phone photos are often rotated via EXIF
        return img.convert("RGB")

    def predict(self, image_bytes):
        """Return [(website_category, probability), ...] sorted high -> low.

        Model labels that share a category (e.g. cardboard + paper) have their probabilities added.
        """
        img = self.open_image(image_bytes)  # also validates that it is an image

        if self.demo:
            probs = self._demo_probs(image_bytes)
        elif self.session is not None:
            logits = self.session.run(None, {self.input_name: preprocess_numpy(img)})[0][0]
            e = np.exp(logits - logits.max())
            probs = (e / e.sum()).tolist()
        else:
            x = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0].tolist()

        totals = {}
        for label, p in zip(self.classes, probs):
            cat = self.label_to_category[label]
            totals[cat] = totals.get(cat, 0.0) + p
        return sorted(totals.items(), key=lambda t: t[1], reverse=True)

    def _demo_probs(self, image_bytes):
        """Fake but repeatable scores per model label (same photo -> same answer)."""
        digest = hashlib.sha256(image_bytes).digest()
        raw = [digest[i] + 1 for i in range(len(self.classes))]
        raw[digest[10] % len(self.classes)] *= 8
        total = float(sum(raw))
        return [r / total for r in raw]
