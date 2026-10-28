"""
Optional: convert your trained .pth into a TorchScript .pt for faster, dependency-light serving.

Usage (from the project root):
    python training/export_torchscript.py --weights path/to/your_model.pth --arch large
Writes backend/model/mobilenetv3_waste.pt  (remove the .pth from backend/model afterwards).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from model import _strip_prefix, build_mobilenet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default="large", choices=["large", "small"])
    args = ap.parse_args()

    classes = json.loads((ROOT / "backend" / "data" / "classes.json").read_text())["model_output_order"]
    obj = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        net = obj
    else:
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(obj, dict) and isinstance(obj.get(key), dict):
                obj = obj[key]
                break
        net = build_mobilenet(len(classes), args.arch)
        net.load_state_dict(_strip_prefix(obj))
    net.eval()

    scripted = torch.jit.trace(net, torch.randn(1, 3, 224, 224))
    out = ROOT / "backend" / "model" / "mobilenetv3_waste.pt"
    scripted.save(str(out))
    print(f"Saved TorchScript model to {out}")


if __name__ == "__main__":
    main()
