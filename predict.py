"""
Predice el signo de una o varias imagenes usando el modelo entrenado.

Uso:
    python predict.py ruta/a/imagen.png
    python predict.py carpeta_con_imagenes/
    python predict.py imagen.png --topk 5

Carga models/best.pt (generado por train.py).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from signs import display_name

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch, num_classes):
    if arch == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    elif arch == "resnet18":
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"arch desconocida: {arch}")
    return net


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = build_model(ckpt["arch"], len(ckpt["classes"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if ckpt.get("normalize_imagenet", True) else ([0.5] * 3, [0.5] * 3)
    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return net, ckpt["classes"], tf


def predict_one(net, tf, classes, img_path, topk):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(net(x), dim=1)[0]
    k = min(topk, len(classes))
    conf, idx = probs.topk(k)
    return [(classes[i], display_name(classes[i]), float(c)) for c, i in zip(conf, idx)]


def gather_images(path: Path):
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def main():
    ap = argparse.ArgumentParser(description="Predice el signo de una imagen.")
    ap.add_argument("input", help="Imagen o carpeta de imagenes.")
    ap.add_argument("--model", default=str(MODELS_DIR / "best.pt"))
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        raise SystemExit(f"No existe el modelo {ckpt_path}. Entrena primero: python train.py")

    net, classes, tf = load_model(ckpt_path)
    images = gather_images(Path(args.input))
    if not images:
        raise SystemExit(f"No se encontraron imagenes en {args.input}")

    for img_path in images:
        preds = predict_one(net, tf, classes, img_path, args.topk)
        print(f"\n{img_path.name}")
        for rank, (slug, name, conf) in enumerate(preds, 1):
            print(f"  {rank}. {name:<20} ({slug})  {conf*100:5.1f}%")


if __name__ == "__main__":
    main()
