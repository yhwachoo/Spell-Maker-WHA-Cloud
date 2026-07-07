"""
Entrena un clasificador de signos de Witch Hat Atelier (CPU-friendly).

Estrategia para pocas imagenes reales:
  - Transfer learning: backbone preentrenado (MobileNetV3-Small por defecto) +
    cabeza nueva. Si no hay internet para bajar pesos, cae a entrenar desde cero.
  - Aumentos de datos fuertes PERO sin volteos por defecto: en el lore, espejar o
    invertir un signo cambia su significado, asi que un flip generaria etiquetas
    erroneas. Usa --allow-flip si tu caso lo permite.
  - Division train/val estratificada automatica desde data/raw.

Uso tipico:
    python train.py
    python train.py --epochs 40 --arch resnet18 --img-size 160
    python train.py --no-pretrained          # forzar entrenamiento desde cero

Salida: models/best.pt  (incluye pesos, lista de clases y config).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from signs import display_name

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def scan_dataset(raw_dir: Path, min_per_class: int = 2):
    """Devuelve (samples, classes). samples = [(path, label_idx), ...]."""
    classes = []
    for d in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        imgs = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= min_per_class:
            classes.append((d.name, imgs))
        elif imgs:
            print(f"  (omito '{d.name}': solo {len(imgs)} img, se necesitan >={min_per_class})")
    class_names = [c[0] for c in classes]
    samples = []
    for idx, (_, imgs) in enumerate(classes):
        for p in imgs:
            samples.append((p, idx))
    return samples, class_names


def stratified_split(samples, val_frac, seed):
    """Divide por clase para que cada clase aparezca en val (si tiene >=2)."""
    by_class = {}
    for p, y in samples:
        by_class.setdefault(y, []).append(p)
    rng = random.Random(seed)
    train, val = [], []
    for y, paths in by_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_frac))) if len(paths) >= 2 else 0
        for p in paths[:n_val]:
            val.append((p, y))
        for p in paths[n_val:]:
            train.append((p, y))
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class SignDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms(img_size, normalize, allow_flip):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else \
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    train_ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3, fill=255),
    ]
    if allow_flip:
        train_ops.insert(1, transforms.RandomHorizontalFlip())
    train_ops += [transforms.ToTensor(), norm]
    val_ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model(arch, num_classes, pretrained, freeze_backbone):
    used_pretrained = pretrained
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        try:
            net = models.mobilenet_v3_small(weights=weights)
        except Exception as e:
            print(f"  No se pudieron bajar pesos preentrenados ({e}). Entreno desde cero.")
            net = models.mobilenet_v3_small(weights=None)
            used_pretrained = False
        if freeze_backbone and used_pretrained:
            for p in net.features.parameters():
                p.requires_grad = False
        in_f = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_f, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            net = models.resnet18(weights=weights)
        except Exception as e:
            print(f"  No se pudieron bajar pesos preentrenados ({e}). Entreno desde cero.")
            net = models.resnet18(weights=None)
            used_pretrained = False
        if freeze_backbone and used_pretrained:
            for p in net.parameters():
                p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"arch desconocida: {arch}")
    return net, used_pretrained


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def run_epoch(net, loader, criterion, optimizer, device, train, scaler=None):
    net.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    use_amp = scaler is not None
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = net(x)
            loss = criterion(out, y)
        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def confusion_report(net, loader, device, class_names):
    """Imprime accuracy por clase y matriz de confusion del mejor modelo."""
    from sklearn.metrics import classification_report, confusion_matrix
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        out = net(x.to(device))
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    names = [display_name(c) for c in class_names]
    print("\n--- Reporte por clase (sobre validacion) ---")
    print(classification_report(ys, ps, labels=list(range(len(class_names))),
                                target_names=names, zero_division=0))
    print("Matriz de confusion (filas=real, columnas=predicho):")
    print(confusion_matrix(ys, ps, labels=list(range(len(class_names)))))


def main():
    ap = argparse.ArgumentParser(description="Entrena el clasificador de signos.")
    ap.add_argument("--data-dir", default=str(RAW_DIR))
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-per-class", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Colab: usa 2).")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="Suaviza etiquetas; ayuda a generalizar.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-pretrained", action="store_true", help="No usar pesos preentrenados.")
    ap.add_argument("--full-finetune", action="store_true", help="Entrenar todo el backbone (no congelar).")
    ap.add_argument("--allow-flip", action="store_true", help="Permitir volteo horizontal (ojo: cambia significado del signo).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"No existe {data_dir}. Corre primero: python prepare_data.py")

    samples, class_names = scan_dataset(data_dir, args.min_per_class)
    if len(class_names) < 2:
        raise SystemExit("Necesitas >=2 clases con >=2 imagenes cada una. Corre: python prepare_data.py --report")
    print(f"Clases ({len(class_names)}): {', '.join(display_name(c) for c in class_names)}")
    print(f"Total imagenes: {len(samples)}")

    train_s, val_s = stratified_split(samples, args.val_frac, args.seed)
    print(f"Train: {len(train_s)}  |  Val: {len(val_s)}")

    pretrained = not args.no_pretrained
    normalize_imagenet = pretrained  # los pesos preentrenados esperan normalizacion ImageNet
    tf_train, tf_val = build_transforms(args.img_size, normalize_imagenet, args.allow_flip)

    pin = device.type == "cuda"
    train_loader = DataLoader(SignDataset(train_s, tf_train), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(SignDataset(val_s, tf_val), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin) if val_s else None

    net, used_pretrained = build_model(args.arch, len(class_names), pretrained, freeze_backbone=not args.full_finetune)
    net.to(device)
    print(f"Modelo: {args.arch} | preentrenado={used_pretrained} | backbone {'congelado' if (not args.full_finetune and used_pretrained) else 'entrenable'}")

    # Pesos por clase para mitigar desbalance.
    counts = np.bincount([y for _, y in train_s], minlength=len(class_names))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None  # mixed precision en GPU

    MODELS_DIR.mkdir(exist_ok=True)
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(net, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        if val_loader:
            va_loss, va_acc = run_epoch(net, val_loader, criterion, optimizer, device, train=False)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        scheduler.step()
        marker = ""
        if va_acc >= best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            marker = "  *"
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}{marker}")

    # Carga el mejor modelo y reporta desempeno por clase.
    if best_state is not None:
        net.load_state_dict(best_state)
    if val_loader:
        confusion_report(net, val_loader, device, class_names)

    ckpt = {
        "state_dict": best_state,
        "classes": class_names,
        "arch": args.arch,
        "img_size": args.img_size,
        "normalize_imagenet": normalize_imagenet,
        "best_val_acc": best_acc,
    }
    out = MODELS_DIR / "best.pt"
    torch.save(ckpt, out)
    (MODELS_DIR / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out}  (mejor val acc = {best_acc:.3f})")
    print("Predecir:  python predict.py <ruta_imagen>")


if __name__ == "__main__":
    main()
