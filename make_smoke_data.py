"""
Genera imagenes SINTETICAS de prueba para verificar el pipeline de punta a punta
SIN necesidad de tus imagenes reales. NO son los signos reales de Witch Hat Atelier:
son sellos esquematicos (anillo exterior + una forma geometrica distinta por clase)
solo para confirmar que train.py / predict.py funcionan en tu maquina.

Uso:
    python make_smoke_data.py                     # 30 img/clase, 4 clases demo
    python make_smoke_data.py --per-class 50 --out data/raw

Borra/ignora estas imagenes cuando tengas las reales.
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).parent

# Clases demo: (slug, funcion de dibujo del "signo" interno).
DEMO_CLASSES = ["column", "convergence", "float", "pull"]


def draw_seal(slug, size, rng):
    """Dibuja un sello sintetico: anillo exterior + figura interna segun la clase."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    cx = cy = size / 2
    jitter = lambda v: v + rng.uniform(-size * 0.03, size * 0.03)

    # Anillo exterior (presente en todos los hechizos).
    r = size * rng.uniform(0.40, 0.46)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=max(2, size // 80))

    # Figura interna distintiva por clase.
    ri = r * 0.55
    if slug == "column":
        # Lineas verticales paralelas (haz/columna).
        for dx in (-ri * 0.4, 0, ri * 0.4):
            d.line([jitter(cx + dx), cy - ri, jitter(cx + dx), cy + ri], fill="black", width=3)
    elif slug == "convergence":
        # Flechas apuntando al centro.
        for ang in range(0, 360, 45):
            a = math.radians(ang)
            x0, y0 = cx + ri * math.cos(a), cy + ri * math.sin(a)
            d.line([x0, y0, cx, cy], fill="black", width=2)
    elif slug == "float":
        # Circulos concentricos (flotacion).
        for f in (0.3, 0.55, 0.8):
            rr = ri * f
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline="black", width=2)
    elif slug == "pull":
        # Flechas apuntando hacia afuera + nucleo.
        d.ellipse([cx - ri * 0.2, cy - ri * 0.2, cx + ri * 0.2, cy + ri * 0.2], fill="black")
        for ang in range(0, 360, 90):
            a = math.radians(ang + rng.uniform(-8, 8))
            x1, y1 = cx + ri * 0.35 * math.cos(a), cy + ri * 0.35 * math.sin(a)
            x2, y2 = cx + ri * math.cos(a), cy + ri * math.sin(a)
            d.line([x1, y1, x2, y2], fill="black", width=3)
    else:
        d.line([cx - ri, cy, cx + ri, cy], fill="black", width=3)

    img = img.rotate(rng.uniform(-12, 12), fillcolor="white", resample=Image.BILINEAR)
    return img


def main():
    ap = argparse.ArgumentParser(description="Genera datos sinteticos de humo.")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)
    for slug in DEMO_CLASSES:
        folder = out / slug
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = draw_seal(slug, args.size, rng)
            img.save(folder / f"smoke_{i:03d}.png")
    print(f"Generadas {args.per_class} img/clase para {DEMO_CLASSES} en {out}")
    print("Verifica:  python prepare_data.py --report")
    print("Entrena:   python train.py --epochs 8")


if __name__ == "__main__":
    main()
