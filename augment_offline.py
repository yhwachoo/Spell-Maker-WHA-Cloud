"""
Expansion sintetica OFFLINE: convierte pocos recortes de referencia en muchas
variantes realistas para entrenar un clasificador preciso con datos escasos.

Por que: tienes ~1 dibujo por hechizo. Un modelo necesita variedad. Aqui generas
N variantes por referencia simulando como se veria el mismo sello redibujado o
fotografiado: rotacion leve, escala, traslacion, perspectiva, grosor de trazo,
desenfoque, brillo/contraste, ruido tipo papel y oclusiones.

Estructura de entrada (un recorte por clase ya basta; mas es mejor):
    data/refs/fuego_farolillo/ref.png
    data/refs/nubes/ref1.png
    data/refs/nubes/ref2.png
    ...

Uso:
    python augment_offline.py                       # 60 variantes/clase -> data/raw
    python augment_offline.py --per-class 120 --size 224
    python augment_offline.py --in data/refs --out data/raw --max-rot 12

NOTA de lore: por defecto NO se voltea (espejar un signo cambia su significado).
Las rotaciones son leves (+-12 grados) porque en el manga la orientacion importa.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def perspective(img, mag, rng):
    w, h = img.size
    d = mag * min(w, h)
    j = lambda: rng.uniform(-d, d)
    quad = (j(), j(), j(), h + j(), w + j(), h + j(), w + j(), j())  # UL, LL, LR, UR
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BILINEAR, fillcolor=255)


def scale_translate(img, rng, scale_range, trans):
    w, h = img.size
    s = rng.uniform(*scale_range)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (w, h), 255)
    mdx, mdy = int(trans * w), int(trans * h)
    ox = (w - nw) // 2 + rng.randint(-mdx, mdx)
    oy = (h - nh) // 2 + rng.randint(-mdy, mdy)
    canvas.paste(r, (ox, oy))
    return canvas


def vary_stroke(img, rng):
    """Engrosa o adelgaza el trazo (lineas oscuras sobre fondo claro)."""
    roll = rng.random()
    if roll < 0.33:
        return img.filter(ImageFilter.MinFilter(3))  # trazo mas grueso
    if roll < 0.66:
        return img.filter(ImageFilter.MaxFilter(3))  # trazo mas fino
    return img


def add_paper_noise(img, sigma):
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def random_erase(img, rng, max_boxes):
    d = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(rng.randint(0, max_boxes)):
        ew, eh = rng.randint(w // 20, w // 6), rng.randint(h // 20, h // 6)
        x, y = rng.randint(0, w - ew), rng.randint(0, h - eh)
        d.rectangle([x, y, x + ew, y + eh], fill=rng.choice([255, 248, 240]))
    return img


def make_variant(ref_img, size, max_rot, rng):
    img = ref_img.convert("L").resize((size, size), Image.BILINEAR)
    img = img.rotate(rng.uniform(-max_rot, max_rot), resample=Image.BILINEAR, fillcolor=255)
    img = perspective(img, rng.uniform(0.0, 0.10), rng)
    img = scale_translate(img, rng, scale_range=(0.82, 1.12), trans=0.06)
    img = vary_stroke(img, rng)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.2))
    img = add_paper_noise(img, sigma=rng.uniform(2, 10))
    img = random_erase(img, rng, max_boxes=2)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Expande recortes de referencia en variantes sinteticas.")
    ap.add_argument("--in", dest="in_dir", default=str(ROOT / "data" / "refs"))
    ap.add_argument("--out", dest="out_dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--max-rot", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="Borra aug_*.png previos en la carpeta destino.")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"No existe {in_dir}. Crea data/refs/<clase>/ y pon 1+ recorte por clase.")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    class_dirs = [d for d in sorted(in_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No hay subcarpetas de clase en {in_dir}.")

    total = 0
    for cdir in class_dirs:
        refs = [p for p in cdir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not refs:
            print(f"  (omito '{cdir.name}': sin imagenes)")
            continue
        dst = out_dir / cdir.name
        dst.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for old in dst.glob("aug_*.png"):
                old.unlink()
        ref_imgs = [Image.open(p) for p in refs]
        for i in range(args.per_class):
            ref = ref_imgs[i % len(ref_imgs)]
            make_variant(ref, args.size, args.max_rot, rng).save(dst / f"aug_{i:04d}.png")
        total += args.per_class
        print(f"  {cdir.name:<24} {len(refs)} ref -> {args.per_class} variantes")

    print(f"\nGeneradas {total} imagenes en {out_dir}")
    print("Revisa:  python prepare_data.py --report")
    print("Entrena: python train.py --arch resnet18 --epochs 40")


if __name__ == "__main__":
    main()
