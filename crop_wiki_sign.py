"""
Recorta el GLIFO de un screenshot de la wiki (tabla Image|Name|Description) y lo
guarda en data/signs/<clase>/. El glifo esta dibujado en ROJO OSCURO/GRANATE; el
nombre y el texto estan en negro/gris/teal. Aprovechamos eso para aislar el glifo
POR COLOR (pixeles rojizos), sin atrapar el texto.

Uso (un archivo, indicando la clase):
    python crop_wiki_sign.py column.png --name column
    python crop_wiki_sign.py captura.webp --name pull --out data/signs

Uso (carpeta cuyos nombres de archivo son la clase):
    python crop_wiki_sign.py carpeta/        # usa el nombre de cada archivo como clase

Salida: data/signs/<clase>/<clase>.png  (glifo sobre fondo blanco, cuadrado).
Carpeta hermana de data/refs/ (misma rama, separada de los hechizos del catalogo).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def crop_glyph(path, left_frac=0.45, pad=0.12):
    """Devuelve el recorte cuadrado del glifo rojizo, o None si no encuentra."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    H, W = R.shape

    # Pixeles rojizos oscuros (granate): R claramente mayor que G y B, y no muy claro.
    maroon = (R - G > 35) & (R - B > 35) & (R < 210) & (R > 50)
    # El glifo esta a la izquierda; ignora la parte derecha (nombre/descripcion).
    maroon[:, int(left_frac * W):] = False

    ys, xs = np.where(maroon)
    if len(xs) < 20:
        return None, 0

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    p = int(pad * max(h, w))
    side = max(h, w) + 2 * p
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    src = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (side, side), "white")
    crop = src.crop((cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side))
    canvas.paste(crop, (0, 0))
    return canvas, len(xs)


def save_for_class(img, slug, out_dir):
    dst = out_dir / slug
    dst.mkdir(parents=True, exist_ok=True)
    img.save(dst / f"{slug}.png")
    return dst / f"{slug}.png"


def main():
    ap = argparse.ArgumentParser(description="Recorta el glifo rojizo de un screenshot de la wiki.")
    ap.add_argument("input", help="Imagen (con --name) o carpeta (nombres de archivo = clase).")
    ap.add_argument("--name", default=None, help="Clase para una sola imagen (p.ej. column).")
    ap.add_argument("--out", default=str(ROOT / "data" / "signs"))
    ap.add_argument("--left-frac", type=float, default=0.45, help="Fraccion izquierda donde buscar el glifo.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"No existe: {p}")

    jobs = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                jobs.append((f, f.stem.lower()))
    else:
        slug = (args.name or p.stem).lower().replace(" ", "_")
        jobs.append((p, slug))

    ok = 0
    for f, slug in jobs:
        img, n = crop_glyph(f, args.left_frac)
        if img is None:
            print(f"  ! {f.name}: no se hallo glifo rojizo (prueba --left-frac mayor)")
            continue
        dst = save_for_class(img, slug, out_dir)
        print(f"  {f.name:<28} -> {dst}  ({n} px rojizos)")
        ok += 1
    print(f"\n{ok}/{len(jobs)} glifos guardados en {out_dir}")
    if ok:
        print("Revisa los recortes; luego expande con:")
        print(f"  python augment_offline.py --in {out_dir} --out data/signs_raw --per-class 80 --clean")


if __name__ == "__main__":
    main()
