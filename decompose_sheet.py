"""
Descompone una HOJA-CATALOGO (varios sellos en una imagen) en recortes individuales
listos para clasificar/expandir. Sin OpenCV: usa scipy.ndimage.

Detecta cada sello aprovechando que el ANILLO es un circulo cerrado: cierra huecos,
rellena el disco y separa cada sello como un componente conexo.

Uso:
    # Modo automatico (detecta circulos):
    python decompose_sheet.py hoja1.png
    python decompose_sheet.py hoja1.png --out data/refs/_unsorted --montage

    # Modo rejilla (hojas ordenadas en grilla R x C, mas fiable si el auto falla):
    python decompose_sheet.py hoja1.png --grid 4x9

    # Extraer SIGNOS sueltos DENTRO de un unico sello (experimental):
    python decompose_sheet.py un_sello.png --level sign

Salida: recortes seal_000.png... en data/refs/_unsorted/<nombre_hoja>/.
Luego RENOMBRA/mueve cada recorte a su carpeta de clase, p.ej. data/refs/nubes/.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).parent


# --------------------------------------------------------------------------- #
def otsu_threshold(arr):
    """Umbral de Otsu (numpy)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sumB = wB = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def to_gray_array(path):
    img = Image.open(path).convert("L")
    return np.asarray(img), img


def ink_mask(arr, thr=None):
    if thr is None:
        thr = otsu_threshold(arr)
    return arr <= thr  # tinta = pixeles oscuros (<= incluye el cluster oscuro de Otsu)


def reading_order(boxes):
    """Ordena bboxes (y0,x0,y1,x1) en orden de lectura: filas arriba->abajo, izq->der."""
    if not boxes:
        return []
    heights = [b[2] - b[0] for b in boxes]
    row_tol = 0.6 * np.median(heights)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    rows, cur, cur_y = [], [], None
    for i in order:
        y0 = boxes[i][0]
        if cur_y is None or y0 - cur_y <= row_tol:
            cur.append(i)
            cur_y = y0 if cur_y is None else cur_y
        else:
            rows.append(cur)
            cur, cur_y = [i], y0
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        out.extend(sorted(r, key=lambda i: boxes[i][1]))
    return out


def crop_square(img, box, pad_frac):
    y0, x0, y1, x1 = box
    h, w = y1 - y0, x1 - x0
    pad = int(pad_frac * max(h, w))
    side = max(h, w) + 2 * pad
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = Image.new("L", (side, side), 255)
    src = img.crop((cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side))
    canvas.paste(src, (0, 0))
    return canvas.convert("RGB")


def save_montage(crops, out_path, cell=128, cols=8):
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, c in enumerate(crops):
        thumb = c.resize((cell, cell))
        sheet.paste(thumb, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(out_path)


# --------------------------------------------------------------------------- #
def detect_seals(arr, img, args):
    """Modo auto: rellena discos cerrados y separa cada sello."""
    mask = ink_mask(arr, args.thr)
    mind = min(arr.shape)
    d = args.dilate if args.dilate > 0 else max(2, mind // 250)
    closed = ndimage.binary_dilation(mask, iterations=d)
    filled = ndimage.binary_fill_holes(closed)
    lbl, n = ndimage.label(filled)
    if n == 0:
        return []
    slices = ndimage.find_objects(lbl)
    counts = np.bincount(lbl.ravel())
    min_side = args.min_frac * mind
    max_side = args.max_frac * mind
    boxes = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        if not (min_side <= min(h, w) and max(h, w) <= max_side):
            continue
        if max(h, w) / max(min(h, w), 1) > args.aspect:   # casi cuadrado (circulo)
            continue
        fill = counts[i] / max(h * w, 1)                   # disco lleno ~ alto
        if fill < args.min_fill:
            continue
        boxes.append((y0, x0, y1, x1))
    order = reading_order(boxes)
    return [crop_square(img, boxes[i], args.pad) for i in order]


def split_grid(arr, img, rows, cols, args):
    H, W = arr.shape
    crops = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * H // rows, (r + 1) * H // rows
            x0, x1 = c * W // cols, (c + 1) * W // cols
            cell = arr[y0:y1, x0:x1]
            mask = ink_mask(cell, args.thr)
            if mask.sum() < 0.002 * cell.size:             # celda casi vacia
                continue
            ys, xs = np.where(mask)
            box = (y0 + ys.min(), x0 + xs.min(), y0 + ys.max(), x0 + xs.max())
            crops.append(crop_square(img, box, args.pad))
    return crops


def detect_signs(arr, img, args):
    """Experimental: extrae signos sueltos dentro de UN sello (quita el anillo)."""
    mask = ink_mask(arr, args.thr)
    mind = min(arr.shape)
    closed = ndimage.binary_dilation(mask, iterations=max(1, mind // 400))
    lbl, n = ndimage.label(closed)
    if n == 0:
        return []
    slices = ndimage.find_objects(lbl)
    boxes = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((i, (y0, x0, y1, x1)))
    if not boxes:
        return []
    # El anillo es el componente de bbox mas grande -> se descarta.
    ring = max(boxes, key=lambda b: (b[1][2] - b[1][0]) * (b[1][3] - b[1][1]))
    sign_boxes = []
    for i, b in boxes:
        if i == ring[0]:
            continue
        h, w = b[2] - b[0], b[3] - b[1]
        if min(h, w) < args.min_frac * mind:               # descarta motas
            continue
        sign_boxes.append(b)
    order = reading_order(sign_boxes)
    return [crop_square(img, sign_boxes[i], args.pad) for i in order]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Descompone una hoja-catalogo en recortes.")
    ap.add_argument("image", help="Ruta de la hoja (o de un sello para --level sign).")
    ap.add_argument("--out", default=None, help="Carpeta destino (def: data/refs/_unsorted/<nombre>).")
    ap.add_argument("--level", choices=["seal", "sign"], default="seal")
    ap.add_argument("--grid", default=None, help="Modo rejilla 'RxC', p.ej. 4x9.")
    ap.add_argument("--thr", type=int, default=None, help="Umbral fijo 0-255 (def: Otsu).")
    ap.add_argument("--dilate", type=int, default=0, help="Iteraciones de dilatacion (def: auto).")
    ap.add_argument("--min-frac", type=float, default=0.06, help="Lado minimo del sello / lado de la hoja.")
    ap.add_argument("--max-frac", type=float, default=0.95)
    ap.add_argument("--aspect", type=float, default=1.8, help="Relacion de aspecto maxima (circulo ~1).")
    ap.add_argument("--min-fill", type=float, default=0.45, help="Relleno minimo del bbox (disco lleno).")
    ap.add_argument("--pad", type=float, default=0.06, help="Margen alrededor del recorte.")
    ap.add_argument("--montage", action="store_true", help="Guarda un mosaico de revision.")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"No existe la imagen: {path}")
    arr, img = to_gray_array(path)

    if args.level == "sign":
        crops = detect_signs(arr, img, args)
        prefix = "sign"
    elif args.grid:
        r, c = (int(x) for x in args.grid.lower().split("x"))
        crops = split_grid(arr, img, r, c, args)
        prefix = "seal"
    else:
        crops = detect_seals(arr, img, args)
        prefix = "seal"

    out = Path(args.out) if args.out else (ROOT / "data" / "refs" / "_unsorted" / path.stem)
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(crops):
        c.save(out / f"{prefix}_{i:03d}.png")
    print(f"Detectados {len(crops)} recortes -> {out}")
    if args.montage and crops:
        m = out / "_montage.png"
        save_montage(crops, m)
        print(f"Mosaico de revision: {m}")
    if crops:
        print("\nSiguiente paso: renombra/mueve cada recorte a su clase, p.ej.")
        print(f"  {out}/{prefix}_000.png  ->  data/refs/nubes/nubes.png")
        print("Luego:  python augment_offline.py --per-class 80 --clean")
    else:
        print("\nSin detecciones. Prueba --grid RxC, baja --min-fill o ajusta --thr/--min-frac.")


if __name__ == "__main__":
    main()
