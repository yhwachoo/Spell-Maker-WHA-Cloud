"""
Analiza un SELLO y estima potencia, balance y direccion segun la mecanica del lore
(ver MECANICA.md). NO entrena ni necesita etiquetas: traduce las reglas a geometria.

Reglas implementadas:
  - "Mas grande/mas tinta = mas potente"      -> cobertura de tinta x tamano.
  - "El lado con signos mas grandes desvia"   -> offset del centroide de los signos
                                                  interiores respecto al centro -> direccion.
  - "Simetria bilateral = estable y recto"     -> diferencia izq/der y arriba/abajo.

Uso:
    python analyze_seal.py sello.png
    python analyze_seal.py carpeta/ --annotate     # guarda imagen con centro/flecha
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def otsu_threshold(arr):
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
        mB, mF = sumB / wB, (sum_all - sumB) / wF
        v = wB * wF * (mB - mF) ** 2
        if v > best_var:
            best_var, best_t = v, t
    return best_t


def compass(angle_deg):
    """Convierte angulo matematico (0=derecha, 90=arriba) a punto cardinal en espanol."""
    dirs = ["derecha", "arriba-derecha", "arriba", "arriba-izquierda",
            "izquierda", "abajo-izquierda", "abajo", "abajo-derecha"]
    idx = int(((angle_deg % 360) + 22.5) // 45) % 8
    return dirs[idx]


def analyze(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    H, W = arr.shape
    thr = otsu_threshold(arr)
    ink = arr <= thr

    ys, xs = np.where(ink)
    if len(xs) == 0:
        return None

    # Centro y radio del sello a partir del bbox de la tinta (el anillo lo domina).
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    R = max((min(y1 - y0, x1 - x0) / 2), 1.0)

    # Distancia de cada pixel de tinta al centro.
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # 1) POTENCIA: cobertura del disco x cuanto llena el sello el encuadre.
    disk_area = math.pi * R * R
    coverage = min(len(xs) / disk_area, 1.0)          # densidad de trazo (0..1)
    size_factor = (2 * R) / min(H, W)                 # cuanto ocupa el sello (0..1)
    potencia = round(coverage * size_factor, 3)

    # 2) DIRECCION: centroide de los SIGNOS INTERIORES (excluye el anillo, r>0.85R).
    interior = dist < 0.85 * R
    if interior.sum() >= 5:
        ix, iy = xs[interior], ys[interior]
        mx, my = ix.mean(), iy.mean()
        off_x, off_y = mx - cx, my - cy
        desvio = math.hypot(off_x, off_y) / R          # 0 = centrado/equilibrado
        angle = math.degrees(math.atan2(-off_y, off_x))  # -off_y: y crece hacia abajo
    else:
        off_x = off_y = 0.0
        desvio, angle = 0.0, 0.0

    # 3) SIMETRIA (sobre signos interiores).
    if interior.sum() >= 5:
        ix, iy = xs[interior], ys[interior]
        left = (ix < cx).sum(); right = (ix >= cx).sum()
        top = (iy < cy).sum(); bot = (iy >= cy).sum()
        sym_lr = 1 - abs(left - right) / max(left + right, 1)
        sym_tb = 1 - abs(top - bot) / max(top + bot, 1)
    else:
        sym_lr = sym_tb = 1.0

    estable = desvio < 0.08 and sym_lr > 0.85
    return {
        "size_px": int(2 * R), "potencia": potencia, "cobertura": round(coverage, 3),
        "desvio": round(desvio, 3), "angulo": round(angle, 1),
        "direccion": compass(angle) if desvio >= 0.08 else "centrado (recto)",
        "sim_izq_der": round(sym_lr, 3), "sim_arr_abj": round(sym_tb, 3),
        "estable": estable, "center": (cx, cy), "offset": (off_x, off_y), "R": R,
    }


def annotate(path, res, out_path):
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    cx, cy = res["center"]; R = res["R"]
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 120, 255))   # centro
    ox, oy = res["offset"]
    scale = (0.9 * R) / max(math.hypot(ox, oy), 1e-6)
    d.line([cx, cy, cx + ox * scale, cy + oy * scale], fill=(255, 0, 0), width=3)  # desvio
    img.save(out_path)


def fmt(name, res):
    if res is None:
        return f"{name}: sin tinta detectada."
    estado = "ESTABLE (sale recto)" if res["estable"] else f"SE DESVIA hacia {res['direccion']}"
    return (f"{name}\n"
            f"  Potencia relativa : {res['potencia']:.3f}  (cobertura {res['cobertura']:.3f}, "
            f"tamano {res['size_px']} px)\n"
            f"  Direccion         : {estado}  (desvio {res['desvio']:.3f}, angulo {res['angulo']:.0f}o)\n"
            f"  Simetria izq/der  : {res['sim_izq_der']:.3f}   arriba/abajo: {res['sim_arr_abj']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Estima potencia/direccion/simetria de un sello.")
    ap.add_argument("input", help="Imagen de un sello o carpeta de sellos.")
    ap.add_argument("--annotate", action="store_true", help="Guarda <img>_annot.png con centro y flecha.")
    args = ap.parse_args()

    p = Path(args.input)
    paths = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    for path in paths:
        res = analyze(path)
        print("\n" + fmt(path.name, res))
        if args.annotate and res:
            out = path.with_name(path.stem + "_annot.png")
            annotate(path, res, out)
            print(f"  Anotada: {out.name}")


if __name__ == "__main__":
    main()
