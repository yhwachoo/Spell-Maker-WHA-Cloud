"""
DECODIFICA un sello: detecta que signos contiene, en que posicion/tamano, y calcula
el VECTOR RESULTANTE del hechizo (usando signs.resultant_vector).

Matcher: TEMPLATE MATCHING ROTACIONAL.
  Cada simbolo se normaliza a un lienzo fijo. Para compararlo con una plantilla del
  vocabulario, se prueba rotando el simbolo en pasos y se mide el solapamiento (IoU)
  con tolerancia (mascaras engrosadas). El mejor IoU sobre rotaciones = parecido.
  Esto maneja la rotacion explicitamente y discrimina line-art mejor que Hu.

Pipeline:
  1) Vocabulario: normaliza cada plantilla data/signs/<slug>/<slug>.png
  2) Por sello: extrae simbolos interiores (analyze_symbols.extract_symbols)
  3) Matchea cada simbolo (IoU rotacional) -> signo + confianza
  4) Construye [{slug, angle, size}] -> resultant_vector()

Comandos:
  python decode_seal.py --validate
  python decode_seal.py --seal lanzallamas
  python decode_seal.py --all
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from signs import resultant_vector, display_name
from analyze_symbols import otsu, extract_symbols

ROOT = Path(__file__).parent
SIGNS_DIR = ROOT / "data" / "signs"
REFS_DIR = ROOT / "data" / "refs"

SHAPE = 40           # tamano de la forma normalizada
CANVAS = 56          # lienzo (con margen para rotar sin cortar)
DILATE = 1           # tolerancia de solapamiento (px)
MATCH_ANGLES = range(0, 360, 20)   # rotaciones que prueba el matcher
MIN_IOU = 0.18       # IoU minimo para aceptar un match en decode


# ---------------------------------------------------------------------------
# Normalizacion y solapamiento
# ---------------------------------------------------------------------------

def load_binary(path):
    arr = np.asarray(Image.open(path).convert("L"))
    return arr <= otsu(arr)


def norm_img(binary):
    """Recorta al bbox, cuadra, reescala a SHAPE y centra en lienzo CANVAS."""
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return None
    b = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape
    s = max(h, w)
    sq = np.zeros((s, s), bool)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = b
    im = Image.fromarray((sq * 255).astype("uint8")).resize((SHAPE, SHAPE), Image.BILINEAR)
    small = np.asarray(im) >= 110
    canvas = np.zeros((CANVAS, CANVAS), bool)
    off = (CANVAS - SHAPE) // 2
    canvas[off:off + SHAPE, off:off + SHAPE] = small
    return canvas


def dilate(b, it=DILATE):
    return ndimage.binary_dilation(b, iterations=it)


def rot(b, ang):
    return ndimage.rotate(b.astype(float), ang, reshape=False, order=1) >= 0.5


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


# ---------------------------------------------------------------------------
# Vocabulario y matcher
# ---------------------------------------------------------------------------

def load_vocabulary():
    """Devuelve dict {slug: dilated_norm_img}."""
    vocab = {}
    for d in sorted(p for p in SIGNS_DIR.iterdir() if p.is_dir()):
        png = d / f"{d.name}.png"
        if not png.exists():
            cand = list(d.glob("*.png"))
            if not cand:
                continue
            png = cand[0]
        nb = norm_img(load_binary(png))
        if nb is not None:
            vocab[d.name] = dilate(nb)
    return vocab


def match(query_bin, vocab, topk=3):
    """Matchea por IoU rotacional + similitud de densidad. Devuelve [(slug, score, conf), ...].

    El factor de densidad penaliza plantillas mucho mas (o menos) densas que el
    simbolo consultado, reduciendo el sesgo hacia signos densos (Marioneta, Ojo).
    """
    q = norm_img(query_bin)
    if q is None:
        return []
    q_rots = [dilate(rot(q, a)) for a in MATCH_ANGLES]
    q_den = float(q_rots[0].mean())
    scores = {}
    for slug, tmpl in vocab.items():
        best = max(iou(qr, tmpl) for qr in q_rots)
        t_den = float(tmpl.mean())
        dens_sim = min(q_den, t_den) / max(q_den, t_den, 1e-9)
        scores[slug] = best * (0.4 + 0.6 * dens_sim)   # densidad pesa 60%
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:topk]
    tot = sum(s for _, s in ranked) + 1e-9
    return [(slug, s, s / tot) for slug, s in ranked]


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------

def validate(angles=(0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)):
    vocab = load_vocabulary()
    slugs = list(vocab.keys())
    print(f"Vocabulario: {len(slugs)} signos\n")
    total = correct = top3 = 0
    confusions = {}
    for slug in slugs:
        base = load_binary(SIGNS_DIR / slug / f"{slug}.png")
        for ang in angles:
            test = ndimage.rotate(base.astype(float), ang, reshape=True, order=1) >= 0.5
            res = match(test, vocab, topk=3)
            if not res:
                continue
            total += 1
            names = [r[0] for r in res]
            if names[0] == slug:
                correct += 1
            else:
                confusions[(slug, names[0])] = confusions.get((slug, names[0]), 0) + 1
            if slug in names:
                top3 += 1
    print(f"Precision top-1: {correct}/{total} = {100*correct/max(total,1):.1f}%")
    print(f"Precision top-3: {top3}/{total} = {100*top3/max(total,1):.1f}%")
    print("\nConfusiones mas comunes (real -> predicho):")
    for (a, b), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {display_name(a):<20} -> {display_name(b):<20} x{c}")


# ---------------------------------------------------------------------------
# Decodificar
# ---------------------------------------------------------------------------

def decode_seal(seal_path, vocab, max_symbols=10, verbose=True):
    comps, ring = extract_symbols(seal_path, min_area=12, max_comp=max_symbols)
    if not comps:
        if verbose:
            print("  (no se extrajeron simbolos)")
        return None
    med = np.median([c["area"] for c in comps])
    detected = []
    for c in comps:
        res = match(c["bin"], vocab, topk=3)
        if not res or res[0][1] < MIN_IOU:
            continue
        slug, sc, conf = res[0]
        detected.append({
            "slug": slug, "angle": c["angle"],
            "size": round((c["area"] / med) ** 0.5, 2),
            "iou": sc, "alts": [r[0] for r in res[1:]],
        })
    if verbose:
        print(f"  Simbolos extraidos: {len(comps)} | matcheados (IoU>={MIN_IOU}): {len(detected)}")
        for dd in sorted(detected, key=lambda x: -x["size"]):
            print(f"    {display_name(dd['slug']):<18} "
                  f"pos={dd['angle']:>6.1f}deg size={dd['size']:.2f} "
                  f"IoU={dd['iou']:.2f}  alt: {', '.join(display_name(a) for a in dd['alts'])}")
    r = resultant_vector(detected)
    if verbose:
        print(f"\n  >> VECTOR DEL HECHIZO:")
        print(f"     Direccion : {r['compass']}")
        print(f"     Angulo    : {r['direction_deg']} deg   Magnitud: {r['magnitude']:.2f}")
        print(f"     Dispersion: {'SI' if r['is_radial'] else 'no'}"
              f"   Inward: {'SI' if r['has_inward'] else 'no'}"
              f"   Outward: {'SI' if r['has_outward'] else 'no'}")
    return {"detected": detected, "vector": r}


def main():
    ap = argparse.ArgumentParser(description="Decodifica un sello en signos + vector.")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seal", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.validate:
        validate()
        return

    vocab = load_vocabulary()
    if args.seal:
        png = REFS_DIR / args.seal / f"{args.seal}.png"
        if not png.exists():
            raise SystemExit(f"No existe {png}")
        print(f"\n=== {display_name(args.seal)} ===")
        decode_seal(png, vocab)
    elif args.all:
        for d in sorted(p for p in REFS_DIR.iterdir() if p.is_dir()):
            png = d / f"{d.name}.png"
            if png.exists():
                print(f"\n=== {display_name(d.name)} ===")
                decode_seal(png, vocab)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
