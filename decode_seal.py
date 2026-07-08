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

from signs import resultant_vector, display_name, canonical_slug
from analyze_symbols import otsu, extract_symbols
from import_wha_symbols import build_name_index, normalize

ROOT = Path(__file__).parent
SIGNS_DIR = ROOT / "data" / "signs"
REFS_DIR = ROOT / "data" / "refs"
WHA_SIGNS_DIR = ROOT / "data" / "wha_symbols" / "signs"

SHAPE = 40           # tamano de la forma normalizada
CANVAS = 56          # lienzo (con margen para rotar sin cortar)
DILATE = 1           # tolerancia de solapamiento (px)
MATCH_ANGLES = range(0, 360, 20)   # rotaciones que prueba el matcher
MIN_IOU = 0.18       # IoU minimo para aceptar un match en decode


# ---------------------------------------------------------------------------
# Normalizacion y solapamiento
# ---------------------------------------------------------------------------

def load_binary(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        # Aplana el alfa sobre blanco (los PNG del fork son trazo/transparente).
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    arr = np.asarray(im.convert("L"))
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

def load_vocabulary(include_wha=True, include_ours=True, canon=True):
    """Devuelve dict {slug: [dilated_norm_img, ...]} (varias plantillas por slug).

    Combina nuestras plantillas dibujadas a mano (data/signs/) con las plantillas
    limpias del fork (data/wha_symbols/signs/), mapeadas al mismo slug via nombre
    canonico. Los signos del fork sin equivalente en signs.py entran como slug
    nuevo (nombre normalizado). El matcher toma el MEJOR parecido entre todas las
    plantillas de un slug, asi que sumar plantillas solo puede ayudar la cobertura.

    canon=True fusiona los sinonimos ingles/espanol (Enlarge==Agrandar, etc.) en un
    solo slug canonico, evitando clases duplicadas que compiten entre si.
    """
    vocab: dict = {}

    def add(slug, png):
        if canon:
            slug = canonical_slug(slug)
        nb = norm_img(load_binary(png))
        if nb is not None:
            vocab.setdefault(slug, []).append(dilate(nb))

    if include_ours:
        for d in sorted(p for p in SIGNS_DIR.iterdir() if p.is_dir()):
            png = d / f"{d.name}.png"
            if not png.exists():
                cand = list(d.glob("*.png"))
                if not cand:
                    continue
                png = cand[0]
            add(d.name, png)

    if include_wha and WHA_SIGNS_DIR.exists():
        idx = build_name_index()
        for png in sorted(WHA_SIGNS_DIR.glob("*.png")):
            slug = idx.get(normalize(png.stem)) or normalize(png.stem).replace(" ", "_")
            add(slug, png)

    return vocab


def match(query_bin, vocab, topk=3):
    """Matchea por IoU rotacional + similitud de densidad. Devuelve [(slug, score, conf), ...].

    Cada slug puede tener varias plantillas; se toma la mejor puntuacion sobre todas
    (plantillas x rotaciones). El factor de densidad penaliza plantillas mucho mas (o
    menos) densas que el simbolo consultado, reduciendo el sesgo hacia signos densos.
    """
    q = norm_img(query_bin)
    if q is None:
        return []
    q_rots = [dilate(rot(q, a)) for a in MATCH_ANGLES]
    q_den = float(q_rots[0].mean())
    scores = {}
    for slug, tmpls in vocab.items():
        best = 0.0
        for tmpl in tmpls:
            iou_best = max(iou(qr, tmpl) for qr in q_rots)
            t_den = float(tmpl.mean())
            dens_sim = min(q_den, t_den) / max(q_den, t_den, 1e-9)
            best = max(best, iou_best * (0.4 + 0.6 * dens_sim))   # densidad pesa 60%
        scores[slug] = best
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:topk]
    tot = sum(s for _, s in ranked) + 1e-9
    return [(slug, s, s / tot) for slug, s in ranked]


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------

def _query_sources(which):
    """Devuelve [(slug, base_binary), ...] para usar como consultas de validacion.

    which='ours' -> plantillas dibujadas a mano (data/signs)
    which='wha'  -> plantillas limpias del fork (data/wha_symbols/signs)
    """
    out = []
    if which == "ours":
        for d in sorted(p for p in SIGNS_DIR.iterdir() if p.is_dir()):
            png = d / f"{d.name}.png"
            if not png.exists():
                cand = list(d.glob("*.png"))
                if not cand:
                    continue
                png = cand[0]
            out.append((d.name, load_binary(png)))
    elif which == "wha":
        idx = build_name_index()
        for png in sorted(WHA_SIGNS_DIR.glob("*.png")):
            slug = idx.get(normalize(png.stem)) or normalize(png.stem).replace(" ", "_")
            out.append((slug, load_binary(png)))
    return out


def _run_validation(queries, vocab,
                    angles=(0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)):
    total = correct = top3 = 0
    confusions = {}
    for slug, base in queries:
        slug = canonical_slug(slug)
        if slug not in vocab:
            continue
        for ang in angles:
            test = ndimage.rotate(base.astype(float), ang, reshape=True, order=1) >= 0.5
            res = match(test, vocab, topk=3)
            if not res:
                continue
            total += 1
            names = [canonical_slug(r[0]) for r in res]
            if names[0] == slug:
                correct += 1
            else:
                confusions[(slug, names[0])] = confusions.get((slug, names[0]), 0) + 1
            if slug in names:
                top3 += 1
    return total, correct, top3, confusions


def _print_result(title, total, correct, top3, confusions, n_conf=10):
    print(f"\n--- {title} ---")
    print(f"  Precision top-1: {correct}/{total} = {100*correct/max(total,1):.1f}%")
    print(f"  Precision top-3: {top3}/{total} = {100*top3/max(total,1):.1f}%")
    if confusions:
        print("  Confusiones (real -> predicho):")
        for (a, b), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:n_conf]:
            print(f"    {display_name(a):<20} -> {display_name(b):<20} x{c}")


def validate():
    """Compara la fiabilidad del matcher antes/despues de sumar las plantillas del fork."""
    vocab_ours = load_vocabulary(include_wha=False)
    vocab_all = load_vocabulary(include_wha=True)
    q_ours = _query_sources("ours")
    q_wha = _query_sources("wha") if WHA_SIGNS_DIR.exists() else []

    print(f"Vocabulario base (solo nuestro): {len(vocab_ours)} slugs")
    print(f"Vocabulario ampliado (+fork):    {len(vocab_all)} slugs")

    # 1) BASELINE: consultas nuestras vs vocabulario nuestro (lo que reportaba antes)
    r = _run_validation(q_ours, vocab_ours)
    _print_result("BASELINE  (query: nuestro | vocab: nuestro)", *r)

    if q_wha:
        # 2) CROSS-DOMAIN: consulta = plantilla limpia del fork, vocab = dibujos a mano.
        #    Mide si el matcher reconoce un signo limpio con solo el garabato como referencia.
        r = _run_validation(q_wha, vocab_ours)
        _print_result("CROSS  (query: fork | vocab: nuestro)  <- transferencia de dominio", *r)

        # 3) CROSS inverso: consulta = garabato, vocab = plantillas limpias del fork.
        r = _run_validation(q_ours, vocab_all)
        _print_result("AMPLIADO  (query: nuestro | vocab: nuestro+fork)", *r)

        # 4) Self-consistencia del set limpio del fork (techo del matcher con buen line-art).
        r = _run_validation(q_wha, vocab_all)
        _print_result("FORK-SELF  (query: fork | vocab: nuestro+fork)", *r)


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
