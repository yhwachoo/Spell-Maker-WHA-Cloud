"""
Descompone cada sello en sus SIMBOLOS, busca COINCIDENCIAS entre sellos (que signo
se repite en que hechizos) y reporta como se combinan.

Pipeline:
  1) Por cada sello (data/refs/<clase>/<clase>.png):
     - detecta el anillo (radio con maximo de tinta) y lo borra,
     - extrae los componentes interiores = simbolos (signos/sigilo).
  2) Describe cada simbolo con MOMENTOS DE HU (invariantes a rotacion/escala),
     porque el mismo signo aparece girado segun su posicion en el anillo.
  3) Agrupa los simbolos de TODOS los sellos (clustering) -> signos recurrentes.
  4) Reporta: cada cluster -> en que hechizos aparece; y co-ocurrencia.

Salida en data/symbols/:
  <clase>/sym_NN.png            simbolos individuales
  _montages/<clase>.png         montaje por sello
  _clusters/cluster_KK.png      montaje de cada signo recurrente (con su hechizo)
  report.txt                    coincidencias y co-ocurrencia

Uso:
  python analyze_symbols.py
  python analyze_symbols.py --min-area 12 --max-comp 14 --cluster-dist 2.5
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).parent
REFS = ROOT / "data" / "refs"
OUT = ROOT / "data" / "symbols"


def otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    tot = arr.size
    s_all = np.dot(np.arange(256), hist)
    sB = wB = 0.0
    best_t, best_v = 127, -1.0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = tot - wB
        if wF == 0:
            break
        sB += t * hist[t]
        mB, mF = sB / wB, (s_all - sB) / wF
        v = wB * wF * (mB - mF) ** 2
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def hu_log(binary):
    """7 momentos de Hu en escala log (invariantes a traslacion/escala/rotacion)."""
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return None
    xb, yb = xs.mean(), ys.mean()
    x, y = xs - xb, ys - yb
    m00 = float(len(xs))

    def mu(p, q):
        return np.sum((x ** p) * (y ** q))

    def eta(p, q):
        return mu(p, q) / (m00 ** (1 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    h = [0.0] * 7
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + \
           (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
           4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
           (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    out = []
    for v in h:
        out.append(-math.copysign(1.0, v) * math.log10(abs(v) + 1e-30))
    return np.array(out, dtype=np.float64)


def extract_symbols(path, min_area, max_comp):
    """Devuelve lista de dicts: {crop(PIL L), bin(np.bool), angle, dist, area}."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    H, W = arr.shape
    cy, cx = H / 2.0, W / 2.0
    ink = arr <= otsu(arr)

    # Radio del anillo: maximo de tinta en el rango exterior.
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = min(H, W) / 2.0
    rint = r[ink].astype(int)
    hist = np.bincount(rint, minlength=int(rmax) + 1)
    lo = int(0.6 * rmax)
    ring_R = lo + int(np.argmax(hist[lo:int(rmax) + 1])) if hist[lo:].size else int(0.9 * rmax)

    interior = ink.copy()
    interior[np.abs(r - ring_R) <= 3] = False     # borra el anillo
    interior[r > ring_R - 1] = False              # y todo lo de afuera

    dil = ndimage.binary_dilation(interior, iterations=1)
    lbl, n = ndimage.label(dil)
    if n == 0:
        return [], (cx, cy, ring_R)
    slices = ndimage.find_objects(lbl)
    comps = []
    for i, sl in enumerate(slices, 1):
        if sl is None:
            continue
        ys, ye = sl[0].start, sl[0].stop
        xs0, xe = sl[1].start, sl[1].stop
        compmask = (lbl[sl] == i) & interior[sl]
        area = int(compmask.sum())
        if area < min_area:
            continue
        ccx, ccy = (xs0 + xe) / 2.0, (ys + ye) / 2.0
        ang = math.degrees(math.atan2(-(ccy - cy), ccx - cx)) % 360
        dist = math.hypot(ccx - cx, ccy - cy) / max(ring_R, 1)
        crop = img.crop((xs0, ys, xe, ye))
        comps.append({"crop": crop, "bin": compmask, "area": area,
                      "angle": ang, "dist": round(dist, 2),
                      "bbox": (int(xs0), int(ys), int(xe), int(ye))})
    comps.sort(key=lambda c: -c["area"])
    return comps[:max_comp], (cx, cy, ring_R)


def montage(items, cell=64, cols=8, labels=None):
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(items):
        t = im.convert("RGB").resize((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(t, (x + 3, y + 3))
        d.rectangle([x, y, x + cell, y + cell], outline=(210, 210, 210))
        if labels:
            d.rectangle([x, y, x + cell, y + 11], fill=(255, 235, 0))
            d.text((x + 2, y + 1), str(labels[i])[:12], fill=(0, 0, 0))
    return sheet


def main():
    ap = argparse.ArgumentParser(description="Descompone sellos en simbolos y halla coincidencias.")
    ap.add_argument("--min-area", type=int, default=12, help="Area minima de un simbolo (px).")
    ap.add_argument("--max-comp", type=int, default=14, help="Maximo de simbolos por sello.")
    ap.add_argument("--cluster-dist", type=float, default=2.5, help="Umbral de distancia para agrupar.")
    args = ap.parse_args()

    seals = sorted(d for d in REFS.iterdir() if d.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_montages").mkdir(exist_ok=True)
    (OUT / "_clusters").mkdir(exist_ok=True)

    all_feats, all_meta, all_crops = [], [], []
    per_seal_counts = {}
    for sd in seals:
        cls = sd.name
        png = sd / f"{cls}.png"
        if not png.exists():
            cand = list(sd.glob("*.png"))
            if not cand:
                continue
            png = cand[0]
        comps, ring = extract_symbols(png, args.min_area, args.max_comp)
        per_seal_counts[cls] = len(comps)
        cdir = OUT / cls
        cdir.mkdir(exist_ok=True)
        seal_imgs = []
        for k, c in enumerate(comps):
            c["crop"].save(cdir / f"sym_{k:02d}.png")
            seal_imgs.append(c["crop"])
            f = hu_log(c["bin"])
            if f is None:
                continue
            all_feats.append(f)
            all_meta.append({"cls": cls, "k": k, "angle": c["angle"], "dist": c["dist"]})
            all_crops.append(c["crop"])
        m = montage(seal_imgs)
        if m:
            m.save(OUT / "_montages" / f"{cls}.png")

    print(f"Sellos procesados: {len(seals)} | simbolos extraidos: {len(all_crops)}")
    if len(all_feats) < 4:
        print("Muy pocos simbolos para agrupar."); return

    X = np.vstack(all_feats)
    # Estandariza cada dimension de Hu.
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=args.cluster_dist,
                                         linkage="average")
    labels = clustering.fit_predict(X)
    n_clusters = labels.max() + 1

    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)

    # Reporta solo clusters que aparecen en >=2 hechizos distintos = "coincidencias".
    report = []
    recurring = []
    for lab, idxs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        spells = sorted({all_meta[i]["cls"] for i in idxs})
        if len(spells) >= 2:
            recurring.append((lab, idxs, spells))
    print(f"Clusters totales: {n_clusters} | signos recurrentes (en >=2 hechizos): {len(recurring)}")

    report.append("=== SIGNOS RECURRENTES (mismo simbolo en >=2 hechizos) ===\n")
    for rank, (lab, idxs, spells) in enumerate(recurring):
        crops = [all_crops[i] for i in idxs]
        labs = [all_meta[i]["cls"] for i in idxs]
        m = montage(crops, labels=labs)
        if m:
            m.save(OUT / "_clusters" / f"cluster_{rank:02d}.png")
        report.append(f"[cluster {rank:02d}]  {len(idxs)} simbolos en {len(spells)} hechizos:")
        report.append("   " + ", ".join(spells) + "\n")

    # Co-ocurrencia: cuantos signos recurrentes comparten cada par de hechizos.
    pair = defaultdict(int)
    for _, idxs, _ in recurring:
        spells = sorted({all_meta[i]["cls"] for i in idxs})
        for a in range(len(spells)):
            for b in range(a + 1, len(spells)):
                pair[(spells[a], spells[b])] += 1
    report.append("=== HECHIZOS QUE MAS SIGNOS COMPARTEN ===")
    for (a, b), c in sorted(pair.items(), key=lambda kv: -kv[1])[:25]:
        report.append(f"   {c:2d} signos en comun:  {a}  <->  {b}")

    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nReporte: {OUT/'report.txt'}")
    print(f"Montajes por sello: {OUT/'_montages'}  | clusters: {OUT/'_clusters'}")


if __name__ == "__main__":
    main()
