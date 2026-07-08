"""
Entrena un CLASIFICADOR DE SIGNOS (scikit-learn) sobre rasgos rotacionalmente
invariantes (sign_features.py), con datos aumentados y DEGRADADOS para cerrar la
brecha de dominio (plantilla limpia -> garabato escaneado a baja resolucion) que
hunde al template matching.

Datos: plantillas nuestras (data/signs) + limpias del fork (data/wha_symbols/signs),
mapeadas al slug canonico. Aumentos: rotacion 0-360 (NUNCA flip: espejar cambia el
significado), jitter de escala, y degradado (downscale+blur+ruido).

Evaluacion honesta:
  - CROSS-DOMAIN: entrena en un dominio, evalua en el otro (comparable al ~37% del
    template matcher).
  - vs TEMPLATE MATCHER: mismas entradas degradadas, clasificador entrenado vs IoU.

Guarda models/sign_clf.joblib (modelo + scaler + clases).

  python train_sign_clf.py
  python train_sign_clf.py --per-template 200
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from sign_features import features, FEATURE_DIM
from signs import canonical_slug, display_name
from decode_seal import load_binary
from import_wha_symbols import build_name_index, normalize

ROOT = Path(__file__).parent
SIGNS_DIR = ROOT / "data" / "signs"
WHA_SIGNS = ROOT / "data" / "wha_symbols" / "signs"
MODELS = ROOT / "models"
OUT = MODELS / "sign_clf.joblib"


# ---------------------------------------------------------------------------
# Plantillas base (bool mask) por slug y dominio
# ---------------------------------------------------------------------------

def load_templates():
    """Devuelve [(slug, domain, bool_mask), ...]."""
    out = []
    for d in sorted(p for p in SIGNS_DIR.iterdir() if p.is_dir()):
        png = d / f"{d.name}.png"
        if not png.exists():
            cand = list(d.glob("*.png"))
            if not cand:
                continue
            png = cand[0]
        out.append((canonical_slug(d.name), "ours", load_binary(png)))
    if WHA_SIGNS.exists():
        idx = build_name_index()
        for png in sorted(WHA_SIGNS.glob("*.png")):
            slug = idx.get(normalize(png.stem)) or normalize(png.stem).replace(" ", "_")
            out.append((canonical_slug(slug), "fork", load_binary(png)))
    return out


# ---------------------------------------------------------------------------
# Aumento + degradado
# ---------------------------------------------------------------------------

def mask_to_u8(mask):
    return np.where(mask, 0, 255).astype("uint8")   # tinta negra sobre blanco


def augment(mask, rng, degrade=True):
    """Rotacion (sin flip) + escala + degradado -> bool mask re-binarizada."""
    im = Image.fromarray(mask_to_u8(mask))
    # recorte al bbox para escalar consistente
    ys, xs = np.nonzero(mask)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = max(im.size)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    im = canvas.resize((64, 64), Image.BILINEAR)

    im = im.rotate(rng.uniform(0, 360), expand=True, fillcolor=255, resample=Image.BILINEAR)
    s = rng.uniform(0.8, 1.2)
    im = im.resize((max(8, int(im.size[0] * s)), max(8, int(im.size[1] * s))), Image.BILINEAR)

    if degrade:
        # baja resolucion agresiva (imita ~30 px por signo del catalogo)
        low = rng.randint(14, 30)
        im = im.resize((low, low), Image.BILINEAR).resize((48, 48), Image.BILINEAR)
        if rng.random() < 0.6:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    arr = np.asarray(im).astype(float)
    if degrade and rng.random() < 0.5:
        arr = arr + rng.normal(0, rng.uniform(4, 16), arr.shape)
    arr = np.clip(arr, 0, 255)
    # umbral otsu simple
    thr = arr.mean() - 0.35 * arr.std()
    return arr <= thr


def build_dataset(templates, per_template, seed=0, degrade=True):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    class R:
        uniform = staticmethod(lambda a, b: rng.uniform(a, b))
        randint = staticmethod(lambda a, b: rng.randint(a, b))
        random = staticmethod(lambda: rng.random())
        normal = staticmethod(lambda m, s, shp: nprng.normal(m, s, shp))

    X, y, dom = [], [], []
    for slug, domain, mask in templates:
        for _ in range(per_template):
            b = augment(mask, R, degrade=degrade)
            if b.sum() < 5:
                continue
            X.append(features(b)); y.append(slug); dom.append(domain)
    return np.array(X), np.array(y), np.array(dom)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ---------------------------------------------------------------------------

def topk_acc(clf, scaler, X, y, k=3):
    proba = clf.predict_proba(scaler.transform(X))
    classes = clf.classes_
    top = np.argsort(-proba, axis=1)[:, :k]
    top1 = np.mean([y[i] == classes[top[i, 0]] for i in range(len(y))])
    top3 = np.mean([y[i] in classes[top[i]] for i in range(len(y))])
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    templates = load_templates()
    slugs = sorted({t[0] for t in templates})
    print(f"Plantillas: {len(templates)} | clases (slug canonico): {len(slugs)} "
          f"| features/dim: {FEATURE_DIM}")

    print("\nGenerando dataset aumentado+degradado...")
    X, y, dom = build_dataset(templates, args.per_template, seed=args.seed)
    print(f"  muestras: {len(X)}  (ours={int((dom=='ours').sum())}, fork={int((dom=='fork').sum())})")

    def fit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=args.seed)
        clf.fit(sc.transform(Xtr), ytr)
        return clf, sc

    # --- CROSS-DOMAIN: entrenar en un dominio, evaluar en el otro ---
    print("\n=== CROSS-DOMAIN (comparable al ~37% del template matcher) ===")
    for tr, te in [("fork", "ours"), ("ours", "fork")]:
        mtr, mte = dom == tr, dom == te
        common = set(y[mtr]) & set(y[mte])
        sel_tr = mtr & np.isin(y, list(common))
        sel_te = mte & np.isin(y, list(common))
        clf, sc = fit(X[sel_tr], y[sel_tr])
        t1, t3 = topk_acc(clf, sc, X[sel_te], y[sel_te])
        print(f"  train={tr:<5} -> test={te:<5}  top-1={100*t1:.1f}%  top-3={100*t3:.1f}%  "
              f"(clases comunes={len(common)})")

    # --- Split held-out por muestra (mezcla ambos dominios) ---
    print("\n=== MIXTO (train/test 80/20, ambos dominios, degradado) ===")
    idx = np.arange(len(X)); np.random.default_rng(args.seed).shuffle(idx)
    cut = int(0.8 * len(idx))
    tr_i, te_i = idx[:cut], idx[cut:]
    clf, sc = fit(X[tr_i], y[tr_i])
    t1, t3 = topk_acc(clf, sc, X[te_i], y[te_i])
    print(f"  top-1={100*t1:.1f}%  top-3={100*t3:.1f}%")

    # --- Modelo desplegado: entrenar con TODO ---
    print("\nEntrenando modelo final con todo el dataset...")
    clf, sc = fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"clf": clf, "scaler": sc, "classes": list(clf.classes_)}, OUT)
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
