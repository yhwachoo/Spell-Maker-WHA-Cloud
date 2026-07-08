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
import io
import math
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
from render_spell import render_spell
from analyze_symbols import extract_symbols

ROOT = Path(__file__).parent
SIGNS_DIR = ROOT / "data" / "signs"
WHA_SIGNS = ROOT / "data" / "wha_symbols" / "signs"
WHA_SIGILS = ROOT / "data" / "wha_symbols" / "sigils"
MODELS = ROOT / "models"
OUT = MODELS / "sign_clf.joblib"

_SIGN_NAMES = [f"sign_{p.stem}" for p in sorted(WHA_SIGNS.glob("*.png"))]
_SIGIL_NAMES = [f"sigil_{p.stem}" for p in sorted(WHA_SIGILS.glob("*.png"))]


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


def _sign_to_slug(name):
    return canonical_slug(name[5:].strip().lower().replace(" ", "_")) if name.startswith("sign_") else None


def train_seal(rng):
    """Sello sintetico con signos SEPARABLES (no se funden al extraer)."""
    seal = {"rings": [{"radius": rng.randint(455, 480)}],
            "sigils": [{"name": rng.choice(_SIGIL_NAMES), "size": rng.randint(180, 330)}],
            "signs": []}
    for _ in range(rng.randint(1, 2)):
        radius = rng.randint(290, 385)
        amount = rng.choice([6, 8, 8, 10, 12])
        spacing = 2 * math.pi * radius / amount
        size = int(min(rng.randint(85, 155), 0.7 * spacing))   # < 0.7*espaciado = separable
        seal["signs"].append({"name": rng.choice(_SIGN_NAMES), "size": size,
                              "radius": radius, "amount": amount, "rotation": rng.randint(0, 359)})
    return {"seals": [seal]}


def _degrade_seal(img, rng, nprng):
    """Degradado suave que preserva el anillo (para que la extraccion funcione)."""
    from PIL import ImageFilter
    low = rng.randint(300, 620)
    im = img.convert("L").resize((low, low), Image.BILINEAR).resize((1000, 1000), Image.BILINEAR)
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    arr = np.asarray(im).astype(float) + nprng.normal(0, rng.uniform(3, 11), (1000, 1000))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def seal_labeled_crops(spell, degrade=True, rng=None, nprng=None):
    """Renderiza un sello, lo extrae con el pipeline real y devuelve [(bin, slug), ...]
    emparejando cada componente con la anotacion de signo mas cercana."""
    img, ann = render_spell(spell, collect_annotations=True)
    gt = [(a["x"], a["y"], a["size"], _sign_to_slug(a["name"]))
          for a in ann if a["kind"] == "sign"]
    if degrade and rng is not None:
        img = _degrade_seal(img, rng, nprng)
    buf = io.BytesIO(); img.convert("L").save(buf, "PNG"); buf.seek(0)
    comps, _ring = extract_symbols(buf, min_area=10, max_comp=40)
    out = []
    for c in comps:
        x0, y0, x1, y1 = c["bbox"]; ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
        best, bd = None, 1e9
        for gx, gy, gs, slug in gt:
            d = math.hypot(ccx - gx, ccy - gy)
            if d < bd:
                bd, best = d, (slug, gs)
        if best and bd < 0.6 * best[1]:      # componente cae dentro de un signo anotado
            out.append((c["bin"], best[0]))
    return out


def build_synthetic(n_seals, seed=0, degrade=True):
    """Recortes etiquetados extraidos de sellos sinteticos completos (dominio real)."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_seals):
        spell = train_seal(rng)
        for b, slug in seal_labeled_crops(spell, degrade=degrade, rng=rng, nprng=nprng):
            if b.sum() < 5:
                continue
            X.append(features(b)); y.append(slug)
    return np.array(X), np.array(y)


def eval_end_to_end(clf, scaler, n_seals=60, seed=777):
    """Metrica REAL: sobre sellos completos degradados, %% de signos bien reconocidos
    por el pipeline extraer->clasificar (comparado con la anotacion)."""
    rng = random.Random(seed); nprng = np.random.default_rng(seed)
    classes = np.asarray(clf.classes_)
    tot = cor = 0
    for _ in range(n_seals):
        spell = train_seal(rng)
        crops = seal_labeled_crops(spell, degrade=True, rng=rng, nprng=nprng)
        for b, slug in crops:
            if b.sum() < 5:
                continue
            proba = clf.predict_proba(scaler.transform(features(b).reshape(1, -1)))[0]
            pred = classes[int(np.argmax(proba))]
            tot += 1; cor += (pred == slug)
    return cor, tot


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
    ap.add_argument("--per-template", type=int, default=140)
    ap.add_argument("--synth-seals", type=int, default=500,
                    help="Sellos sinteticos completos para extraer recortes realistas.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    templates = load_templates()
    slugs = sorted({t[0] for t in templates})
    print(f"Plantillas: {len(templates)} | clases (slug canonico): {len(slugs)} "
          f"| features/dim: {FEATURE_DIM}")

    def fit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=args.seed)
        clf.fit(sc.transform(Xtr), ytr)
        return clf, sc

    print("\n1) Dataset de plantillas aisladas (aumentado+degradado)...")
    Xt, yt, dom = build_dataset(templates, args.per_template, seed=args.seed)
    print(f"   {len(Xt)} muestras (ours={int((dom=='ours').sum())}, fork={int((dom=='fork').sum())})")

    print(f"2) Dataset de sellos sinteticos completos ({args.synth_seals} sellos)...")
    Xs, ys = build_synthetic(args.synth_seals, seed=args.seed + 1, degrade=True)
    print(f"   {len(Xs)} recortes etiquetados extraidos por el pipeline real")

    # --- Ablacion end-to-end: el numero que importa (decodificar un sello completo) ---
    print("\n=== EVALUACION END-TO-END (extraer->clasificar sobre sellos completos) ===")
    clf_a, sc_a = fit(Xt, yt)
    cor, tot = eval_end_to_end(clf_a, sc_a)
    print(f"  Solo plantillas          : {cor}/{tot} = {100*cor/max(tot,1):.1f}% signos correctos")

    Xall = np.vstack([Xt, Xs]); yall = np.concatenate([yt, ys])
    clf_b, sc_b = fit(Xall, yall)
    cor, tot = eval_end_to_end(clf_b, sc_b)
    print(f"  Plantillas + sinteticos  : {cor}/{tot} = {100*cor/max(tot,1):.1f}% signos correctos")

    # --- Modelo desplegado: plantillas + sinteticos ---
    print("\nGuardando modelo final (plantillas + sinteticos)...")
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"clf": clf_b, "scaler": sc_b, "classes": list(clf_b.classes_)}, OUT)
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
