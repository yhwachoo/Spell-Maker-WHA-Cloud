"""
RENDERIZADOR de hechizos en el formato spell.json del fork wha-spell-maker.

Porta la geometria de dibujo de `src/sketch.ts` (p5.js) a Pillow: pila de
transformaciones (translate/rotate/scale), simetria radial de signos
(amount / rotation / radius / offsetStrafe), sigilo central y anillos con
apertura. Sirve para:

  (a) VALIDAR nuestro pipeline reproduciendo los examples/*.json del fork y
      comparandolos con sus PNG de referencia, y
  (b) GENERAR sellos sinteticos con anotaciones perfectas (posicion/tamano/
      rotacion/identidad de cada signo) para entrenar un detector real.

Los simbolos se cargan de data/wha_symbols/ (line-art negro sobre blanco); el
alfa se reconstruye del propio trazo, asi que se pueden teñir.

  python render_spell.py --spell examples/"Sylph Shoes.json" --out /tmp/sylph.png
  python render_spell.py --validate-examples --examples-dir /workspace/spell-maker-wha/examples

GPL v3: los simbolos derivan del proyecto de DaviAMSilva (ver data/wha_symbols/).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
WHA = ROOT / "data" / "wha_symbols"

CANVAS = 1000
CENTER = 500.0
GLOBAL_RING_OFFSET = 270.0   # sketch.ts: defaults.globalRingOffsetAngle

# Defaults tomados de src/schemas/spell.json
D_SEAL = dict(visible=True, angle=0, scale=100, offsetX=0, offsetY=0)
D_RING = dict(visible=True, color="#000000", filled=False, fillColor="#ffffff",
              radius=450, weight=10, openingSize=0, openingAngle=0, offsetX=0, offsetY=0)
D_SIGIL = dict(visible=True, tinted=False, color="#000000", size=400, angle=0,
               offsetX=0, offsetY=0)
D_SIGN = dict(visible=True, tinted=False, color="#000000", size=200, angle=0,
              offsetStrafe=0, radius=325, amount=8, amountSkip=0, rotation=0,
              offsetX=0, offsetY=0)

CAT_BY_PREFIX = {"sign": "signs", "sigil": "sigils", "shape": "shapes",
                 "forbidden": "forbiddens", "toh": "tohs"}


# ---------------------------------------------------------------------------
# Pila de transformaciones (equivalente a push/pop/translate/rotate/scale de p5)
# ---------------------------------------------------------------------------

class TransformStack:
    def __init__(self):
        self.m = np.eye(3)
        self.stack = []

    def push(self):
        self.stack.append(self.m.copy())

    def pop(self):
        self.m = self.stack.pop()

    def translate(self, tx, ty):
        t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)
        self.m = self.m @ t

    def rotate(self, deg):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        self.m = self.m @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

    def scale(self, sx, sy=None):
        sy = sx if sy is None else sy
        self.m = self.m @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)

    def origin(self):
        p = self.m @ np.array([0, 0, 1.0])
        return p[0], p[1]

    def net_scale(self):
        return math.hypot(self.m[0, 0], self.m[1, 0])

    def net_rotation_deg(self):
        return math.degrees(math.atan2(self.m[1, 0], self.m[0, 0]))


# ---------------------------------------------------------------------------
# Carga y teñido de simbolos
# ---------------------------------------------------------------------------

_CACHE = {}


def load_symbol_rgba(name, color="#000000"):
    """name='sign_Convergence' -> RGBA teñido de `color`, alfa reconstruido del trazo."""
    key = (name, color)
    if key in _CACHE:
        return _CACHE[key]
    prefix, _, base = name.partition("_")
    cat = CAT_BY_PREFIX.get(prefix)
    if not cat:
        return None
    png = WHA / cat / f"{base}.png"
    if not png.exists():
        return None
    src = np.asarray(Image.open(png).convert("RGBA"))
    alpha = src[..., 3]   # los PNG del fork ya traen alfa real (trazo opaco)
    r = int(color[1:3], 16); g = int(color[3:5], 16); b = int(color[5:7], 16)
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), "uint8")
    rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
    rgba[..., 3] = alpha
    img = Image.fromarray(rgba, "RGBA")
    _CACHE[key] = img
    return img


def _get(obj, key, defaults):
    v = obj.get(key)
    return defaults[key] if v is None else v


def stamp_symbol(canvas, ts, sym_name, size, color):
    """Dibuja un simbolo en la posicion/rotacion/escala actual de la pila."""
    img = load_symbol_rgba(sym_name, color)
    if img is None:
        return None
    s = ts.net_scale()
    rot = ts.net_rotation_deg()
    ox, oy = ts.origin()
    px = max(1, int(round(size * s)))
    im = img.resize((px, px), Image.BILINEAR)
    # p5 rota en sentido horario (y hacia abajo) para angulo positivo;
    # PIL.rotate es antihorario -> usamos -rot.
    im = im.rotate(-rot, expand=True, resample=Image.BILINEAR)
    w, h = im.size
    canvas.alpha_composite(im, (int(round(ox - w / 2)), int(round(oy - h / 2))))
    return (ox, oy, size * s, rot)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render_spell(spell, collect_annotations=False):
    bg_on = spell.get("background", True)
    bg_color = spell.get("backgroundColor", "#ffffff")
    if bg_on:
        canvas = Image.new("RGBA", (CANVAS, CANVAS), bg_color)
    else:
        canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))

    from PIL import ImageDraw
    ann = []

    for seal in spell.get("seals", []):
        if not _get(seal, "visible", D_SEAL):
            continue
        ts = TransformStack()
        ts.push()
        ts.translate(CENTER + _get(seal, "offsetX", D_SEAL),
                     CENTER + _get(seal, "offsetY", D_SEAL))
        ts.scale(_get(seal, "scale", D_SEAL) / 100.0)
        ts.rotate(_get(seal, "angle", D_SEAL))

        # --- Rellenos de anillo (detras de los simbolos) ---
        for ring in seal.get("rings", []):
            if not _get(ring, "visible", D_RING) or not _get(ring, "filled", D_RING):
                continue
            ts.push()
            ts.translate(_get(ring, "offsetX", D_RING), _get(ring, "offsetY", D_RING))
            cx, cy = ts.origin()
            rad = _get(ring, "radius", D_RING) * ts.net_scale()
            d = ImageDraw.Draw(canvas)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      fill=_get(ring, "fillColor", D_RING))
            ts.pop()

        # --- Sigilos y signos ---
        for sigil in seal.get("sigils", []):
            if not _get(sigil, "visible", D_SIGIL):
                continue
            color = _get(sigil, "color", D_SIGIL) if _get(sigil, "tinted", D_SIGIL) else "#000000"
            ts.push()
            ts.translate(_get(sigil, "offsetX", D_SIGIL), _get(sigil, "offsetY", D_SIGIL))
            ts.rotate(_get(sigil, "angle", D_SIGIL))
            info = stamp_symbol(canvas, ts, sigil["name"], _get(sigil, "size", D_SIGIL), color)
            ts.pop()
            if collect_annotations and info:
                ann.append(dict(kind="sigil", name=sigil["name"],
                                x=info[0], y=info[1], size=info[2], rot=info[3]))

        for sign in seal.get("signs", []):
            if not _get(sign, "visible", D_SIGN):
                continue
            color = _get(sign, "color", D_SIGN) if _get(sign, "tinted", D_SIGN) else "#000000"
            amount = _get(sign, "amount", D_SIGN)
            draw_n = amount - _get(sign, "amountSkip", D_SIGN)
            for i in range(int(draw_n)):
                ts.push()
                ts.translate(_get(sign, "offsetX", D_SIGN), _get(sign, "offsetY", D_SIGN))
                ts.rotate(_get(sign, "rotation", D_SIGN) + i * 360.0 / amount)
                ts.translate(0, -_get(sign, "radius", D_SIGN))
                ts.translate(_get(sign, "offsetStrafe", D_SIGN), 0)
                ts.rotate(_get(sign, "angle", D_SIGN))
                info = stamp_symbol(canvas, ts, sign["name"], _get(sign, "size", D_SIGN), color)
                ts.pop()
                if collect_annotations and info:
                    ann.append(dict(kind="sign", name=sign["name"],
                                    x=info[0], y=info[1], size=info[2], rot=info[3]))

        # --- Trazos de anillo (encima) ---
        for ring in seal.get("rings", []):
            if not _get(ring, "visible", D_RING):
                continue
            ts.push()
            ts.translate(_get(ring, "offsetX", D_RING), _get(ring, "offsetY", D_RING))
            cx, cy = ts.origin()
            scale = ts.net_scale()
            rad = _get(ring, "radius", D_RING) * scale
            weight = max(1, int(round(_get(ring, "weight", D_RING) * scale)))
            opening = _get(ring, "openingSize", D_RING)
            d = ImageDraw.Draw(canvas)
            box = [cx - rad, cy - rad, cx + rad, cy + rad]
            if opening <= 0:
                d.ellipse(box, outline=_get(ring, "color", D_RING), width=weight)
            else:
                base = (GLOBAL_RING_OFFSET + _get(ring, "openingAngle", D_RING)
                        + ts.net_rotation_deg())
                start = (base + opening / 2) % 360
                end = (base - opening / 2) % 360 + 360
                d.arc(box, start, end, fill=_get(ring, "color", D_RING), width=weight)
            ts.pop()

        ts.pop()

    return (canvas.convert("RGB"), ann) if collect_annotations else canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Validacion contra los PNG de referencia del fork
# ---------------------------------------------------------------------------

def compare_images(a: Image.Image, b: Image.Image):
    """IoU de tinta (pixeles oscuros) entre dos renders, a igual tamano."""
    b = b.resize(a.size)
    aa = np.asarray(a.convert("L")) < 128
    bb = np.asarray(b.convert("L")) < 128
    u = (aa | bb).sum()
    return float((aa & bb).sum() / u) if u else 0.0


def validate_examples(examples_dir):
    examples_dir = Path(examples_dir)
    jsons = sorted(examples_dir.glob("*.json"))
    if not jsons:
        raise SystemExit(f"No hay JSON en {examples_dir}")
    print(f"Validando {len(jsons)} ejemplos contra sus PNG de referencia:\n")
    ious = []
    for jf in jsons:
        spell = json.loads(jf.read_text())
        render = render_spell(spell)
        ref_png = jf.with_suffix(".png")
        if ref_png.exists():
            ref = Image.open(ref_png)
            iou = compare_images(render, ref)
            ious.append(iou)
            print(f"  {jf.stem:<24} IoU tinta vs referencia = {iou:.3f}")
        else:
            print(f"  {jf.stem:<24} (sin PNG de referencia)")
    if ious:
        print(f"\n  IoU medio: {sum(ious)/len(ious):.3f}")


# ---------------------------------------------------------------------------
# Generador de datos sinteticos con anotaciones perfectas
# ---------------------------------------------------------------------------

def _available_signs():
    """Nombres de signo disponibles como plantilla (formato spell.json)."""
    return [f"sign_{p.stem}" for p in sorted((WHA / "signs").glob("*.png"))]


def _available_sigils():
    return [f"sigil_{p.stem}" for p in sorted((WHA / "sigils").glob("*.png"))]


def random_spell(rng):
    """Un sello aleatorio: sigilo central + 1-3 anillos de signos en simetria radial."""
    signs = _available_signs()
    sigils = _available_sigils()
    n_rings = rng.randint(1, 3)
    seal = {"rings": [{"radius": rng.choice([420, 450, 470])}],
            "sigils": [{"name": rng.choice(sigils), "size": rng.randint(250, 520)}],
            "signs": []}
    for _ in range(n_rings):
        seal["signs"].append({
            "name": rng.choice(signs),
            "size": rng.randint(120, 240),
            "radius": rng.randint(250, 380),
            "amount": rng.choice([4, 5, 6, 8, 8, 10, 12]),
            "rotation": rng.randint(0, 359),
        })
    return {"seals": [seal]}


def degrade(img, rng):
    """Aproxima el dominio 'garabato escaneado a baja resolucion' del catalogo."""
    from PIL import ImageFilter
    small = max(120, int(1000 * rng.uniform(0.10, 0.22)))   # baja resolucion agresiva
    im = img.resize((small, small), Image.BILINEAR).resize((1000, 1000), Image.BILINEAR)
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.6)))
    arr = np.asarray(im.convert("L")).astype(float)
    arr += rng.normal(0, rng.uniform(3, 14), arr.shape)       # ruido
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")


def generate_synthetic(out_dir, n, seed=0, do_degrade=True):
    import random
    from signs import canonical_slug
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(n):
        spell = random_spell(rng)
        img, ann = render_spell(spell, collect_annotations=True)
        # slug canonico por anotacion (para etiquetas de deteccion)
        for a in ann:
            if a["name"].startswith("sign_"):
                a["slug"] = canonical_slug(a["name"][5:].strip().lower().replace(" ", "_"))
        if do_degrade:
            # degrade usa numpy rng aparte para reproducibilidad
            class _R:
                random = staticmethod(lambda: rng.random())
                uniform = staticmethod(lambda a, b: rng.uniform(a, b))
                normal = staticmethod(lambda m, s, shape: nprng.normal(m, s, shape))
            img = degrade(img, _R)
        name = f"synth_{i:05d}.png"
        img.save(out_dir / "images" / name)
        index.append({"image": f"images/{name}", "spell": spell,
                      "annotations": [a for a in ann if a["kind"] == "sign"]})
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"Generados {n} sellos sinteticos en {out_dir}")
    print(f"  index.json con anotaciones (slug/x/y/size/rot) de cada signo.")
    total_signs = sum(len(e["annotations"]) for e in index)
    print(f"  {total_signs} signos anotados en total "
          f"({total_signs/max(n,1):.1f} por sello).")


def main():
    ap = argparse.ArgumentParser(description="Renderiza hechizos formato spell.json.")
    ap.add_argument("--spell", help="Ruta a un spell.json a renderizar.")
    ap.add_argument("--out", default="spell_render.png")
    ap.add_argument("--validate-examples", action="store_true")
    ap.add_argument("--examples-dir", default="/workspace/spell-maker-wha/examples")
    ap.add_argument("--synth", type=int, metavar="N",
                    help="Genera N sellos sinteticos con anotaciones.")
    ap.add_argument("--synth-out", default="data/_synth")
    ap.add_argument("--no-degrade", action="store_true",
                    help="No degradar (mantener render limpio).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.validate_examples:
        validate_examples(args.examples_dir)
        return
    if args.synth:
        generate_synthetic(args.synth_out, args.synth, seed=args.seed,
                           do_degrade=not args.no_degrade)
        return
    if args.spell:
        spell = json.loads(Path(args.spell).read_text())
        img = render_spell(spell)
        img.save(args.out)
        print(f"Render guardado en {args.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
