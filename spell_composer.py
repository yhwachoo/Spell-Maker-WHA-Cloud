"""
COMPOSITOR DE HECHIZOS - Witch Hat Atelier

Disenas un sello eligiendo signos + posicion + tamano. El compositor:
  1) DIBUJA el sello (anillo + signos reales de data/signs/ colocados y rotados),
  2) PREDICE el vector resultante (direccion/potencia),
  3) DESCRIBE el efecto combinado en lenguaje natural (lore).

Tambien exporta describe_spell() para reutilizar en otros scripts (anotaciones, etc.).

Uso:
  python spell_composer.py --compose "column:90:1.5,column:90:1.5,direccion:90:1.2" --out mi_hechizo.png
  python spell_composer.py --compose "pull:0:1,pull:90:1,pull:180:1,pull:270:1"
  python spell_composer.py --compose "lluvia:0:1,float:0:1" --sigil fire_glyph

  formato: slug:angulo:tamano   (angulo 0=der 90=arr 180=izq 270=abj)
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from signs import resultant_vector, display_name, SLUG_TO_INFO

ROOT = Path(__file__).parent
SIGNS_DIR = ROOT / "data" / "signs"
REFS_DIR = ROOT / "data" / "refs"


def resolve_png(slug):
    """Busca el PNG de un signo/sigilo en data/signs/ y luego en data/refs/."""
    for base in (SIGNS_DIR, REFS_DIR):
        p = base / slug / f"{slug}.png"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Dibujo del sello
# ---------------------------------------------------------------------------

def _stamp(canvas, tmpl_path, cx, cy, target, rotate_deg=0):
    t = np.asarray(Image.open(tmpl_path).convert("L"))
    ys, xs = np.nonzero(t <= 128)
    if len(xs) == 0:
        return
    tb = t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray(tb)
    h, w = tb.shape
    s = target / max(h, w)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    if rotate_deg:
        im = im.rotate(rotate_deg, expand=True, fillcolor=255)
    a = np.asarray(im)
    hh, ww = a.shape
    y0, x0 = int(cy - hh / 2), int(cx - ww / 2)
    y1, x1 = y0 + hh, x0 + ww
    if y0 < 0 or x0 < 0 or y1 > canvas.shape[0] or x1 > canvas.shape[1]:
        return
    canvas[y0:y1, x0:x1][a <= 128] = 0


def draw_seal(sign_list, out_path, size=520, sigil=None):
    cx = cy = size // 2
    ring_r = int(size * 0.42)
    place_r = int(size * 0.27)
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r], outline=0, width=4)
    canvas = np.asarray(img).copy()

    # Sigilo central (opcional)
    if sigil:
        sp = resolve_png(sigil)
        if sp:
            _stamp(canvas, sp, cx, cy, int(size * 0.20))

    # Abanica signos que comparten (casi) el mismo angulo, para que no se encimen.
    groups = {}
    for s in sign_list:
        groups.setdefault(round(s["angle"] / 8) * 8, []).append(s)

    for base_ang, group in groups.items():
        k = len(group)
        for i, s in enumerate(group):
            slug, sz = s["slug"], s["size"]
            sp = resolve_png(slug)
            if not sp:
                continue
            draw_ang = s["angle"] + (i - (k - 1) / 2) * 18   # abanico visual
            a = math.radians(draw_ang)
            px = cx + place_r * math.cos(a)
            py = cy - place_r * math.sin(a)
            info = SLUG_TO_INFO.get(slug, {})
            rot = (90 - s["angle"]) if info.get("direction", {}).get("rotate_with_pos") else 0
            _stamp(canvas, sp, px, py, int(size * 0.13 * sz), rotate_deg=rot)

    Image.fromarray(canvas).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Descripcion del hechizo (reutilizable)
# ---------------------------------------------------------------------------

def describe_spell(sign_list, sigil=None):
    """Devuelve (texto_descripcion, dict_vector)."""
    r = resultant_vector(sign_list)
    lines = []

    # 1. Direccion / forma de manifestacion
    if r["is_radial"] and r["magnitude"] < 0.3:
        lines.append("- Manifestacion: DISPERSION RADIAL (la magia sale en todas direcciones).")
    elif r["compass"].startswith("convergencia"):
        lines.append("- Manifestacion: CONVERGE en el centro (la magia se concentra / sube por el sigilo).")
    elif r["compass"].startswith("magia emana"):
        lines.append("- Manifestacion: la magia EMANA DEL ESPACIO entre signos opuestos.")
    elif r["magnitude"] >= 0.3:
        fuerza = "fuerte" if r["magnitude"] >= 2 else ("media" if r["magnitude"] >= 1 else "leve")
        lines.append(f"- Direccion: se proyecta hacia {r['compass'].upper()} "
                     f"(angulo {r['direction_deg']}deg, potencia {fuerza} {r['magnitude']:.1f}).")
    else:
        lines.append("- Manifestacion: centrada / sin direccion dominante.")

    if r["has_inward"] and r["has_outward"]:
        lines.append("- AVISO: hay signos hacia adentro Y hacia afuera (efecto en zona intermedia).")

    # 2. Elemento
    if sigil:
        lines.append(f"- Elemento (sigilo central): {display_name(sigil)}.")

    # 3. Efectos de cada signo
    seen = set()
    lines.append("- Efectos de los signos:")
    for s in sign_list:
        slug = s["slug"]
        if slug in seen:
            continue
        seen.add(slug)
        info = SLUG_TO_INFO.get(slug)
        eff = info["effect"] if info else "(desconocido)"
        n = sum(1 for x in sign_list if x["slug"] == slug)
        mult = f" x{n}" if n > 1 else ""
        lines.append(f"    - {display_name(slug)}{mult}: {eff}")

    # 4. Interacciones notables
    slugs = {s["slug"] for s in sign_list}
    notas = []
    if {"float"} & slugs:
        notas.append("Float mantiene el efecto/objetos flotando.")
    if {"repeticion", "repetition"} & slugs:
        notas.append("Repeticion resetea continuamente (efecto sostenido en el tiempo).")
    if {"tornillo", "bolt"} & slugs and r["magnitude"] >= 0.3:
        notas.append("Con un signo direccional, los rayos se disparan a alta velocidad.")
    if {"radial"} & slugs:
        notas.append("Radial REDUCE la potencia del hechizo.")
    if {"mira", "crosshair"} & slugs:
        notas.append("Solo afecta objetos con el mismo aspecto magico.")
    if notas:
        lines.append("- Interacciones:")
        for nt in notas:
            lines.append(f"    - {nt}")

    return "\n".join(lines), r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_compose(raw):
    out = []
    for tok in raw.split(","):
        p = tok.strip().split(":")
        out.append({"slug": p[0],
                    "angle": float(p[1]) if len(p) > 1 else 0.0,
                    "size": float(p[2]) if len(p) > 2 else 1.0})
    return out


def main():
    ap = argparse.ArgumentParser(description="Compositor de hechizos.")
    ap.add_argument("--compose", required=True, help="'slug:ang:size,...'")
    ap.add_argument("--sigil", default=None, help="Sigilo central (ej. fire_glyph).")
    ap.add_argument("--out", default="hechizo.png", help="PNG de salida del sello.")
    args = ap.parse_args()

    signs = parse_compose(args.compose)
    print("\n=== HECHIZO COMPUESTO ===")
    print("Signos:")
    for s in signs:
        print(f"  - {display_name(s['slug']):<20} pos={s['angle']:>6.1f}deg  size={s['size']:.2f}")

    text, r = describe_spell(signs, sigil=args.sigil)
    print("\n=== PREDICCION DE COMPORTAMIENTO ===")
    print(text)

    out = draw_seal(signs, args.out, sigil=args.sigil)
    print(f"\nSello dibujado: {out}")


if __name__ == "__main__":
    main()
