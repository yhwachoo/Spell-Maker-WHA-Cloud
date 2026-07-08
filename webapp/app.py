"""
App web para TESTEAR el reconocimiento de sellos de Witch Hat Atelier.

Dos modos:
  - RECONOCER: subes/dibujas/eliges un sello -> el sistema extrae sus simbolos,
    los matchea contra el vocabulario de signos, dibuja las cajas detectadas y
    calcula el vector resultante + descripcion del efecto.
  - COMPONER:  eliges signos (posicion/tamano) -> se dibuja el sello (render_spell)
    y se predice su comportamiento. Sirve para el ciclo componer -> reconocer.

Uso:
  pip install -r requirements.txt      # incluye flask
  python webapp/app.py                  # abre http://127.0.0.1:5000

Nota honesta: el reconocimiento por template matching es DEBIL a baja resolucion
(brecha de dominio ~37% cross-domain). Esta app existe para VER esa limitacion y
para iterar; el camino a futuro es un detector entrenado con datos sinteticos
(render_spell.py --synth).
"""
import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from analyze_symbols import extract_symbols  # noqa: E402
from decode_seal import load_vocabulary, match, match_clf, load_classifier, MIN_IOU  # noqa: E402
from signs import resultant_vector, display_name, SLUG_TO_INFO  # noqa: E402
from spell_composer import describe_spell  # noqa: E402
from render_spell import render_spell  # noqa: E402

app = Flask(__name__)

REFS = ROOT / "data" / "refs"
WHA_SIGNS = ROOT / "data" / "wha_symbols" / "signs"
WHA_SIGILS = ROOT / "data" / "wha_symbols" / "sigils"

# Vocabulario cargado una vez (nuestras plantillas + las limpias del fork).
VOCAB = load_vocabulary(include_wha=True)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_to_img(data_url: str) -> Image.Image:
    _, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def decode_image(pil_img: Image.Image, max_symbols=12, method="clf"):
    """Extrae simbolos, los matchea y calcula el vector. Devuelve dict serializable.

    method: 'clf' (clasificador entrenado, robusto a degradacion) o 'template'
    (IoU rotacional). Si 'clf' no esta disponible, cae a 'template'.
    """
    clf_model = load_classifier() if method == "clf" else None
    used_method = "clf" if clf_model else "template"
    # Aplana sobre blanco y guarda a un buffer que extract_symbols pueda abrir.
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.alpha_composite(pil_img)
    flat = bg.convert("L")
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    buf.seek(0)

    comps, ring = extract_symbols(buf, min_area=12, max_comp=max_symbols)
    overlay = flat.convert("RGB")
    d = ImageDraw.Draw(overlay)
    cx, cy, ring_R = ring
    d.ellipse([cx - ring_R, cy - ring_R, cx + ring_R, cy + ring_R], outline=(120, 170, 255), width=2)

    detected = []
    med = np.median([c["area"] for c in comps]) if comps else 1.0
    min_conf = 0.12 if used_method == "clf" else MIN_IOU
    for c in comps:
        if used_method == "clf":
            res = match_clf(c["bin"], clf_model, topk=3)
        else:
            res = match(c["bin"], VOCAB, topk=3)
        if not res or res[0][1] < min_conf:
            continue
        slug, sc, conf = res[0]
        x0, y0, x1, y1 = c["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=(230, 60, 60), width=3)
        d.text((x0 + 2, y0 + 2), display_name(slug), fill=(230, 60, 60))
        detected.append({
            "slug": slug, "name": display_name(slug),
            "angle": round(c["angle"], 1),
            "size": round((c["area"] / med) ** 0.5, 2),
            "iou": round(sc, 3),
            "alts": [display_name(r[0]) for r in res[1:]],
            "bbox": [x0, y0, x1, y1],
        })

    vector = resultant_vector(detected)
    text, _ = describe_spell(detected) if detected else ("(no se detectaron signos)", vector)
    return {
        "overlay": img_to_b64(overlay),
        "symbols": detected,
        "n_extracted": len(comps),
        "vector": vector,
        "description": text,
        "method": used_method,
    }


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/catalog")
def api_catalog():
    items = []
    if REFS.exists():
        for d in sorted(p for p in REFS.iterdir() if p.is_dir()):
            png = d / f"{d.name}.png"
            if png.exists():
                items.append({"name": d.name, "label": display_name(d.name)})
    return jsonify(items)


@app.route("/api/signs")
def api_signs():
    """Signos disponibles como plantilla para el compositor."""
    out = []
    for p in sorted(WHA_SIGNS.glob("*.png")):
        out.append({"value": f"sign_{p.stem}", "label": p.stem})
    return jsonify(out)


@app.route("/api/sigils")
def api_sigils():
    out = []
    for p in sorted(WHA_SIGILS.glob("*.png")):
        out.append({"value": f"sigil_{p.stem}", "label": p.stem})
    return jsonify(out)


@app.route("/api/decode", methods=["POST"])
def api_decode():
    try:
        if "file" in request.files and request.files["file"].filename:
            img = Image.open(request.files["file"].stream).convert("RGBA")
        else:
            data = request.get_json(silent=True) or {}
            if data.get("catalog"):
                png = REFS / data["catalog"] / f"{data['catalog']}.png"
                if not png.exists():
                    return jsonify({"error": "sello no encontrado"}), 404
                img = Image.open(png).convert("RGBA")
            elif data.get("dataurl"):
                img = b64_to_img(data["dataurl"])
            else:
                return jsonify({"error": "sin imagen"}), 400
        method = request.form.get("method") or (request.get_json(silent=True) or {}).get("method") or "clf"
        return jsonify(decode_image(img, method=method))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/compose", methods=["POST"])
def api_compose():
    data = request.get_json(silent=True) or {}
    signs = data.get("signs", [])
    sigil = data.get("sigil")
    seal = {"rings": [{}], "sigils": [], "signs": []}
    if sigil:
        seal["sigils"].append({"name": sigil, "size": 360})
    sign_list = []
    for s in signs:
        name = s.get("name")
        amount = int(s.get("amount", 8) or 8)
        radius = int(s.get("radius", 330) or 330)
        size = int(s.get("size", 150) or 150)
        rotation = float(s.get("rotation", 0) or 0)
        seal["signs"].append({"name": name, "amount": amount, "radius": radius,
                              "size": size, "rotation": rotation})
        # Para la prediccion de vector: expandir el anillo a copias.
        base = name.split("_", 1)[1].lower().replace(" ", "_") if "_" in name else name
        from signs import canonical_slug, SLUG_TO_DIREC
        slug = canonical_slug(base)
        if slug in SLUG_TO_DIREC:
            for i in range(amount):
                place = rotation + i * 360.0 / amount
                pos = (90.0 - place) % 360.0
                sign_list.append({"slug": slug, "angle": pos, "size": size / 200.0})
    spell = {"seals": [seal]}
    try:
        img = render_spell(spell)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    text, vec = describe_spell(sign_list, sigil=None) if sign_list else ("(sin signos)", resultant_vector([]))
    return jsonify({"image": img_to_b64(img), "vector": vec, "description": text, "spell": spell})


if __name__ == "__main__":
    print("Vocabulario:", len(VOCAB), "slugs")
    app.run(host="127.0.0.1", port=5000, debug=False)
