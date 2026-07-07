"""
ANOTACION ASISTIDA de hechizos del catalogo.

Como la deteccion automatica falla a baja resolucion, aqui se anota A MANO que signos
contiene cada sello (mirando el dibujo real). El motor de vectores (signs.py) produce
entonces el analisis FIABLE: separa la DETECCION (debil) del ANALISIS (fuerte).

Para cada hechizo anotado, el script:
  1) reconstruye el sello con el compositor (para verificar visualmente la anotacion),
  2) genera un montaje  REAL | RECONSTRUIDO,
  3) imprime la prediccion de comportamiento (vector + efecto).

Las anotaciones llevan "confianza" y "razon" — son interpretaciones a VERIFICAR, no
verdad oficial. Edita/anade libremente siguiendo el mismo formato.

Uso:
  python spell_annotations.py                 # analiza + reconstruye todas
  python spell_annotations.py --name nubes    # solo una
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from spell_composer import describe_spell, draw_seal, resolve_png
from signs import display_name

ROOT = Path(__file__).parent
REFS_DIR = ROOT / "data" / "refs"
OUT_DIR = ROOT / "data" / "annotations"


# ---------------------------------------------------------------------------
# Anotaciones (formato: slug, angle [0=der,90=arr], size). Editar/ampliar.
# ---------------------------------------------------------------------------

ANNOTATIONS = {
    "escudo_de_hielo": {
        "sigil": "ice_glyph",
        "confianza": "media",
        "razon": "Flecha central grande hacia ARRIBA = direccion del escudo (se crea "
                 "delante/arriba del creador). Signos menores radiales = proteccion.",
        "signs": [
            {"slug": "direccion", "angle": 90,  "size": 1.8},   # flecha dominante arriba
            {"slug": "direccion", "angle": 210, "size": 0.8},
            {"slug": "direccion", "angle": 330, "size": 0.8},
        ],
    },
    "nubes": {
        "sigil": None,
        "confianza": "media",
        "razon": "Flor central = Ondulante (transforma material en nube esponjosa). "
                 "Float la mantiene flotando. Sin direccion dominante.",
        "signs": [
            {"slug": "ondulante", "angle": 90,  "size": 1.3},
            {"slug": "float",     "angle": 270, "size": 1.0},
            {"slug": "direccion", "angle": 0,   "size": 0.7},
            {"slug": "direccion", "angle": 180, "size": 0.7},
        ],
    },
    "circulo_volador": {
        "sigil": None,
        "confianza": "media",
        "razon": "Glifo soarboot central (Float) + varios signos apuntando ADENTRO "
                 "que mantienen el objeto suspendido en el aire (convergencia de soporte).",
        "signs": [
            {"slug": "float", "angle": 90, "size": 1.4},
            {"slug": "levitation", "angle": 30,  "size": 1.0},
            {"slug": "levitation", "angle": 150, "size": 1.0},
            {"slug": "levitation", "angle": 270, "size": 1.0},
        ],
    },
    "disparo_de_agua": {
        "sigil": None,
        "confianza": "baja",
        "razon": "Sigilo de agua central + signos de Convergencia/Direccion que concentran "
                 "y disparan el agua en una direccion (chorro).",
        "signs": [
            {"slug": "convergencia", "angle": 90, "size": 1.2},
            {"slug": "direccion",    "angle": 90, "size": 1.5},
        ],
    },
}


# ---------------------------------------------------------------------------
def compare_montage(name, recon_path, out_path, cell=300):
    real_p = REFS_DIR / name / f"{name}.png"
    sheet = Image.new("RGB", (cell * 2, cell + 24), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (label, p) in enumerate([("REAL", real_p), ("RECONSTRUIDO", recon_path)]):
        if p and Path(p).exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 16, cell - 16))
            sheet.paste(im, (i * cell + 8, 8))
        d.rectangle([i * cell, cell, i * cell + cell, cell + 24], fill=(255, 235, 0))
        d.text((i * cell + 6, cell + 5), label, fill=(0, 0, 0))
    sheet.save(out_path)


def analyze(name, ann):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n{display_name(name).upper()}   (confianza: {ann['confianza']})")
    print(f"{'='*60}")
    print(f"Razon de la anotacion: {ann['razon']}\n")
    print("Signos anotados:")
    for s in ann["signs"]:
        print(f"  - {display_name(s['slug']):<16} pos={s['angle']:>5.0f}deg size={s['size']:.1f}")
    text, r = describe_spell(ann["signs"], sigil=ann.get("sigil"))
    print("\nPREDICCION DE COMPORTAMIENTO:")
    print(text)
    recon = OUT_DIR / f"{name}_recon.png"
    draw_seal(ann["signs"], recon, sigil=ann.get("sigil"))
    cmp_path = OUT_DIR / f"{name}_compare.png"
    compare_montage(name, recon, cmp_path)
    print(f"\nComparacion REAL|RECONSTRUIDO: {cmp_path}")


def main():
    ap = argparse.ArgumentParser(description="Analiza hechizos anotados a mano.")
    ap.add_argument("--name", default=None, help="Solo este hechizo.")
    args = ap.parse_args()
    items = {args.name: ANNOTATIONS[args.name]} if args.name else ANNOTATIONS
    for name, ann in items.items():
        analyze(name, ann)


if __name__ == "__main__":
    main()
