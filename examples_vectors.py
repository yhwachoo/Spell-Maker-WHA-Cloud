"""
Puente examples/*.json (formato del fork)  ->  motor de vectores (signs.py).

Convierte cada hechizo canon del fork a nuestra lista de signos y calcula el
vector resultante, para VALIDAR que el motor da un comportamiento coherente con
lo que describe el lore (p. ej. sellos simetricos -> sin deriva neta; anillos
'inward' -> convergencia).

Cada entrada de signo del fork es un ANILLO de `amount` copias en simetria
radial; aqui se expande a copias individuales con su angulo de posicion en
nuestra convencion (0=derecha, 90=arriba, antihorario+).

  python examples_vectors.py --examples-dir /workspace/spell-maker-wha/examples
"""
import argparse
import json
from pathlib import Path

from signs import resultant_vector, canonical_slug, SLUG_TO_DIREC, display_name
from spell_composer import describe_spell
from import_wha_symbols import build_name_index, normalize

_IDX = build_name_index()


def fork_sign_to_slug(name):
    """'sign_Convergence' -> slug canonico de signs.py (o None si desconocido)."""
    base = name.split("_", 1)[1] if "_" in name else name
    slug = _IDX.get(normalize(base)) or normalize(base).replace(" ", "_")
    return canonical_slug(slug)


def expand_signs(seal):
    """Expande los anillos de signos a copias individuales [{slug, angle, size}]."""
    out = []
    for s in seal.get("signs", []):
        slug = fork_sign_to_slug(s["name"])
        if slug not in SLUG_TO_DIREC:
            # signo nuevo del fork sin metadatos de vector: se ignora en el motor
            continue
        amount = s.get("amount", 8) or 8
        rotation = s.get("rotation", 0) or 0
        self_angle = s.get("angle", 0) or 0
        size = (s.get("size", 200) or 200) / 200.0   # relativo al tamano por defecto
        for i in range(int(amount)):
            place = rotation + i * 360.0 / amount          # rotacion horaria desde arriba
            pos_deg = (90.0 - place - self_angle) % 360.0  # nuestra convencion
            out.append({"slug": slug, "angle": pos_deg, "size": size})
    return out


def analyze_example(jf: Path):
    spell = json.loads(jf.read_text())
    print(f"\n=== {spell.get('name', jf.stem)} ===")
    all_signs = []
    for seal in spell.get("seals", []):
        all_signs.extend(expand_signs(seal))
    if not all_signs:
        print("  (sin signos con metadatos de vector)")
        return
    kinds = {}
    for s in all_signs:
        kinds[display_name(s["slug"])] = kinds.get(display_name(s["slug"]), 0) + 1
    print("  Signos (expandidos):", ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())))
    r = resultant_vector(all_signs)
    print(f"  Vector resultante : {r['compass']}")
    print(f"    magnitud={r['magnitude']:.2f}  angulo={r['direction_deg']}deg  "
          f"radial={r['is_radial']}  inward={r['has_inward']}  outward={r['has_outward']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default="/workspace/spell-maker-wha/examples")
    args = ap.parse_args()
    for jf in sorted(Path(args.examples_dir).glob("*.json")):
        analyze_example(jf)


if __name__ == "__main__":
    main()
