"""
Analiza y visualiza el VECTOR RESULTANTE de un sello dado su composicion de signos.

Uso:
  python vector_analysis.py              # corre todos los demos del lore
  python vector_analysis.py --demo todos_arriba
  python vector_analysis.py --compose "column:90:1.5,direccion:90:1.2,pull:270:1.0"
    formato: slug:angulo_posicion:tamano_relativo
    angulo: 0=derecha, 90=arriba, 180=izquierda, 270=abajo
"""
import argparse
import math
from typing import List, Dict

from signs import resultant_vector, display_name, SLUG_TO_DIREC


# ---------------------------------------------------------------------------
# Visualizacion ASCII
# ---------------------------------------------------------------------------

def ascii_compass(dx, dy, mag, size=9):
    """Cuadricula ASCII con el vector resultante."""
    grid = [['.' for _ in range(size)] for _ in range(size)]
    cx = cy = size // 2
    grid[cy][cx] = 'O'
    if mag > 0.01:
        nx, ny = dx / mag, dy / mag
        steps = size // 2 - 1
        for t in range(1, steps + 1):
            px = cx + int(round(nx * t))
            py = cy - int(round(ny * t))   # -ny: y de imagen -> pantalla
            if 0 <= px < size and 0 <= py < size:
                grid[py][px] = '*' if t < steps else '>'
    return '\n'.join('  ' + ' '.join(row) for row in grid)


def describe_sign(slug, angle, size):
    info = SLUG_TO_DIREC.get(slug, {})
    bv = info.get("base_vec", None)
    t  = info.get("type", "?")
    bv_map = {"inward": "->ADENTRO", "outward": "<-AFUERA",
              "radial_out": "<->RADIAL", None: "sin-vec"}
    dir_txt = bv_map.get(bv, f"local{bv}")
    return (f"  {display_name(slug):<22}"
            f"  ang={angle:>6.1f}  size={size:.2f}"
            f"  {dir_txt:<12}  [{t}]")


def print_result(title: str, sign_list: List[Dict]):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print("  Signos:")
    for s in sign_list:
        print(describe_sign(s['slug'], s['angle'], s['size']))

    r = resultant_vector(sign_list)
    print(f"\n  Resultado:")
    print(f"    Direccion  : {r['compass']}")
    print(f"    Angulo     : {r['direction_deg']} deg"
          f"  (0=der, 90=arr, 180=izq, 270=abj)")
    print(f"    Magnitud   : {r['magnitude']:.3f}")
    print(f"    Dispersion : {'SI' if r['is_radial'] else 'no'}"
          f"   Inward: {'SI' if r['has_inward'] else 'no'}"
          f"   Outward: {'SI' if r['has_outward'] else 'no'}"
          f"   Signos activos: {r['active_signs']}")
    print(f"\n{ascii_compass(r['dx'], r['dy'], r['magnitude'])}")
    return r


# ---------------------------------------------------------------------------
# Demos del lore
# ---------------------------------------------------------------------------

DEMOS = {

    # Regla 1: todos los signos de direccion apuntan al mismo lado
    "todos_arriba": {
        "title": "REGLA 1 - Todos los signos de Direccion apuntan hacia ARRIBA",
        "desc":  "3 x Direccion en la misma posicion (90=arriba). "
                 "Todos suman en la misma direccion -> hechizo va ARRIBA con fuerza 3x.",
        "signs": [
            {"slug": "direccion", "angle": 90,  "size": 1.0},
            {"slug": "direccion", "angle": 90,  "size": 1.0},
            {"slug": "direccion", "angle": 90,  "size": 1.0},
        ],
    },

    "todos_derecha": {
        "title": "REGLA 1 - Todos los signos de Direccion apuntan hacia la DERECHA",
        "desc":  "3 x Direccion a 0 grados. Hechizo va a la DERECHA.",
        "signs": [
            {"slug": "direccion", "angle": 0,  "size": 1.0},
            {"slug": "direccion", "angle": 0,  "size": 1.0},
            {"slug": "direccion", "angle": 0,  "size": 1.0},
        ],
    },

    # Regla 2: todos apuntan hacia afuera -> dispersion
    "todos_afuera": {
        "title": "REGLA 2 - 4 x Column en los 4 puntos cardinales (afuera)",
        "desc":  "Cada Column apunta hacia afuera desde su posicion. "
                 "Los vectores se cancelan -> dispersion en todas direcciones.",
        "signs": [
            {"slug": "column", "angle":   0, "size": 1.0},   # derecha
            {"slug": "column", "angle":  90, "size": 1.0},   # arriba
            {"slug": "column", "angle": 180, "size": 1.0},   # izquierda
            {"slug": "column", "angle": 270, "size": 1.0},   # abajo
        ],
    },

    # Regla 3: todos apuntan hacia adentro -> convergencia
    "todos_adentro": {
        "title": "REGLA 3 - 4 x Pull en los 4 puntos cardinales (adentro)",
        "desc":  "Pull en cada cardinal apunta hacia el centro. "
                 "Vectores se cancelan -> magia converge en el centro del sello.",
        "signs": [
            {"slug": "pull", "angle":   0, "size": 1.0},
            {"slug": "pull", "angle":  90, "size": 1.0},
            {"slug": "pull", "angle": 180, "size": 1.0},
            {"slug": "pull", "angle": 270, "size": 1.0},
        ],
    },

    # Regla 4: dos signos apuntandose -> magia entre ellos
    "apuntando_entre_si": {
        "title": "REGLA 4 - Dos Column opuestos apuntandose el uno al otro",
        "desc":  "Column a 90 (arriba) apunta UP. Column a 270 (abajo) apunta DOWN. "
                 "Se cancelan -> magia emana del ESPACIO entre ellos (zona media).",
        "signs": [
            {"slug": "column", "angle":  90, "size": 1.0},
            {"slug": "column", "angle": 270, "size": 1.0},
        ],
    },

    # Regla 5: un signo mas grande desvia el hechizo
    "desvio_por_peso": {
        "title": "REGLA 5 - Un signo mas grande desvia el hechizo",
        "desc":  "4 x Column en cardinales, pero el de la derecha (ang=0) es 2.5x mas grande. "
                 "Su vector domina -> hechizo se desvia hacia la DERECHA.",
        "signs": [
            {"slug": "column", "angle":   0, "size": 2.5},  # grande a la derecha
            {"slug": "column", "angle":  90, "size": 1.0},
            {"slug": "column", "angle": 180, "size": 1.0},
            {"slug": "column", "angle": 270, "size": 1.0},
        ],
    },

    # Regla 6: mezcla inward + outward -> zona entre signos
    "inward_outward": {
        "title": "REGLA 6 - Pull (adentro) + Column (afuera) en lados opuestos",
        "desc":  "Pull arriba atrae hacia centro, Column abajo empuja hacia afuera. "
                 "Ambos se anulan en net pero en sentidos opuestos -> zona entre ellos.",
        "signs": [
            {"slug": "pull",   "angle":  90, "size": 1.0},  # adentro desde arriba
            {"slug": "column", "angle": 270, "size": 1.0},  # afuera hacia abajo
        ],
    },

    # Composicion realista: lanzallamas
    "lanzallamas": {
        "title": "COMPOSICION - Lanzallamas (Column x2 + Direccion, todos arriba)",
        "desc":  "Dos Column y un Direccion apuntando todos a 90 (arriba). "
                 "Haz potente hacia arriba/adelante.",
        "signs": [
            {"slug": "column",    "angle": 90, "size": 1.5},
            {"slug": "column",    "angle": 90, "size": 1.5},
            {"slug": "direccion", "angle": 90, "size": 1.2},
        ],
    },

    # Composicion: disparo de agua (columna + direccion + pull lateral)
    "disparo_agua": {
        "title": "COMPOSICION - Disparo de agua (Column adelante, Pull lateral)",
        "desc":  "Column apunta arriba (haz principal). "
                 "Pull a 0 y 180 (lados) se cancelan entre si. "
                 "Resultado: haz dirigido hacia arriba.",
        "signs": [
            {"slug": "column", "angle":  90, "size": 1.2},
            {"slug": "pull",   "angle":   0, "size": 0.8},
            {"slug": "pull",   "angle": 180, "size": 0.8},
        ],
    },

    # Composicion: desvio intencional (barrera de agua)
    "barrera_agua": {
        "title": "COMPOSICION - Barrera de agua (dispersion radial)",
        "desc":  "Lluvia + Dispersion crean efecto radial omnidireccional. "
                 "Float mantiene el efecto en el aire. Sin vector neto = barrera esferica.",
        "signs": [
            {"slug": "lluvia",    "angle":  0, "size": 1.0},
            {"slug": "dispersion","angle":  0, "size": 1.0},
            {"slug": "float",     "angle":  0, "size": 1.0},
        ],
    },

    # Angulo oblicuo
    "diagonal": {
        "title": "COMPOSICION - Tres Direccion en diagonal (45 grados)",
        "desc":  "Direccion x3 apuntando a 45 grados (arriba-derecha). "
                 "Resultado: hechizo va en diagonal arriba-derecha.",
        "signs": [
            {"slug": "direccion", "angle": 45, "size": 1.0},
            {"slug": "direccion", "angle": 45, "size": 1.0},
            {"slug": "direccion", "angle": 45, "size": 1.0},
        ],
    },
}


# ---------------------------------------------------------------------------
# Tabla guia rapida
# ---------------------------------------------------------------------------

def print_guide():
    print(f"\n{'='*64}")
    print("  GUIA RAPIDA - Vector por tipo de signo")
    print(f"{'='*64}")
    order = ["directional", "semi_directional", "non_directional", "asymmetric"]
    prev = None
    for t in order:
        entries = [(slug, info) for slug, info in sorted(SLUG_TO_DIREC.items())
                   if info["type"] == t]
        if not entries:
            continue
        print(f"\n  [{t.upper()}]")
        for slug, info in entries:
            bv = info.get("base_vec")
            bv_map = {"inward": "->ADENTRO", "outward": "<-AFUERA",
                      "radial_out": "<->RADIAL", None: "sin-vector"}
            bv_str = bv_map.get(bv, f"local{bv}")
            inv = info.get("invert_vec")
            if inv and inv != bv:
                inv_map = {"inward": "->ADENTRO", "outward": "<-AFUERA",
                           "radial_out": "<->RADIAL", None: "sin-vector"}
                inv_str = f"  | invertido: {inv_map.get(inv, str(inv))}"
            else:
                inv_str = ""
            rp = "  [rota con posicion]" if info.get("rotate_with_pos") else ""
            print(f"    {display_name(slug):<24} {bv_str:<14}{inv_str}{rp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_compose(raw: str) -> List[Dict]:
    signs = []
    for tok in raw.split(","):
        parts = tok.strip().split(":")
        slug  = parts[0]
        angle = float(parts[1]) if len(parts) > 1 else 0.0
        size  = float(parts[2]) if len(parts) > 2 else 1.0
        signs.append({"slug": slug, "angle": angle, "size": size})
    return signs


def main():
    ap = argparse.ArgumentParser(description="Analiza el vector resultante de un sello.")
    ap.add_argument("--demo",    default=None)
    ap.add_argument("--compose", default=None,
                    help="Composicion manual: 'slug:ang:size,...'")
    ap.add_argument("--guide",   action="store_true",
                    help="Mostrar tabla de vectores por signo.")
    args = ap.parse_args()

    if args.compose:
        sign_list = parse_compose(args.compose)
        print_result("Composicion personalizada", sign_list)
        return

    if args.guide:
        print_guide()
        return

    if args.demo and args.demo != "all":
        d = DEMOS.get(args.demo)
        if d:
            print_result(d["title"], d["signs"])
            print(f"  Descripcion: {d['desc']}")
        else:
            print(f"Demo '{args.demo}' no encontrado.")
            print(f"Disponibles: {list(DEMOS.keys())}")
        return

    # Todos los demos
    print("\n  SISTEMA DE VECTORES - Witch Hat Atelier")
    print("  Angulos: 0=derecha  90=arriba  180=izquierda  270=abajo\n")
    for key, d in DEMOS.items():
        r = print_result(d["title"], d["signs"])
        print(f"  >> {d['desc']}")

    print_guide()


if __name__ == "__main__":
    main()
