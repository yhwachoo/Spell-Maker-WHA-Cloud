"""
IMPORTA los simbolos limpios del fork wha-spell-maker (DaviAMSilva, GPL v3) como
plantillas de alta calidad para decode_seal.py y como piezas de dibujo.

Los PNG del fork son line-art NEGRO sobre fondo TRANSPARENTE. Aqui se aplanan a
NEGRO-sobre-BLANCO (mismo formato que data/signs/), se recortan al bounding box y
se guardan en data/wha_symbols/<categoria>/<Nombre>.png.

Ademas construye el MAPEO nombre-del-fork -> slug de signs.py (por nombre canonico
en ingles + una tabla de alias para los slugs con nombre en espanol) e imprime que
signos coinciden y cuales son NUEVOS.

  python import_wha_symbols.py --src /workspace/spell-maker-wha/symbols
  python import_wha_symbols.py --src <ruta> --report   # solo imprime el mapeo

GPL v3: se conserva LICENSE y la atribucion a DaviAMSilva (ver data/wha_symbols/).
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from signs import SIGNS, SLUG_TO_NAME

ROOT = Path(__file__).parent
DEST = ROOT / "data" / "wha_symbols"
CATEGORIES = ["signs", "sigils", "shapes", "forbiddens", "tohs"]

# Alias: nombre-del-fork (normalizado) -> slug de signs.py, cuando el slug de
# signs.py tiene nombre en espanol y no coincide con el ingles del fork.
ALIASES = {
    "repetition": "repeticion",
    "convergence": "convergencia",
}


def normalize(name: str) -> str:
    """minusculas, sin parentesis (p.ej. 'Enlarge (Inverted)' -> 'enlarge')."""
    name = re.sub(r"\(.*?\)", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def build_name_index():
    """nombre_canonico_normalizado -> slug (a partir de signs.py) + alias."""
    idx = {}
    for slug, disp in SLUG_TO_NAME.items():
        idx[normalize(disp)] = slug
    idx.update(ALIASES)
    return idx


def import_symbols(src_root: Path, report_only=False):
    idx = build_name_index()
    known_slugs = {s["slug"] for s in SIGNS}

    summary = {"matched": [], "new": [], "by_cat": {}}

    for cat in CATEGORIES:
        cat_dir = src_root / cat
        if not cat_dir.is_dir():
            continue
        out_dir = DEST / cat
        if not report_only:
            out_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for png in sorted(cat_dir.glob("*.png")):
            fork_name = png.stem
            names.append(fork_name)
            if not report_only:
                # Se copian VERBATIM (line-art negro sobre transparente, con su
                # padding original): el renderizador necesita el padding para
                # reproducir la escala del fork, y decode_seal recorta/aplana al
                # cargar. Copiarlos sin modificar tambien respeta mejor la GPL.
                shutil.copy(png, out_dir / f"{fork_name}.png")
            if cat == "signs":
                slug = idx.get(normalize(fork_name))
                if slug and slug in known_slugs:
                    summary["matched"].append((fork_name, slug))
                else:
                    summary["new"].append(fork_name)
        summary["by_cat"][cat] = names

    return summary


def write_attribution():
    DEST.mkdir(parents=True, exist_ok=True)
    src_license = Path("/workspace/spell-maker-wha/LICENSE")
    if src_license.exists():
        shutil.copy(src_license, DEST / "LICENSE")
    (DEST / "ATTRIBUTION.md").write_text(
        "# Atribucion de simbolos\n\n"
        "Los PNG de esta carpeta derivan del proyecto **wha-spell-maker**\n"
        "de **DaviAMSilva** (https://github.com/DaviAMSilva/wha-spell-maker),\n"
        "via el fork https://github.com/Yhwachoo/Spell-Maker-WHA.\n\n"
        "Licencia original: **GPL v3** (ver `LICENSE`). Los simbolos se han\n"
        "aplanado (line-art negro sobre blanco) y recortado al bounding box\n"
        "para usarlos como plantillas de `decode_seal.py`; la obra original y\n"
        "su autoria se conservan bajo los terminos de la GPL v3.\n\n"
        "Los simbolos de la magia de Witch Hat Atelier son creacion de\n"
        "**Kamome Shirahama**.\n",
        encoding="utf-8",
    )


def print_report(summary):
    print("\n=== MAPEO DE SIGNOS (fork -> nuestro slug) ===\n")
    print(f"Signos del fork: {len(summary['by_cat'].get('signs', []))}")
    print(f"  Coinciden con signs.py: {len(summary['matched'])}")
    for fork, slug in sorted(summary["matched"]):
        print(f"    {fork:<22} -> {slug}")
    print(f"\n  NUEVOS (sin entrada en signs.py): {len(summary['new'])}")
    for fork in sorted(summary["new"]):
        print(f"    {fork}")
    print("\n=== OTRAS CATEGORIAS IMPORTADAS ===")
    for cat in ["sigils", "shapes", "forbiddens", "tohs"]:
        items = summary["by_cat"].get(cat, [])
        print(f"  {cat:<12}: {len(items)}  ({', '.join(items[:6])}{'...' if len(items) > 6 else ''})")


def main():
    ap = argparse.ArgumentParser(description="Importa simbolos del fork como plantillas.")
    ap.add_argument("--src", default="/workspace/spell-maker-wha/symbols")
    ap.add_argument("--report", action="store_true", help="solo imprime el mapeo, no copia.")
    args = ap.parse_args()

    src_root = Path(args.src)
    if not src_root.is_dir():
        raise SystemExit(f"No existe {src_root}")

    summary = import_symbols(src_root, report_only=args.report)
    if not args.report:
        write_attribution()
        print(f"Importado en {DEST}")
    print_report(summary)


if __name__ == "__main__":
    main()
