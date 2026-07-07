"""
Prepara la estructura de carpetas y reporta cuantas imagenes hay por signo.

Uso:
    python prepare_data.py            # crea data/raw/<signo>/ y muestra conteos
    python prepare_data.py --report   # solo reporta (no crea nada)

Deja tus imagenes asi:
    data/raw/column/img1.png
    data/raw/column/img2.jpg
    data/raw/pull/foto.png
    ...
El nombre de cada subcarpeta es la etiqueta (clase). No necesitas usar los 35
signos: con que tengas >= 2 imagenes en al menos 2 carpetas, ya puedes entrenar.
"""
import argparse
from pathlib import Path

from signs import SIGNS, display_name

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
RAW_DIR = Path(__file__).parent / "data" / "raw"


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser(description="Prepara/inspecciona el dataset de signos.")
    ap.add_argument("--report", action="store_true", help="Solo reportar, no crear carpetas.")
    args = ap.parse_args()

    if not args.report:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for s in SIGNS:
            (RAW_DIR / s["slug"]).mkdir(exist_ok=True)
        print(f"Carpetas creadas/verificadas en: {RAW_DIR}\n")

    # Reporta conteos: incluye carpetas del catalogo y cualquier carpeta extra que el usuario haya creado.
    known = {s["slug"] for s in SIGNS}
    existing = {p.name for p in RAW_DIR.iterdir() if p.is_dir()} if RAW_DIR.is_dir() else set()
    all_slugs = sorted(known | existing)

    total = 0
    usable = 0
    print(f"{'SIGNO':<22}{'CARPETA':<22}{'IMAGENES':>9}")
    print("-" * 53)
    for slug in all_slugs:
        n = count_images(RAW_DIR / slug)
        total += n
        if n >= 2:
            usable += 1
        flag = "" if n >= 2 else ("  <- vacia" if n == 0 else "  <- solo 1 (necesita >=2)")
        print(f"{display_name(slug):<22}{slug:<22}{n:>9}{flag}")
    print("-" * 53)
    print(f"Total imagenes: {total}  |  Clases utilizables (>=2 img): {usable}")
    if usable < 2:
        print("\nAun no puedes entrenar: necesitas >=2 imagenes en al menos 2 carpetas.")
    else:
        print(f"\nListo para entrenar con {usable} clases:  python train.py")


if __name__ == "__main__":
    main()
