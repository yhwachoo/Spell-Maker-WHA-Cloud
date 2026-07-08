# Witch Hat Atelier — Análisis de hechizos y signos

Proyecto para **reconocer, descomponer y analizar** los sellos mágicos de Witch Hat
Atelier. Tres capacidades:

1. **Clasificar hechizos** — dada la imagen de un sello, predecir qué hechizo es
   (`train.py` / `predict.py`).
2. **Decodificar signos → vector** — detectar qué signos contiene un sello y calcular
   la dirección/potencia resultante (`decode_seal.py` + motor de vectores).
3. **Componer y analizar** — diseñar hechizos nuevos y predecir su comportamiento, o
   anotar los del catálogo (`spell_composer.py`, `spell_annotations.py`, `analyze_seal.py`).

## Estado actual
- **Clasificador**: 37 hechizos del catálogo, `models/best.pt`. Val acc ~74% (ver honestidad abajo).
- **Vocabulario de signos**: 24 signos dibujados en `data/signs/` (Column, Pull, Crush, Dirección, Convergencia...).
- **Etiquetas corregidas**: se detectó y corrigió un desfase que afectaba ~16 clases del catálogo (backup en `data/refs_backup/`).

> ⚠️ **Honestidad estadística.** Cada clase tiene ~1 dibujo real, expandido con variantes
> sintéticas. El val acc es **optimista** (train y val derivan del mismo dibujo). Para
> evaluación real hacen falta **2–3 recortes distintos** por clase.

> 🧭 **Lore: no se voltea.** Espejar/invertir un signo cambia su significado, así que los
> aumentos no voltean ni rotan fuerte por defecto (`--allow-flip` para forzarlo).

---

## Mapa de archivos

| Archivo | Qué hace |
|---|---|
| **Clasificador** | |
| `train.py` | Entrena el clasificador (GPU/CPU auto, AMP, label smoothing, matriz de confusión). |
| `predict.py` | Predice el hechizo de una imagen o carpeta. |
| `train_colab.ipynb` | Notebook autocontenido para Colab con GPU. |
| `prepare_data.py` | Crea carpetas y reporta conteos por clase. |
| `augment_offline.py` | Expande pocos recortes en variantes sintéticas (`refs/` → `raw/`). |
| `make_smoke_data.py` | Datos sintéticos para probar el pipeline. |
| **Datos / recorte** | |
| `decompose_sheet.py` | Recorta los sellos de una hoja-catálogo. |
| `crop_wiki_sign.py` | Recorta el glifo (rojo) de un screenshot de la wiki → `data/signs/`. |
| **Motor de vectores y análisis** | |
| `signs.py` | Catálogo de signos + **vectores direccionales** + `resultant_vector()`. |
| `vector_analysis.py` | Demos y CLI del motor de vectores. |
| `decode_seal.py` | Detecta signos en un sello (template matching rotacional) → vector. |
| `spell_composer.py` | Diseña un hechizo (signos+posición) → dibuja + predice + describe. |
| `spell_annotations.py` | Anotación manual asistida del catálogo + análisis fiable. |
| `analyze_seal.py` | Estima potencia/dirección/simetría de un sello (heurístico, sin entrenar). |
| `analyze_symbols.py` | Descompone sellos en símbolos y busca coincidencias entre hechizos. |
| **Integración con wha-spell-maker (fork)** | |
| `import_wha_symbols.py` | Importa los símbolos limpios del fork (DaviAMSilva, GPL v3) como plantillas + mapeo nombre→slug. |
| `render_spell.py` | Renderiza hechizos en formato `spell.json` (porta `sketch.ts`); valida contra los ejemplos y **genera datos sintéticos anotados**. |
| `examples_vectors.py` | Puente `spell.json` → motor de vectores; valida el motor contra los hechizos canon del fork. |
| `MECANICA.md` | Investigación: cómo tamaño/pesos/orientación determinan el hechizo. |
| `data/refs/` | 1 recorte semilla por hechizo (entrada de `augment_offline`). |
| `data/signs/` | Vocabulario de signos (plantillas para `decode_seal`). |
| `data/wha_symbols/` | Símbolos limpios del fork (44 signos, 31 sigilos, shapes, forbiddens, glifos) — GPL v3, ver `ATTRIBUTION.md`. |
| `models/best.pt` | Modelo entrenado. |

---

## Flujos de uso

### Entrenar el clasificador
```
python augment_offline.py --per-class 40 --clean       # refs/ -> raw/
python train.py --arch mobilenet_v3_small --epochs 15  # CPU
python train.py --arch resnet18 --epochs 40            # más capacidad (ideal en Colab/GPU)
python predict.py data/refs/nubes/nubes.png
```
En **Colab** (GPU gratis): sube `train_colab.ipynb`, activa GPU, ejecuta las celdas.

### Decodificar un sello en signos + vector
```
python decode_seal.py --validate          # fiabilidad del matcher (≈76% top-1 / 86% top-3)
python decode_seal.py --seal lanzallamas  # decodifica un sello del catálogo
```
> Nota: a la resolución del catálogo (~30 px/signo) el matcher es **poco fiable**
> (brecha de dominio). Funciona bien con sellos limpios (estilo plantilla).

### Componer un hechizo nuevo
```
python spell_composer.py --compose "column:90:1.5,column:90:1.5,direccion:90:1.2" --sigil fire_glyph --out hechizo.png
```
Formato `slug:ángulo:tamaño` (ángulo 0=derecha, 90=arriba). Dibuja el sello, predice el
vector y describe el efecto combinado.

### Anotar y analizar el catálogo (fiable)
```
python spell_annotations.py            # reconstruye + analiza los hechizos anotados a mano
```
Separa la **detección** (débil a baja res) del **análisis** (fiable vía motor de vectores).

### Analizar potencia/dirección sin entrenar
```
python analyze_seal.py data/refs/nubes/ --annotate
```

---

## Pipeline de datos (de imágenes a clases)

1. **Hoja-catálogo** → `python decompose_sheet.py hoja.png --montage` → recortes en `data/refs/_unsorted/`.
2. **Renombrar** cada recorte a su clase (`data/refs/<hechizo>/<hechizo>.png`). ⚠️ Verifica
   contra los captions: un desfase aquí corrompe todo el dataset (ya pasó una vez).
3. **Signos de la wiki** → guarda screenshots y `python crop_wiki_sign.py carpeta/` → `data/signs/`.
4. **Expandir** → `python augment_offline.py --per-class 40 --clean`.

---

## Integración con wha-spell-maker (fork de DaviAMSilva)

Se analizó el fork [Yhwachoo/Spell-Maker-WHA](https://github.com/Yhwachoo/Spell-Maker-WHA)
(original de DaviAMSilva, **GPL v3**). Es un editor web que hace lo **inverso** a
nosotros: dado un hechizo como datos (`spell.json`) lo **dibuja** con p5. Se rescató:

```
python import_wha_symbols.py            # importa symbols/ -> data/wha_symbols/ (+mapeo)
python decode_seal.py --validate        # matcher con multi-plantilla + validación cruzada
python render_spell.py --validate-examples --examples-dir <fork>/examples
python render_spell.py --synth 2000 --synth-out data/_synth   # datos sintéticos anotados
python examples_vectors.py --examples-dir <fork>/examples     # motor de vectores vs canon
```

Resultados:
- **Símbolos**: 44 signos del fork mapeados a nuestros slugs (33 coinciden, 11 nuevos:
  Coil, Conceal, Empower, Envelop, Focus, Project, Reflect, Solidify, Stability,
  Stillness, Stretch) + 31 sigilos limpios. `decode_seal` ahora admite **varias
  plantillas por slug** (dibujo a mano + limpia del fork) y canoniza sinónimos
  ES/EN (Enlarge=Agrandar…).
- **Matcher**: baseline intacto (76% top-1 / 86% top-3). La transferencia de dominio
  limpio↔garabato es **~37%**, y el techo limpio-vs-limpio es **~73%** → el template
  matching está cerca de su límite; el camino es un **detector entrenado**.
- **Renderizador**: reproduce los ejemplos canon casi pixel-perfect (Sylph Shoes IoU
  0.79, Crystal Petals 0.75; los bajos son por imágenes *custom* que no tenemos).
  Habilita generar sellos sintéticos con anotación perfecta (slug/posición/tamaño/
  rotación de cada signo) para entrenar detección.
- **Motor de vectores**: validado contra los hechizos canon del fork — todos los
  sellos simétricos dan deriva neta ≈0, distinguiendo dispersión-radial (Crystal
  Petals) de convergencia-inward (Sylph Shoes), coherente con el lore.

> GPL v3: los símbolos en `data/wha_symbols/` conservan `LICENSE` y atribución a
> DaviAMSilva (`ATTRIBUTION.md`). Los símbolos de la magia son de Kamome Shirahama.

## Clasificador de signos entrenado (supera al template matching)

El template matching se estanca y colapsa con imágenes degradadas (catálogo escaneado).
La alternativa es un **clasificador entrenado** sobre rasgos rotacionalmente invariantes
(`sign_features.py`) con datos aumentados y **degradados** (baja resolución + blur + ruido):

```
python train_sign_clf.py            # entrena y evalúa -> models/sign_clf.joblib
```

El entrenamiento usa **dos fuentes**: (1) plantillas aisladas aumentadas+degradadas y
(2) **recortes extraídos de sellos sintéticos completos** (`render_spell` como fábrica de
datos), pasados por el mismo pipeline de extracción del decodificador. Así el clasificador
aprende cómo se ven los signos *tras* la extracción (residuo del anillo, roce de vecinos,
formas parciales), no plantillas idealizadas.

**Robustez a degradación** (recorte de signo aislado):

| Método | top-1 | top-3 |
|---|---|---|
| Template matching (IoU) | 32% | 49% |
| Clasificador (solo plantillas) | 99% | 99.8% |

**End-to-end** (métrica real: decodificar un sello completo, extraer→clasificar):

| Entrenamiento | signos correctos |
|---|---|
| Solo plantillas aisladas | 48.6% |
| **Plantillas + sellos sintéticos** | **82.7%** |

> Matiz honesto: el end-to-end se mide sobre sellos sintéticos degradados (estilo limpio
> del editor). La generalización a un dibujo a mano **nuevo** (cross-domain) es 44–60%.
> Nunca se voltea en el aumento (espejar cambia el significado del signo). Cerrar el hueco
> con el trazo manga real es el siguiente paso.

El clasificador está integrado en `decode_seal.py` (`--method clf`, con fallback a
template) y es el método por defecto en la app web.

## App web de reconocimiento (`webapp/`)

Banco de pruebas visual para **ver cómo el sistema reconoce un sello**:

```
pip install -r requirements.txt
python webapp/app.py            # http://127.0.0.1:5000
```

- **Reconocer**: sube una imagen, **dibújala en el navegador** o elige un sello del
  catálogo → muestra las cajas de los símbolos detectados, el signo asignado a cada
  uno (con IoU y alternativas), el vector resultante y la descripción del efecto.
- **Componer**: elige sigilo + signos (radio/simetría/rotación) → dibuja el sello con
  `render_spell` y predice su comportamiento. Permite el ciclo *componer → reconocer*.

> El reconocimiento por template-matching es **débil a baja resolución** (~37%
> cross-domain); la app sirve justo para ver esa limitación e iterar hacia un
> detector entrenado con `render_spell.py --synth`.

## Mecánica del lore
Ver **`MECANICA.md`**: tamaño → potencia, pesos/simetría → desvío, orientación → dirección,
y el sistema de vectores implementado en `signs.py`. Idea central: *los signos son fuerzas
que se equilibran dentro del anillo*.

## Limitaciones honestas y siguientes pasos
- **1 dibujo por clase** → val acc optimista. Conseguir 2–3 recortes distintos por clase
  (p. ej. recortando los pergaminos de la hoja de compilación) es lo que más sube la precisión real.
- **Detección de signos a baja resolución**: poco fiable; el camino es plantillas en estilo
  manga o sellos de mayor resolución.
- Posibles extensiones: detección (localizar varios signos), exportar a ONNX, clasificador
  de signos independiente (`data/signs/`).
