# Contexto del proyecto — Witch Hat Atelier: análisis de hechizos y signos

Este archivo es el handoff para la sesión en la nube. Léelo completo antes de actuar.
Documentación técnica detallada: `README.md` (herramientas y flujos) y `MECANICA.md`
(reglas del lore + sistema de vectores).

## Qué busca el usuario (objetivo general)

Construir un sistema completo alrededor de la magia de Witch Hat Atelier:

1. **Reconocer hechizos**: dado el dibujo de un sello, identificar qué hechizo es
   (clasificador CNN, `train.py`/`predict.py`).
2. **Decodificar sellos**: detectar qué **signos** (modificadores) contiene un sello,
   su posición/tamaño/orientación, y calcular el **vector resultante** del hechizo
   (dirección + potencia) según las reglas del lore (`decode_seal.py`, `signs.py`).
3. **Componer hechizos nuevos**: diseñar sellos eligiendo signos y predecir su
   comportamiento (`spell_composer.py`).
4. Fidelidad al lore: la orientación de los signos importa (arriba/abajo/adentro/afuera),
   espejar un signo cambia su significado (por eso NUNCA se usa flip como augmentation),
   el tamaño da potencia, el desbalance desvía el hechizo.

## Estado actual (dónde quedamos)

- **Clasificador**: 37 hechizos del catálogo en español, val acc ~74% (honesto tras
  corregir un desfase de etiquetas que afectaba 17 clases). Limitación clave: **1 solo
  dibujo real por clase** → el val acc es optimista; falta un 2º dibujo por clase.
- **Vocabulario de signos**: 24 signos dibujados como plantillas en `data/signs/`
  (column, pull, crush, direccion, convergencia, lluvia, ojo, vision...).
- **Motor de vectores** (`signs.py` + `vector_analysis.py`): cada signo tiene vector
  direccional (inward/outward/local/radial), reglas del lore implementadas y verificadas.
- **decode_seal.py**: matcher por template matching rotacional. Validación: 76% top-1 /
  86% top-3 sobre plantillas. **Falla con los sellos del catálogo** (~30 px por signo,
  brecha de dominio entre plantillas limpias y garabatos escaneados).
- **spell_annotations.py**: anotación manual asistida (separa detección débil de
  análisis fiable).

## Tarea prioritaria para esta sesión en la nube

El usuario (GitHub: **Yhwachoo**) hizo fork de wha-spell-maker. Analizar su fork:
https://github.com/Yhwachoo/Spell-Maker-WHA (original: DaviAMSilva/wha-spell-maker).
Es un "spell maker" web (TypeScript/p5) para WHA, licencia **GPL v3** (mantener
atribución si se copia algo).

**Analizar ese repo a fondo y proponer/implementar qué rescatar.** En una exploración
preliminar ya se vio que contiene:

- `symbols/` — **PNGs limpios y de alta calidad** de sigilos (Fire, Earth, Water, Dragon,
  Bird, Fish, Flower...), shapes (Circle, Triangle, Star...) y forbiddens
  (Time Stop, Scalewolf Curse...). **Esto es lo más valioso**: sirven como
  (a) plantillas de mejor calidad para `decode_seal.py` (cerrar la brecha de dominio),
  (b) segundas muestras / clases nuevas para el clasificador,
  (c) piezas de dibujo para `spell_composer.py`.
- `src/symbols.ts` — catálogo/estructura de símbolos con su lógica (comparar con nuestro
  `signs.py`: nombres, categorías, quizá orientaciones).
- `src/sketch.ts` — cómo dibujan/rotan/colocan los símbolos en el sello (útil para
  mejorar `draw_seal()` de `spell_composer.py`).
- `examples/*.json` — hechizos completos ya compuestos (Sylph Shoes, Crystal Petals
  Path...) con su estructura de datos: **sirven como hechizos anotados de referencia**
  para validar nuestro motor de vectores y como test set.
- `src/schemas/spell.json` — schema del formato de hechizo (candidato a formato de
  intercambio con nuestro `spell_annotations.py`).

Pasos sugeridos (validar con el usuario antes de cambios grandes):
1. Mapear `symbols/` contra nuestras clases (`data/signs/`, `data/refs/`): qué coincide,
   qué es nuevo.
2. Importar los PNGs útiles como plantillas de `decode_seal` y re-validar el matcher.
3. Estudiar `examples/*.json` + `spell.json` para adoptar un formato de hechizo común.
4. Ver si `src/symbols.ts` trae metadatos de orientación/categoría que completen `signs.py`.

## Reglas del proyecto (no romper)

- **Nunca usar flip/espejo como augmentation** (cambia el significado del signo).
- Rotaciones leves solamente (±12°) en augmentation.
- Etiquetas del catálogo: verificar SIEMPRE contra los captions (`source_sheets/review/`);
  ya hubo un desfase que corrompió 17 clases y costó re-entrenar.
- Ser honesto con las métricas: con 1 dibujo/clase el val acc es optimista; decirlo.
- GPL v3 en todo lo que venga del fork: conservar LICENSE y atribución a DaviAMSilva.

## Entorno

- Python 3.11, torch CPU en local (en la nube puede haber GPU: `train.py` la detecta solo).
- `pip install -r requirements.txt` si falta algo.
- `models/`, `data/raw/` están en .gitignore: se regeneran con
  `python augment_offline.py --per-class 40 --clean && python train.py`.
