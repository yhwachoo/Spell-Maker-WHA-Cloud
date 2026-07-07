# Mecánica de los sellos — Witch Hat Atelier

Cómo el **tamaño**, los **pesos/simetría** y la **orientación** de los símbolos
determinan la **potencia** y la **dirección** de un hechizo. Resumen de investigación
con fuentes al final. Idea central: **los signos se comportan como fuerzas físicas que
deben equilibrarse dentro del anillo.**

## Anatomía de un sello (glyph)
| Parte | Rol |
|---|---|
| **Sigilo (Rune)** | Centro. Define el **tipo** de magia (fuego, agua, viento, luz, ...). |
| **Signos (Keystones)** | Alrededor del sigilo. **Modifican** forma, tamaño y dirección del efecto. |
| **Anillo (Glyph)** | Círculo exterior. **Conecta y activa**: al cerrarse, se invoca la magia. |

- Todo signo debe **tocar o estar dentro del anillo**; si no, no contribuye.
- Un **anillo solo** (sin signos) = **descarga de energía = explosión**.

## 1. Tamaño → potencia
- **Sello más grande = más potente.** Trazo limpio = más estable y duradero.
- En signos **no-direccionales** y **semi-direccionales**, agrandar el signo sube **la
  fuerza** del efecto (no la dirección).
- **Varios sellos pequeños idénticos enlazados** pueden superar en potencia a **uno
  grande** del mismo tamaño total.

## 2. Pesos / desbalance / "una parte más larga"
- Cada signo ejerce **"presión" proporcional a su tamaño/longitud**.
- Un signo **más largo o más grande** tiene **más poder** → **presión despareja** → el
  hechizo **se desvía hacia ese lado**.
- Regla: el hechizo **se manifiesta hacia donde hay más signos, o los más grandes**.

## 3. Dos zonas iguales / simetría
- Signos **equilibrados (simetría bilateral)** → hechizo **estable y recto**.
- Si está desbalanceado, **añadir más signos** ayuda a **promediar** y reequilibrar.
- Buena práctica: **mantener al menos simetría bilateral**.

## 4. Orientación / rotación
- **Inclinar los signos** → el hechizo **gira**. Más inclinación = **más giro, menos alcance**.
- En signos direccionales, **el ángulo (y a veces el tamaño) fija la dirección**.
- **Invertir/voltear** cambia el comportamiento según el **tipo** de signo (abajo).

## Comportamiento por tipo de signo
| Tipo | Tamaño | Inversión / volteo | Ejemplos |
|---|---|---|---|
| **Direccional** | Ángulo (y a veces tamaño) fija la **dirección**; líneas largas suelen apuntar adentro. | Invertir cambia el patrón (p. ej. *Column* invertido emite omnidireccional, como *Dispersion*). | Column, Pull, Levitation, Region |
| **Semi-direccional** | Tamaño altera **la fuerza**, no la dirección. | Invertir (mirando afuera) = **efecto opuesto**; si es radialmente simétrico, "voltear de adentro hacia afuera". | Crush, Strengthen, Enlarge, Rain |
| **No-direccional** | Tamaño solo cambia **la potencia**. | **No se pueden invertir** (simetría radial). | Float, Crosshair, Bolt, Orb |
| **Asimétrico** | Impredecible. | Se desconoce si invertir/reflejar/rotar cambia algo. | Sign of Wind, Purify |

### Casos concretos
- **Pull:** flechas adentro = atrae; afuera = empuja; en ángulo = torsión.
- **Levitation:** el tamaño define el **peso** que sostiene; la orientación, la dirección.
- **Region:** adentro = confina dentro del anillo; afuera = solo exterior; opuestos = solo el anillo.
- **Convergence:** apuntando adentro = define el **punto de foco**.

## Sistema de vectores (implementado en `signs.py` + `vector_analysis.py`)

### Reglas completas (lore + verificadas en código)

| Caso | Resultado |
|---|---|
| Todos los signos apuntan **misma dirección** | Magia va en esa dirección. Magnitud = suma. |
| Todos apuntan **hacia adentro** (inward) | Convergencia en el centro. Vectores se cancelan pero magia sube por el sigilo. |
| Todos apuntan **hacia afuera** (outward) | Dispersión radial. Vectores se cancelan, magia sale en todas direcciones. |
| Dos signos **apuntándose el uno al otro** | Vectores se cancelan. Magia emana del **espacio entre ellos**. |
| Un signo más **grande/largo** en un lado | Ese vector domina. Hechizo se **desvía** hacia ese lado. |
| Signos en **ángulo oblicuo** | Suma vectorial. Resultado diagonal. |
| Signos **sin vector** (non-directional) | No contribuyen a la dirección. Solo modifican el efecto. |

### Ángulos de posición en el anillo
```
           90 (arriba)
              |
180 (izq) ---O--- 0 (derecha)
              |
           270 (abajo)
```

### Uso del motor de vectores
```python
from signs import resultant_vector

# Ejemplo: lanzallamas (dos Column + un Direccion todos apuntando arriba)
signs = [
    {"slug": "column",    "angle": 90, "size": 1.5},
    {"slug": "column",    "angle": 90, "size": 1.5},
    {"slug": "direccion", "angle": 90, "size": 1.2},
]
r = resultant_vector(signs)
# r["compass"]      -> "arriba"
# r["magnitude"]    -> 4.44  (fuerza combinada)
# r["direction_deg"]-> 90.0
```

```
python vector_analysis.py                        # todos los demos
python vector_analysis.py --demo desvio_por_peso # demo especifico
python vector_analysis.py --compose "column:90:1.5,pull:270:1.0"
python vector_analysis.py --guide                # tabla de vectores
```

### Tabla de vectores por signo

| Signo | Vector base | Invertido | Rota con pos |
|---|---|---|---|
| Column | LOCAL↑ (apunta afuera) | RADIAL (dispersión) | ✓ |
| Levitation | ADENTRO | AFUERA | - |
| Pull | ADENTRO | AFUERA | - |
| Direccion | LOCAL↑ (el pico del chevron) | LOCAL↓ | ✓ |
| Convergencia | ADENTRO | AFUERA | - |
| Dispersion | RADIAL | RADIAL | - |
| Lluvia | RADIAL | RADIAL | - |
| Agrandar | AFUERA (crece) | ADENTRO (encoge) | - |
| Float, Mira, Tornillo, Orb... | sin vector | — | - |

## Implicaciones para el modelo de IA
- **No voltear** en el aumento de datos: el espejo cambia el significado → etiqueta falsa.
- **Rotaciones leves** solamente: la orientación codifica la dirección del hechizo.
- Las features que importan son **geométricas**: tamaño relativo, longitud de trazos,
  simetría izquierda/derecha y ángulo de las flechas.

## Extensión: estimar potencia y dirección (implementada en `analyze_seal.py`)
Las reglas anteriores se pueden **medir directamente de la imagen**, sin entrenar ni
etiquetar, traduciendo el lore a geometría:

| Concepto del lore | Métrica geométrica |
|---|---|
| "Más grande = más potente" | Cobertura de tinta × tamaño del sello → **potencia relativa**. |
| "El lado con signos más grandes desvía el hechizo" | **Offset del centroide** de tinta respecto al centro del anillo → **dirección** (ángulo) y **magnitud** de desvío. |
| "Simetría bilateral = estable y recto" | Diferencia de tinta **mitad izquierda vs derecha** (y arriba/abajo) → **índice de simetría**. |
| "Inclinación → giro" | (Futuro) momento angular de los trazos respecto al centro. |

> Una versión con *deep learning* (segunda cabeza de regresión sobre el backbone)
> requeriría **etiquetas de potencia/dirección** que la obra no da numéricamente; por eso
> el analizador heurístico es el enfoque correcto y honesto por ahora.

## Fuentes
- [Magic — Independent Witch Hat Atelier Wiki](https://witchhatatelier.telepedia.net/wiki/Magic)
- [Signs Explained — Independent Witch Hat Atelier Wiki](https://witchhatatelier.telepedia.net/wiki/Signs_Explained)
- [Magic — Witch Hat Atelier Wiki (Fandom)](https://witch-hat-atelier.fandom.com/wiki/Magic)
- [Witch Hat Atelier's Magic System, Explained — GameRant](https://gamerant.com/witch-hat-atelier-magic-system-explained/)
