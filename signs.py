"""
Catalogo canonico de signos de Witch Hat Atelier.
Ahora incluye la codificacion de VECTORES DIRECCIONALES por signo.

== Sistema de vectores ==

Cada signo tiene un campo "direction" con:

  type:
    "directional"      - la orientacion/angulo del signo controla hacia donde va la magia
    "semi_directional" - el tamano cambia la fuerza; invertir cambia el efecto (no la dir.)
    "non_directional"  - simetria radial; ni rotacion ni inversion cambian el efecto
    "asymmetric"       - comportamiento impredecible; se desconoce el efecto de rotar/reflejar

  base_vec:
    (dx, dy)        - vector LOCAL del signo (frame propio, sin rotar). Convenciones:
                        (0,-1) = apunta ARRIBA    (0, 1) = apunta ABAJO
                        (-1,0) = apunta IZQUIERDA (1, 0) = apunta DERECHA
    "inward"        - siempre hacia el centro del sello (independiente de su rotacion)
    "outward"       - siempre alejandose del centro del sello
    "radial_out"    - se dispersa en TODAS las direcciones desde el centro (sin vector neto)
    None            - sin contribucion direccional

  invert_vec:
    igual que base_vec, pero cuando el signo esta invertido/volteado.
    None = inversion desconocida o sin efecto.

  rotate_with_pos: bool
    True  = al colocar el signo en angulo θ alrededor del sello, su vector local
            rota con θ (el signo 'apunta' hacia donde esta orientado en el anillo).
    False = el vector es absoluto; no cambia segun donde este colocado.

  weight: float
    Factor de contribucion al vector resultante (1.0 = normal).
    Signos mas grandes en el sello = multiplicador mayor (aplicado en analyze_seal.py).

== Logica del vector resultante (reglas del lore) ==

  1. Todos los signos apuntan hacia ADELANTE (outward)  → magia sale hacia afuera.
  2. Todos apuntan hacia ADENTRO (inward)               → magia se concentra / sube por el sigilo.
  3. Dos signos apuntandose el uno al otro              → magia emana del ESPACIO entre ellos.
  4. Signos en distintas direcciones                    → suma vectorial = direccion resultante.
  5. Un lado con signos mas grandes domina              → hechizo se desvia hacia ese lado.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Catálogo principal
# ---------------------------------------------------------------------------

SIGNS: List[Dict[str, Any]] = [

    # ── DIRECTIONAL ─────────────────────────────────────────────────────────

    {
        "slug": "column", "name": "Column", "official": True,
        "effect": "Proyecta la magia como columna/haz en la direccion del signo.",
        "direction": {
            "type": "directional",
            "base_vec": (0, -1),        # punta larga apunta ARRIBA (hacia donde sale el haz)
            "invert_vec": "radial_out", # invertido = omnidireccional (como Dispersion)
            "rotate_with_pos": True,    # rotar el signo cambia la direccion del haz
            "weight": 1.0,
            "notes": "La barra corta queda hacia afuera; el haz sale por el lado largo. "
                     "Invertido emite en todas direcciones."
        }
    },
    {
        "slug": "levitation", "name": "Levitation", "official": True,
        "effect": "Levita el efecto/objetos sobre el sello; con viento = movimiento aereo.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",       # punta de flecha apunta hacia el sigilo (hacia adentro)
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "La punta tipo flecha apunta hacia adentro. El tamano define el peso soportado. "
                     "Con sigilo de viento controla la direccion de movimiento."
        }
    },
    {
        "slug": "pull", "name": "Pull", "official": True,
        "effect": "Atrae hacia el sello lo que coincida con el sigilo.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",       # la punta/flecha apunta hacia el centro = jala hacia aca
            "invert_vec": "outward",    # invertido = empuja hacia afuera
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Flecha apuntando adentro = atrae. Afuera = empuja. "
                     "En angulo = movimiento con torsion."
        }
    },
    {
        "slug": "region", "name": "Region", "official": True,
        "effect": "Define donde se manifiesta la magia respecto al sello.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",       # apuntando adentro = efecto confinado dentro del anillo
            "invert_vec": "outward",    # apuntando afuera = solo afecta exterior
            "rotate_with_pos": False,
            "weight": 0.8,
            "notes": "Adentro = magia solo dentro del anillo. Afuera = solo exterior. "
                     "Signos opuestos = magia solo EN el anillo."
        }
    },
    {
        "slug": "collection", "name": "Collection", "official": True,
        "effect": "Reune materiales situados sobre y alrededor del sello.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",       # lado abierto hacia adentro = recoge hacia el centro
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 0.9,
            "notes": "Lado abierto orientado hacia adentro por defecto."
        }
    },
    {
        "slug": "sights_set", "name": "Sights Set", "official": True,
        "effect": "Permite apuntar/controlar el hechizo con la mente.",
        "direction": {
            "type": "directional",
            "base_vec": (0, -1),        # apunta hacia el objetivo
            "invert_vec": None,
            "rotate_with_pos": True,
            "weight": 1.0,
            "notes": "El angulo del signo determina el vector de apuntado."
        }
    },
    {
        "slug": "gather", "name": "Gather", "official": True,
        "effect": "Similar a Collection; atrae material activamente.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Igual que Collection pero con atraccion activa."
        }
    },

    # ── SIGNOS WIKI (directional) ────────────────────────────────────────────

    {
        "slug": "direccion", "name": "Direccion", "official": True,
        "effect": "Controla la direccion en que se manifiesta la magia.",
        "direction": {
            "type": "directional",
            "base_vec": (0, -1),        # el pico del chevron apunta hacia donde va la magia
            "invert_vec": (0, 1),       # invertido = magia va en sentido contrario
            "rotate_with_pos": True,
            "weight": 1.2,             # signo puro de direccion, peso alto
            "notes": "El pico de la V/chevron define la direccion. "
                     "Todos apuntando mismo lado = magia va ahi. "
                     "Apuntandose mutuamente = magia emana del espacio entre ellos."
        }
    },
    {
        "slug": "recopilacion", "name": "Recopilacion", "official": True,
        "effect": "Reune el material ubicado encima y alrededor del glifo.",
        "direction": {
            "type": "directional",
            "base_vec": "inward",       # lado abierto hacia adentro = recoge hacia aca
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 0.9,
            "notes": "Lado abierto suele orientarse hacia adentro."
        }
    },
    {
        "slug": "pajaro", "name": "Pajaro", "official": True,
        "effect": "Genera una proyeccion con forma de pajaro que vuela.",
        "direction": {
            "type": "directional",
            "base_vec": (0, -1),        # alas/chevron apuntan hacia arriba = direccion de vuelo
            "invert_vec": (0, 1),
            "rotate_with_pos": True,
            "weight": 1.0,
            "notes": "La parte frontal (chevron) puede ser parte separada del signo. "
                     "El sigilo central va dentro. Direccion = donde apunta el frente del pajaro."
        }
    },
    {
        "slug": "doblar", "name": "Doblar", "official": False,
        "effect": "Dobla/altera la realidad; efecto variable segun contexto.",
        "direction": {
            "type": "directional",
            "base_vec": (0, -1),        # la curva J redirige la magia; la apertura apunta arriba
            "invert_vec": (0, 1),       # espejado = dobla en sentido opuesto
            "rotate_with_pos": True,
            "weight": 0.8,
            "notes": "Signo asimetrico: espejarlo cambia el sentido del doblez. "
                     "Efectos de Ojo y Doblar son indistinguibles en los ejemplos conocidos."
        }
    },

    # ── SEMI-DIRECTIONAL ─────────────────────────────────────────────────────

    {
        "slug": "dispersion", "name": "Dispersion", "official": True,
        "effect": "La magia se derrama hacia afuera en todas direcciones.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "radial_out",   # se dispersa en TODAS las direcciones
            "invert_vec": "radial_out", # ambas variantes vistas; diferencia poco clara
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Similar a Column pero en todas direcciones. "
                     "Sin vector neto: la magia sale por todos lados."
        }
    },
    {
        "slug": "crush", "name": "Crush", "official": True,
        "effect": "Pulveriza objetos; invertido los recompone temporalmente.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,
            "invert_vec": None,         # invertido = efecto opuesto (recomponer)
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Sin componente direccional. Tamano = fuerza de aplastamiento."
        }
    },
    {
        "slug": "convergencia", "name": "Convergencia", "official": True,
        "effect": "La magia converge hacia un unico punto central.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "inward",       # el vertice del triangulo apunta hacia adentro
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 1.1,
            "notes": "Vertice tipicamente apunta hacia adentro. "
                     "Tambien puede rigidizar particulas sueltas (arena -> solido)."
        }
    },
    {
        "slug": "weave", "name": "Weave", "official": True,
        "effect": "Convierte solidos en cintas largas y flexibles.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Sin direccion propia. Debe rodear el sigilo central."
        }
    },
    {
        "slug": "strengthen", "name": "Strengthen", "official": True,
        "effect": "Hace los objetos mas fuertes y duraderos.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Tamano = grado de refuerzo. Sin componente direccional."
        }
    },
    {
        "slug": "entwine", "name": "Entwine", "official": True,
        "effect": "Hace que un objeto se enrosque alrededor de otros.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "outward",      # el enroscamiento va hacia afuera del sello
            "invert_vec": "inward",
            "rotate_with_pos": False,
            "weight": 0.8,
            "notes": "El objeto se extiende hacia afuera para enroscarse."
        }
    },
    {
        "slug": "aeriforms_defined", "name": "Aeriforms Defined", "official": True,
        "effect": "Modificador del sigilo de viento.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 0.7,
            "notes": "Modifica como se comporta el viento; efectos exactos poco claros."
        }
    },
    {
        "slug": "glaives", "name": "Glaives", "official": True,
        "effect": "Define cuan profundo se incrusta la magia en la carne.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "inward",       # la incrustacion va hacia adentro del objetivo
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 0.9,
            "notes": "Solo visto con sigilos de criatura."
        }
    },
    {
        "slug": "enlarge", "name": "Enlarge", "official": False,
        "effect": "Los objetos crecen (esquinas afuera) o encogen (esquinas adentro).",
        "direction": {
            "type": "semi_directional",
            "base_vec": "outward",      # esquinas apuntando afuera = crecimiento
            "invert_vec": "inward",     # esquinas adentro = encogimiento
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "La orientacion de las esquinas controla grow vs shrink."
        }
    },
    {
        "slug": "agrandar", "name": "Agrandar", "official": False,
        "effect": "Los objetos crecen (esquinas afuera) o encogen (esquinas adentro).",
        "direction": {
            "type": "semi_directional",
            "base_vec": "outward",
            "invert_vec": "inward",
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Equivalente espanol de Enlarge."
        }
    },
    {
        "slug": "rain", "name": "Rain", "official": False,
        "effect": "Produce un efecto similar a lluvia en el area inmediata.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "outward",      # la lluvia cae/emana hacia afuera desde el centro
            "invert_vec": "inward",
            "rotate_with_pos": False,
            "weight": 0.9,
            "notes": "Efecto de area que radia desde el sello hacia afuera."
        }
    },
    {
        "slug": "lluvia", "name": "Lluvia", "official": True,
        "effect": "Genera magia de forma similar a la precipitacion en su area inmediata.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "radial_out",   # los rayos del cuadrado van en todas direcciones
            "invert_vec": "radial_out",
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Debe rodear el sigilo. El espacio vacio en el centro es para el sigilo. "
                     "Funciona con agua, luz, fuego u otras magias."
        }
    },
    {
        "slug": "radial", "name": "Radial", "official": False,
        "effect": "Probablemente reduce la potencia del hechizo.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,           # el arco no tiene componente direccional claro
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 0.5,             # su funcion principal es REDUCIR potencia
            "notes": "El arco sugiere curvatura/redireccion. "
                     "Convierte fuego en calor (reduccion de efectividad)."
        }
    },
    {
        "slug": "bind", "name": "Bind", "official": False,
        "effect": "Actua como pegamento; mantiene un objeto unido.",
        "direction": {
            "type": "semi_directional",
            "base_vec": "inward",       # junta hacia adentro (cohesion)
            "invert_vec": "outward",
            "rotate_with_pos": False,
            "weight": 0.8,
            "notes": "Funcion de cohesion/adherencia."
        }
    },
    {
        "slug": "tejer", "name": "Tejer", "official": True,
        "effect": "Convierte solidos en cintas largas y flexibles.",
        "direction": {
            "type": "semi_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Debe rodear el sigilo central. Sin componente vectorial propio."
        }
    },
    {
        "slug": "marioneta_bailarina", "name": "Marioneta Bailarina", "official": True,
        "effect": "Hace que los objetos se muevan rapidamente en el aire (con sigilo de viento).",
        "direction": {
            "type": "semi_directional",
            "base_vec": "radial_out",   # el movimiento es omnidireccional rapido
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 0.9,
            "notes": "El sigilo central va en el espacio vacio del centro. "
                     "El circulo puede estar dentro del anillo del hechizo o ser parte de el."
        }
    },

    # ── NON-DIRECTIONAL ──────────────────────────────────────────────────────

    {
        "slug": "float", "name": "Float", "official": True,
        "effect": "Hace flotar las cosas ignorando la gravedad.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Simetria radial. No se puede invertir. Tamano = capacidad de carga."
        }
    },
    {
        "slug": "billow", "name": "Billow", "official": True,
        "effect": "Genera una nube mullida y comoda.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "No direccional. Puede reemplazar un sigilo dentro del hechizo."
        }
    },
    {
        "slug": "ondulante", "name": "Ondulante", "official": True,
        "effect": "Transforma el material en una nube esponjosa.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Flor 4 petalos = simetria radial. Puede reemplazar un sigilo."
        }
    },
    {
        "slug": "repeticion", "name": "Repeticion", "official": True,
        "effect": "Resetea continuamente los objetos afectados a su estado anterior.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Simbolo circular. Puede reemplazar un sigilo dentro del hechizo."
        }
    },
    {
        "slug": "cool", "name": "Cool", "official": True,
        "effect": "Enfria las cosas.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Solo ajusta temperatura. Sin componente direccional."
        }
    },
    {
        "slug": "diamond", "name": "Diamond", "official": False,
        "effect": "El hechizo solo afecta objetos cercanos, no el objeto dibujado.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Controla el OBJETIVO del hechizo, no su direccion."
        }
    },
    {
        "slug": "diamante", "name": "Diamante", "official": False,
        "effect": "El hechizo solo afecta objetos cercanos, no el objeto dibujado.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Equivalente espanol de Diamond."
        }
    },
    {
        "slug": "window", "name": "Window", "official": False,
        "effect": "El hechizo solo afecta al objeto sobre el que se dibuja.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Controla el OBJETIVO, no la direccion."
        }
    },
    {
        "slug": "ventana", "name": "Ventana", "official": False,
        "effect": "El hechizo solo afecta al objeto sobre el que se dibuja.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Equivalente espanol de Window."
        }
    },
    {
        "slug": "crosshair", "name": "Crosshair", "official": False,
        "effect": "Apunta solo a objetos con el aspecto magico correspondiente.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Cruz simetrica. Filtra el objetivo, no la direccion."
        }
    },
    {
        "slug": "mira", "name": "Mira", "official": False,
        "effect": "Hace que la magia solo se manifieste en objetos del mismo aspecto magico.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Cruz (+) simetrica. Tres funciones posibles segun contexto."
        }
    },
    {
        "slug": "bolt", "name": "Bolt", "official": False,
        "effect": "La magia se manifiesta como proyectiles tipo dardo.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "La direccion de los proyectiles la controlan los signos direccionales del sello."
        }
    },
    {
        "slug": "tornillo", "name": "Tornillo", "official": False,
        "effect": "Provoca que la magia se manifieste en forma de rayos.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Al combinarse con un signo direccional, los rayos se disparan a alta velocidad."
        }
    },
    {
        "slug": "orb", "name": "Orb", "official": False,
        "effect": "Crea un espacio esferico donde se acumula el material controlado.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "La esfera se forma sobre el sello. Sin vector propio."
        }
    },
    {
        "slug": "vision", "name": "Vision", "official": False,
        "effect": "Ligado a percepcion visual y manipulacion de luz. Crea ilusiones con Ojo.",
        "direction": {
            "type": "non_directional",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Estrella de 8 rayos = simetria radial. No necesita sigilo para funcionar. "
                     "Debe colocarse en el centro del hechizo."
        }
    },

    # ── ASYMMETRIC / UNKNOWN ─────────────────────────────────────────────────

    {
        "slug": "sign_of_wind", "name": "Sign of Wind", "official": True,
        "effect": "Propiedades ligadas al viento; funcion poco clara.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Comportamiento desconocido al invertir/reflejar/rotar."
        }
    },
    {
        "slug": "purify", "name": "Purify", "official": False,
        "effect": "Separa las impurezas de la magia de su sigilo.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Asimetrico. Desconocido si invertir/rotar cambia el efecto."
        }
    },
    {
        "slug": "eye", "name": "Eye", "official": False,
        "effect": "Ligado a ilusiones y manipulacion de luz. Efectos indistinguibles de Doblar.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Solo 2 apariciones. Con Vision y Doblar funde objetos con las sombras."
        }
    },
    {
        "slug": "ojo", "name": "Ojo", "official": False,
        "effect": "Ligado a ilusiones y manipulacion de luz. Efectos indistinguibles de Doblar.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Equivalente espanol de Eye."
        }
    },
    {
        "slug": "link", "name": "Link", "official": False,
        "effect": "Vincula la magia entre el objeto dibujado y otros del mismo origen.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 0.8,
            "notes": "Funcion de vinculo entre objetos. Sin direccion clara."
        }
    },
    {
        "slug": "puppet", "name": "Puppet", "official": False,
        "effect": "Controla con la mente el movimiento del objeto dibujado.",
        "direction": {
            "type": "asymmetric",
            "base_vec": None,
            "invert_vec": None,
            "rotate_with_pos": False,
            "weight": 1.0,
            "notes": "Direccion controlada mentalmente, no por el signo en si."
        }
    },
]

# ---------------------------------------------------------------------------
# Sinonimos: el mismo signo aparece con slug en ingles y en espanol.
# Se agrupan para no contarlos como clases distintas (p. ej. al validar el
# matcher o al fusionar plantillas). El primero de cada grupo es el canonico.
# ---------------------------------------------------------------------------
SYNONYM_GROUPS: List[Tuple[str, ...]] = [
    ("enlarge", "agrandar"),
    ("diamond", "diamante"),
    ("window", "ventana"),
    ("crosshair", "mira"),
    ("eye", "ojo"),
    ("collection", "recopilacion"),
    ("rain", "lluvia"),
    ("weave", "tejer"),
    ("puppet", "marioneta_bailarina"),
    ("repeticion", "repetition"),
]

_CANON = {}
for _grp in SYNONYM_GROUPS:
    for _s in _grp:
        _CANON[_s] = _grp[0]


def canonical_slug(slug: str) -> str:
    """Devuelve el slug canonico (ingles) si el signo tiene sinonimos."""
    return _CANON.get(slug, slug)


# ---------------------------------------------------------------------------
# Mapas de acceso rápido
# ---------------------------------------------------------------------------
SLUG_TO_NAME   = {s["slug"]: s["name"]      for s in SIGNS}
SLUG_TO_INFO   = {s["slug"]: s              for s in SIGNS}
SLUG_TO_DIREC  = {s["slug"]: s["direction"] for s in SIGNS}


def display_name(slug: str) -> str:
    return SLUG_TO_NAME.get(slug, slug)


def get_direction(slug: str) -> Optional[Dict]:
    return SLUG_TO_DIREC.get(slug)


# ---------------------------------------------------------------------------
# Motor de vectores
# ---------------------------------------------------------------------------

def _rotate_vec(dx: float, dy: float, angle_rad: float) -> Tuple[float, float]:
    """Rota un vector (dx,dy) por angle_rad (sentido antihorario)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return dx*c - dy*s, dx*s + dy*c


def sign_vector(slug: str, position_angle_deg: float,
                size_factor: float = 1.0) -> Optional[Tuple[float, float]]:
    """
    Devuelve el vector de contribucion (dx, dy) de un signo colocado
    en position_angle_deg dentro de un sello.

    position_angle_deg: angulo de la POSICION del signo en el anillo
      (0=derecha, 90=arriba, 180=izquierda, 270=abajo).
      Para signos rotate_with_pos=True, tambien define hacia donde apunta:
        90 (arriba en el anillo) -> el signo apunta hacia arriba.
      Para inward/outward, define desde que punto del anillo actua.

    size_factor: relativo al tamano promedio de los signos del sello (1.0=normal).

    Retorna None para signos sin componente direccional.

    Sistema de coordenadas de imagen (y crece hacia abajo):
      (0,-1) = arriba    (0, 1) = abajo
      (1, 0) = derecha  (-1, 0) = izquierda
    """
    info = SLUG_TO_DIREC.get(slug)
    if info is None:
        return None

    bv = info["base_vec"]
    w  = info["weight"] * size_factor

    if bv is None:
        return None

    theta = math.radians(position_angle_deg)

    if bv == "inward":
        # Apunta desde la posicion del signo HACIA el centro del sello.
        # En coords imagen: posicion (cos θ, -sin θ) → inward = (-cos θ, sin θ)
        return (-math.cos(theta) * w, math.sin(theta) * w)

    if bv == "outward":
        # Apunta DESDE el centro hacia la posicion del signo.
        return (math.cos(theta) * w, -math.sin(theta) * w)

    if bv == "radial_out":
        # Sin vector neto (dispersion simetrica).
        return (0.0, 0.0)

    # Vector local (dx, dy):
    # Si rotate_with_pos=True, el local-UP del signo apunta hacia AFUERA
    # del anillo en la posicion angle. La rotacion correcta es (pi/2 - theta).
    dx, dy = bv
    if info["rotate_with_pos"]:
        dx, dy = _rotate_vec(dx, dy, math.pi / 2 - theta)
    return (dx * w, dy * w)


def resultant_vector(sign_list: List[Dict]) -> Dict:
    """
    Calcula el vector resultante de una lista de signos en un sello.

    sign_list: lista de dicts con:
      {"slug": str, "angle": float, "size": float}
      angle = grados (0=derecha, 90=arriba)
      size  = tamano relativo (1.0=normal)

    Devuelve:
      {"dx": float, "dy": float,
       "magnitude": float,
       "direction_deg": float,   # angulo del resultante (0=der, 90=arr)
       "compass": str,           # descripcion en espanol
       "is_radial": bool,        # True si todos los signos son radial_out
       "has_inward": bool,
       "has_outward": bool,
       "active_signs": int}
    """
    vx, vy = 0.0, 0.0
    has_inward = has_outward = is_radial = False
    active = 0

    for s in sign_list:
        slug  = s.get("slug", "")
        angle = s.get("angle", 0.0)
        size  = s.get("size", 1.0)
        info  = SLUG_TO_DIREC.get(slug)
        if info is None:
            continue
        bv = info["base_vec"]
        if bv == "radial_out":
            is_radial = True
            active += 1
            continue
        if bv == "inward":
            has_inward = True
        if bv == "outward":
            has_outward = True
        vec = sign_vector(slug, angle, size)
        if vec is not None:
            vx += vec[0]; vy += vec[1]
            active += 1

    mag = math.hypot(vx, vy)
    if mag > 1e-9:
        angle_deg = math.degrees(math.atan2(-vy, vx)) % 360  # -vy: coords imagen
    else:
        angle_deg = 0.0

    # Descripcion en espanol
    dirs = ["derecha", "arriba-derecha", "arriba", "arriba-izquierda",
            "izquierda", "abajo-izquierda", "abajo", "abajo-derecha"]

    if is_radial and mag < 0.05:
        compass = "dispersion radial (todas direcciones)"
    elif active >= 2 and mag < 0.08:
        # Los vectores se cancelan: o todos adentro (convergencia) o se apuntan
        if has_inward and not has_outward:
            compass = "convergencia en el centro del sello"
        else:
            compass = "magia emana del espacio entre los signos"
    elif mag < 0.05:
        compass = "centrado (sin direccion dominante)"
    else:
        compass = dirs[int((angle_deg + 22.5) / 45) % 8]

    return {
        "dx": round(vx, 4), "dy": round(vy, 4),
        "magnitude": round(mag, 4),
        "direction_deg": round(angle_deg, 1),
        "compass": compass,
        "is_radial": is_radial,
        "has_inward": has_inward,
        "has_outward": has_outward,
        "active_signs": active
    }
