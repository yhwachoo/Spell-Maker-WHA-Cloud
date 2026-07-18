---
title: WHA Reconocedor de Sellos
emoji: 🪄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
---

# Witch Hat Atelier · Reconocedor de Sellos

Banco de pruebas para **reconocer y componer sellos mágicos** de Witch Hat Atelier.

- **Reconocer**: sube una imagen, dibújala en el navegador o elige un sello del
  catálogo → detecta los signos (clasificador entrenado, robusto a baja resolución),
  calcula el vector resultante y describe el efecto.
- **Componer**: elige sigilo + signos y predice el comportamiento del hechizo.

Este Space es un contenedor Docker que clona el repo
[Spell-Maker-WHA-Cloud](https://github.com/yhwachoo/Spell-Maker-WHA-Cloud),
instala las dependencias de runtime y sirve la app Flask con gunicorn.

Símbolos derivados del proyecto **wha-spell-maker** de DaviAMSilva (GPL v3);
la magia de Witch Hat Atelier es creación de Kamome Shirahama.
