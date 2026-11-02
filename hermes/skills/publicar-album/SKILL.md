---
name: publicar-album
description: "Renderiza álbumes de a uno, pide aprobación por Telegram, sube a YouTube y programa según reclamos."
version: 0.1.0
author: Daniel Banariba
platforms: [linux]
metadata:
  hermes:
    tags: [YouTube, Render, Telegram, Copyright]
---

# Publicar álbumes en YouTube con aprobación

Llevás el pipeline de `click-auto-editor` de a un álbum por vez. El trabajo
pesado lo hace `pipeline_agente.py`; vos decidís el orden, le mostrás cada
video al usuario por Telegram y nunca subís nada sin su "subir".

## Cuándo usarla

- "Publicá los álbumes", "seguí con los videos", "arrancá el pipeline".
- Termina un proceso en segundo plano que lanzaste con esta skill.

## Cómo correr los comandos

Todos con la herramienta `terminal`, `workdir` =
`/home/banar/Desktop/click-auto-editor`, y siempre con el Python del proyecto:

```
env/bin/python pipeline_agente.py <comando> [opciones]
```

Cada comando imprime UNA línea JSON al final. Leé `ok` y, si es `false`,
`motivo`. No leas los logs salvo que haga falta explicar un error.

| Comando | Cómo lanzarlo | Qué devuelve |
|---|---|---|
| `estado` | primer plano | `pendientes`, `por_aprobar`, `render_en_curso` |
| `estado --esperar-render` | segundo plano | lo mismo, cuando termina el render en curso |
| `renderizar` | segundo plano | `album`, `carpeta`, `restantes` |
| `verificar --carpeta "<carpeta>"` | primer plano, `timeout=600` | `ok`, `captura`, `resumen` |
| `subir --carpeta "<carpeta>"` | segundo plano | `video_id`, `titulo`, `carpeta_borrada` |
| `programar --video-id <id> --esperar` | segundo plano | `resumen`, `publica_en` |

"Segundo plano" es siempre `background=true`, `notify=true` y
`persist_on_release=true`: un render o una subida duran más que una
conversación y no se pueden cortar a la mitad.

Poné siempre la carpeta entre comillas: los nombres tienen espacios y paréntesis.

## Cómo hablarle al usuario

Estás en su chat de Telegram: tu respuesta de texto le llega sola. No hay
herramienta `send_message`. Para mandarle una foto, corré en primer plano
`hermes send --to telegram` y pasale el texto por stdin con un heredoc, con
la ruta de la imagen en una línea `MEDIA:` al final:

```
hermes send --to telegram <<'FIN'
<texto>
MEDIA:<ruta de la imagen>
FIN
```

Escribí la ruta tal cual la devolvió el comando, sin comillas.

## Procedimiento

1. **Al empezar**, corré `estado`:
   - Por cada `pendientes`, lanzá en segundo plano
     `programar --video-id <id> --esperar`.
   - Si `render_en_curso` es `true`, hay un render que no lanzaste vos (otra
     sesión o el usuario a mano): no lances otro. Lanzá en segundo plano
     `estado --esperar-render` y, cuando termine, volvé a este paso con lo
     que devolvió.
   - Si `por_aprobar` tiene carpetas, son álbumes ya renderizados que esperan
     el OK: seguí desde el paso 4 con la primera.
   - Si no, seguí con el paso 2.
2. **Renderizá** el próximo álbum con `renderizar` en segundo plano. Un solo
   render a la vez: nunca lances otro mientras uno corre.
3. **Cuando termine el render:**
   - `motivo: sin_albumes` → avisá por Telegram que no quedan álbumes y pará.
   - `motivo: sin_espacio` → avisá con los GB libres y pará.
   - `motivo: disco_no_disponible` → el disco de datos no está montado;
     avisá con el `detalle` y pará.
   - `motivo: render_en_curso` → ya hay otro render corriendo; hacé lo mismo
     que en el paso 1.
   - `motivo: render_fallido` → mandá el `album` y las últimas líneas de
     `log_final`. Si `movido_a_error404` es `true`, el álbum ya quedó aparte:
     seguí con el paso 2. Si es `false`, el álbum sigue en la cola y se
     volvería a elegir: pará y esperá instrucciones.
   - `ok: true` → seguí con el paso 4.
4. **Verificá** con `verificar --carpeta "<carpeta>"`. Si devuelve
   `motivo: sin_video`, avisá que el render no dejó el mp4 y volvé al paso 2.
5. **Mostrale el video al usuario** con `hermes send` (ver "Cómo hablarle
   al usuario"): el texto de `resumen` sin cambiarlo y `MEDIA:<captura>`.
   Si `captura` es `null`, mandá solo el `resumen` en tu respuesta y aclará
   que no hubo captura. Si `ok` es `false`, agregá una línea que diga que
   hay tests fallidos.
6. **Preguntale** con `clarify`, una sola pregunta:
   `¿Subo "<album>"?` con `choices: ["subir", "rechazar"]`.
7. **Según la respuesta:**
   - `subir` → lanzá `subir --carpeta "<carpeta>"` en segundo plano.
   - `rechazar` → no toques la carpeta; avisá que quedó en `Renderizando/`
     y volvé al paso 2.
   - Cualquier otro texto → hacé lo que pidió, pero NUNCA subas sin un
     "subir" explícito. Si no está claro, preguntá de nuevo.
   - Sin respuesta (`timed_out` o `undelivered`) → no subas; volvé a
     preguntar.
8. **Cuando termine la subida:**
   - `ok: true` → lanzá `programar --video-id <video_id> --esperar` en
     segundo plano y enseguida volvé al paso 2. La subida ya borró la carpeta;
     si `carpeta_borrada` es `false`, avisá que quedó ocupando espacio.
   - `motivo: limite_canal` → avisá que YouTube no deja subir más por hoy
     (la carpeta sigue en su lugar) y pará hasta que el usuario diga seguir.
   - `motivo: sin_credenciales` → avisá que no hay credenciales con cuota y
     pará.
   - Otro error → avisá con el `motivo` y preguntá si reintentar.
9. **Cuando termine un `programar`:**
   - `ok: true` → mandá el `resumen` por Telegram.
   - `motivo: reclamo_desconocido` → Studio no dejó claro si hay reclamo.
     Mandá la captura con `hermes send` (`MEDIA:<captura_studio>`); si es
     `null`, respondé con el `detalle` y el enlace
     `https://studio.youtube.com/video/<id>/copyright`. Después preguntá
     con `clarify`:
     `¿"<titulo>" tiene reclamo de derechos de autor?` con
     `choices: ["sí", "no"]`. Después corré
     `programar --video-id <id> --reclamo si` (o `no`) en primer plano y
     mandá el `resumen`.
   - `motivo: fallo` o `no_existe` → YouTube rechazó o perdió el video;
     avisá con el `video_id`.
   - Otro error → avisá con el `motivo`; el video queda en `estado` para
     reintentar.

## Reglas

- Nunca subas sin un "subir" del usuario para ESE álbum.
- Nunca borres carpetas a mano: solo `subir` borra, y solo después de subir.
- Nunca cambies fechas a mano: `programar` decide (24 por día, una por hora;
  con reclamo, dos años después).
- Un render a la vez. La subida y el `programar` pueden correr mientras
  renderiza el siguiente.
- Si el usuario escribe algo mientras esperás el botón, la pregunta se
  cancela: atendé lo que dijo y volvé a preguntar.
