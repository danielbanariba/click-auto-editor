<div align="center">
  <h1 align="center">Algoritmos de edicion automatica</h1>
  <p align="center">Pipeline para crear videos 4K de albums completos con estetica VHS y subirlos a YouTube.</p>
</div>

## Resumen
- Render automatico de albums completos (intro + portada + tracklist + VHS).
- Pipeline GPU con C++/CUDA (vhs_render) y fallback a FFmpeg.
- Subidas por YouTube Data API con descripcion automatizada y playlists.
- Automatizacion Playwright para impugnaciones y apelaciones en YouTube Studio.
- Utilidades de limpieza y analisis de canal.
- Filtro de portadas de riesgo: NSFW legacy y revisión de posibles cadáveres reales.

## Estructura del repo
- `config.py`: rutas y parametros del pipeline.
- `content/`: intro y overlays (incluye `0000000000000000.mp4` y `vhs_noise.mp4`).
- `5. ffmpeg_render.py`: render principal.
- `9. subir_video_API.py`: subida y programacion por API.
- `10. inpunar_video.py`: impugnaciones con Playwright.
- `11. apelacion.py`: apelaciones con Playwright.
- `12. mapear_playlists.py`: playlists por banda/genero.
- `13. filtrar_portadas.py`: filtro de portadas de riesgo (NSFW / cadáver real).
- `limpieza/`: utilidades para normalizar y limpiar carpetas.
- `effects/`: utilidades de imagen (sombra de portada, etc.).
- `subir_video/`: helpers de autenticacion y API de YouTube.
- `cpp/`: renderer VHS GPU (C++/CUDA).
- `vhs_effect/`: libreria VHS en Python (tests/examples).
- `analisis_canal/`: pipeline de analisis (ver `analisis_canal/README.md`).
- `selectores/`: JSON de selectores/acciones para Playwright.
- `Comentarios/`: templates de texto legacy.

## Flujo completo (de carpeta a YouTube)
1. Preparacion de carpetas
   - `config.py` define `BASE_DIR` y `DIR_AUDIO_SCRIPTS`.
   - Cada album vive en una carpeta `Banda - Album/` con audios (`.mp3`, `.flac`, `.wav`, `.m4a`) y una portada `cover.*`.
   - Si falta portada, el render intenta extraerla de los metadatos.
   - Scripts de apoyo en `limpieza/` ayudan a normalizar nombres y remover basura.

2. Renderizado (`5. ffmpeg_render.py`)
   - Copia opcional a SSD (`USE_SSD_TEMP`) para evitar bottleneck de I/O.
   - Crea la sombra de la portada (`effects/sombra.py`) y calcula colores para overlays.
   - Genera tracklist overlay (opcional) usando DeathGrind API o nombres de archivos.
   - Usa `content/0000000000000000.mp4` como intro y `content/vhs_noise.mp4` como overlay VHS.
   - Pipeline principal:
     - `USE_CPP_VHS=True`: `cpp/build/vhs_render` (CUDA + NVENC) y mezcla el audio con FFmpeg.
     - Fallback: FFmpeg (NVENC si hay GPU, libx264 si no hay NVENC).
   - Resultado: `Banda - Album.mp4` y la carpeta se mueve a `DIR_UPLOAD`.

3. Subida a YouTube (`9. subir_video_API.py`)
   - Usa YouTube Data API (OAuth) y sube desde `DIR_UPLOAD`.
   - Arma titulo y descripcion (generos, year, tracklist, links).
   - Enriquecimiento opcional con DeathGrind API y `generos_activos.txt`.
   - Programa en lote por defecto (slots separados por horas) o sube inmediato.
   - Crea playlists por banda y genero; al finalizar mueve la carpeta a `DIR_YA_SUBIDOS`.

4. Post-publicacion y mantenimiento
   - `12. mapear_playlists.py`: recorre videos publicos y agrega a playlists por banda/genero.
   - `10. inpunar_video.py` y `11. apelacion.py`: automatizan reclamos en YouTube Studio con Playwright.
   - `analisis_canal/`: extrae datos y genera reportes del canal.

## Instalacion rapida
```sh
pip install -r requirements.txt
pip install -r vhs_effect/requirements.txt  # opcional (OpenCV/SciPy)
playwright install chromium
python config.py  # verifica entorno y crea carpetas
```

### Renderer C++/CUDA (opcional, recomendado)
```sh
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

## Comandos rapidos
```sh
python "5. ffmpeg_render.py"                  # render paralelo (default)
python "5. ffmpeg_render.py" --test           # prueba con 1 video
python "5. ffmpeg_render.py" --parallel       # paralelo explicito

python "9. subir_video_API.py" --limite 1
python "9. subir_video_API.py" --todo --cantidad-lote 24 --gap-horas 1
python "9. subir_video_API.py" --modo-inmediato --limite 1

python "13. filtrar_portadas.py" --politica cadaver-real --backend ollama --limite 20 --dry-run
python "13. filtrar_portadas.py" --politica nsfw --auto-descargar --umbral 0.35

python "10. inpunar_video.py" --aprender
python "10. inpunar_video.py" --auto-detect --max 5
python "10. inpunar_video.py" --auto-detect --solo-detectar
python "11. apelacion.py"
python "11. apelacion.py" --auto-detect --max 5
python "12. mapear_playlists.py" --limite 200
```

## Reclamos automáticos en Studio
- `10. inpunar_video.py --auto-detect`: abre `Contenido`, escanea la tabla y detecta filas con `Derechos de autor`.
- Guarda una cola JSON en `data/impugnar_claims_queue.json` o `data/apelacion_claims_queue.json`.
- Guarda checkpoint en `data/impugnar_claims_checkpoint.txt` o `data/apelacion_claims_checkpoint.txt` para no repetir videos ya procesados.
- Si el perfil interno no tiene sesion, intenta reutilizar cookies activas de Brave/Chromium automaticamente.
- `--solo-detectar`: solo escanea y arma la cola para revision manual.
- `--max-scan`: limita cuantos videos con reclamo detectar en el escaneo.
- `--max`: limita cuantos videos procesar en esa corrida.
- `--reintentos-modal`: reintenta abrir el modal por video cuando Studio responde lento o queda estado residual en corridas largas.

## MCP de navegador (para que cualquier IA lo controle)
- Archivo: `mcp_browser_server.py`
- Objetivo: exponer herramientas MCP para que un agente (Codex/Claude/Desktop, etc.) pueda abrir navegador real, inspeccionar UI y hacer clicks/typing.
- Transporte por defecto: `stdio`.

### Ejecutar el servidor MCP
```sh
./env/bin/python mcp_browser_server.py
```

### Herramientas MCP incluidas
- Navegador general:
  - `browser_start`, `browser_stop`, `browser_status`
  - `browser_goto`, `browser_click`, `browser_type`, `browser_press`
  - `browser_snapshot`, `browser_screenshot`, `browser_eval_js`
- Helpers YouTube Studio:
  - `youtube_ensure_content`
  - `youtube_scan_claims`
  - `youtube_open_claim_modal`
  - `youtube_close_claim_modal`
  - `youtube_dismiss_overlays`

### Notas
- Reutiliza perfil con sesión activa (Brave/Chromium) y cookies Google para entrar a Studio.
- Clona a un runtime profile temporal para evitar bloquear tu perfil principal.
- Para acciones sensibles (impugnaciones), recomienda validar estado visual + logs de respuesta.

## Credenciales y entorno
- YouTube: `client_secrets.json` (o `YOUTUBE_CLIENT_SECRETS`) + `token.json`.
- DeathGrind: crear `.env` con `DEATHGRIND_EMAIL` y `DEATHGRIND_PASSWORD` para metadatos.
- Paths/IO: ajustar `BASE_DIR`, `USE_FAST_BASE`, `FAST_BASE_DIR` en `config.py`.

## Legacy
Los pasos antiguos 0-4 y la subida por coordenadas viven en `old/` como referencia.
