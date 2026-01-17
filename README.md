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

## Estructura del repo
- `config.py`: rutas y parametros del pipeline.
- `content/`: intro y overlays (incluye `0000000000000000.mp4` y `vhs_noise.mp4`).
- `5. ffmpeg_render.py`: render principal.
- `9. subir_video_API.py`: subida y programacion por API.
- `10. inpunar_video.py`: impugnaciones con Playwright.
- `11. apelacion.py`: apelaciones con Playwright.
- `12. mapear_playlists.py`: playlists por banda/genero.
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
python "9. subir_video_API.py" --todo --cantidad-lote 12 --gap-horas 2
python "9. subir_video_API.py" --modo-inmediato --limite 1

python "10. inpunar_video.py" --aprender
python "11. apelacion.py"
python "12. mapear_playlists.py" --limite 200
```

## Credenciales y entorno
- YouTube: `client_secrets.json` (o `YOUTUBE_CLIENT_SECRETS`) + `token.json`.
- DeathGrind: crear `.env` con `DEATHGRIND_EMAIL` y `DEATHGRIND_PASSWORD` para metadatos.
- Paths/IO: ajustar `BASE_DIR`, `USE_FAST_BASE`, `FAST_BASE_DIR` en `config.py`.

## Legacy
Los pasos antiguos 0-4 y la subida por coordenadas viven en `old/` como referencia.
