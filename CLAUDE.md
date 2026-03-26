# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated pipeline for creating full-album music videos with VHS aesthetics and uploading them to YouTube. Runs on Linux (Arch/CachyOS). Each album folder (audio files + cover image) is rendered into a 4K video and uploaded with auto-generated descriptions, tags, and playlist assignments.

## Quick Start

```bash
pip install -r requirements.txt
pip install -r vhs_effect/requirements.txt   # OpenCV/SciPy for VHS effects
playwright install chromium                  # for YouTube Studio automation
python config.py                             # validate environment + create directories
```

### C++/CUDA renderer (recommended for GPU rendering)

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
# Binary: cpp/build/vhs_render
```

## Running the Pipeline

Scripts are numbered but **not all are sequential**. The active pipeline is:

```
1. verificacion_previa.py   → Pre-flight: checks if album is already on YouTube
2. ffmpeg_render.py         → Main render engine (VHS + CUDA/NVENC)
3. subir_video_API.py       → YouTube upload via Data API (scheduled batches)
10. inpunar_video.py        → Playwright: copyright dispute automation
11. apelacion.py            → Playwright: copyright appeal automation
12. mapear_playlists.py     → Maps videos to playlists (runs as systemd service)
13. filtrar_portadas.py     → NSFW cover art filter (ONNX model)
```

The old Windows/Adobe pipeline (steps 0-8) lives in `old/` as reference only.

### Common commands

```bash
python "2. ffmpeg_render.py"                     # render all (parallel by default)
python "2. ffmpeg_render.py" --test              # render 1 folder as smoke test
python "2. ffmpeg_render.py" --folder "Band - Album"  # render specific folder

python "3. subir_video_API.py" --limite 1        # upload 1 video
python "3. subir_video_API.py" --todo --cantidad-lote 24 --gap-horas 1
python "3. subir_video_API.py" --modo-inmediato --limite 1

python "12. mapear_playlists.py" --limite 200
python "10. inpunar_video.py" --aprender         # record new selectors
```

### Tests

```bash
pytest vhs_effect/test_vhs.py                    # all VHS unit tests
pytest vhs_effect/test_vhs.py -k color_processor # specific test
python vhs_effect/test_vhs.py                    # without pytest
```

## Architecture

### Render pipeline (`2. ffmpeg_render.py`, ~111 KB - largest script)

For each album folder in `DIR_AUDIO_SCRIPTS`:
1. Optionally copies to ramdisk (`/mnt/ramdisk_render`) for fast I/O
2. Reads audio metadata (mutagen), generates tracklist overlay with auto-sized fonts
3. Creates drop shadow for cover image (`effects/sombra.py`)
4. Fetches track titles from DeathGrind API or falls back to filenames
5. Runs profanity censorship (`limpieza/censura.py`)
6. Renders via C++/CUDA binary (`cpp/build/vhs_render`) if `USE_CPP_VHS=True`, else FFmpeg
7. Muxes audio and moves output to `DIR_UPLOAD`

### Upload pipeline (`3. subir_video_API.py`, ~109 KB)

- Authenticates via `subir_video/authenticate.py` (rotates through multiple credential sets in `credentials/`)
- Builds rich video descriptions with streaming links (Bandcamp, Spotify, etc.) and social links
- Schedules uploads in batches at configurable time slots
- Creates/manages playlists per band and genre
- Handles quota exceeded errors by rotating credentials

### Credential rotation (`subir_video/authenticate.py`)

Multiple OAuth credential pairs in `credentials/`:
- `client_secrets_{prefix}_{n}.json` / `token_{prefix}_{n}.json`
- Prefixes: `upload` (2 sets), `playlists` (8 sets)
- Auto-rotates on quota exhaustion via `authenticate_next(prefix)`

### Configuration (`config.py`)

Central config for all scripts. Key settings:
- `BASE_DIR`: root data directory (external drive at `/mnt/Entretenimiento/01_edicion_automatizada`)
- `USE_CPP_VHS`: enable C++/CUDA VHS renderer (default `True`)
- `USE_GPU` / `USE_RAMDISK`: GPU and ramdisk flags
- `ALLOW_FFMPEG_FALLBACK`: fall back to FFmpeg if CUDA fails
- `MAX_PARALLEL_RENDERS`: concurrent render count (currently 1 for 4K)
- `VIDEO_WIDTH`/`VIDEO_HEIGHT`: 3840x2160 (4K)

### Key modules

| Module | Purpose |
|--------|---------|
| `vhs_effect/` | Self-contained Python VHS effect library with CPU/GPU paths |
| `cpp/` | C++/CUDA VHS renderer (targets SM 8.6 / RTX 3090 Ti) |
| `effects/sombra.py` | Drop shadow generation for album covers |
| `limpieza/censura.py` | Profanity filter with leetspeak handling |
| `limpieza/` (other) | Folder cleanup utilities (normalize names, remove junk files) |
| `selectores/` | JSON selector definitions for Playwright automation |
| `analisis_canal/` | Separate 3-step sub-pipeline for YouTube channel analytics |
| `content/` | Intro video (`0000000000000000.mp4`) and VHS noise overlay |

## Code Style

- **Language**: logs, prompts, and commit messages in **Spanish**
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- **Paths**: always use `pathlib.Path`, never `os.path.join()` or string concatenation
- **Imports**: stdlib → third-party → local (see AGENTS.md for full example)
- **Subprocesses**: `subprocess.run([...], check=True)`, avoid `shell=True`
- **Files**: always specify `encoding="utf-8"`
- **Errors**: catch specific exceptions, never bare `except Exception:`
- **Commits**: short Spanish messages, no Conventional Commits prefix

## Persistent State Files

- `mapear_playlists_checkpoint.txt`: tracks which videos have been mapped to playlists
- `data/playlist_links_cache.json`: cached playlist IDs per band/genre
- `bandas-subidas-al-canal.txt`: tracks uploaded video titles to prevent duplicates
- `generos_activos.txt`: tab-separated list of active music genres with IDs

## Credentials and Secrets (never commit)

- `credentials/`: OAuth client secrets and tokens (gitignored)
- `.env`: `DEATHGRIND_EMAIL`, `DEATHGRIND_PASSWORD`
- `client_secrets.json`, `token.json`, `PASS.py`, `.playwright_profile`

## Known Friction

- Render is 4K and heavy; `MAX_PARALLEL_RENDERS=1` is the safe default
- Ramdisk requires `sudo` without password for mount (configured via sudoers)
- CUDA fallback chain: C++/CUDA → FFmpeg NVENC → FFmpeg libx264
- Playwright sleeps are calibrated; reducing them breaks YouTube Studio automation
- `12. mapear_playlists.py` runs as a systemd service (`mapear_playlists.service`) with auto-restart
