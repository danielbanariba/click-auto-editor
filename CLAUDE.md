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

### Publishing service (Telegram approval)

`servicio_publicar.py` runs as a systemd user service
(`systemd/publicar-albumes.service`) and publishes one album at a time:
render → checks → a YouTube search screenshot, then one Telegram message
with the video screenshot and the Subir / Rechazar / Eliminar buttons → on
Subir, private upload and folder deletion (Eliminar deletes without uploading) →
claim check → monetization → publish or schedule. Nothing is uploaded and no other album is rendered
until the user presses a button. It needs `PUBLICAR_TELEGRAM_TOKEN` and
`PUBLICAR_TELEGRAM_CHAT_ID` in `.env` (its own bot: Hermes keeps
`@banar_precios_bot`). Open questions survive restarts in
`data/agente/servicio.json`.

Albums come from `01_limpieza_de_impurezas`, then `Album/`, then `EP/`
(`DIRS_ORIGEN_RENDER` in `config.py`), newest year first. The render moves
each one to the SSD and the folder is deleted after publishing. When
DeathGrind does not know the release type, an `[EP]` suffix in the folder
name makes the title "Full EP".

```bash
journalctl --user -u publicar-albumes -f         # what the service is doing
systemctl --user restart publicar-albumes        # after changing the code
```

Each step is a `pipeline_agente.py` subcommand that prints one JSON line;
logs, screenshots and state go to `data/agente/` (gitignored).

```bash
env/bin/python pipeline_agente.py renderizar             # next album (cola, Album/, EP/) → Renderizando/
env/bin/python pipeline_agente.py verificar --carpeta "<dir>"  # audio/duration/4K checks, screenshot + YouTube search screenshot
env/bin/python pipeline_agente.py subir --carpeta "<dir>"      # private upload, then deletes the album folder
env/bin/python pipeline_agente.py eliminar --carpeta "<dir>"   # deletes the album folder without uploading (already published)
env/bin/python pipeline_agente.py programar --video-id ID --esperar  # claim check → monetization on → public now / first free moment, or +2 years if claimed
env/bin/python pipeline_agente.py eliminar_video --video-id ID  # deletes the video from YouTube (irreversible; the service only runs it from a Telegram button)
env/bin/python pipeline_agente.py impugnar --video-id ID [--sin-enviar]  # disputes every claim of the video in Studio (a legal statement in the user's name); --sin-enviar fills the first form and closes it without sending
env/bin/python pipeline_agente.py estado [--esperar-render]  # unscheduled uploads, rendered albums awaiting approval, render running
```

Scheduling: the videos stay at least one hour apart. An unclaimed video is
made public right away when no other video was published or is scheduled
less than one hour from now (published times come from the newest page of the
channel's uploads, scheduled ones from the API and `programados.jsonl`);
otherwise it is scheduled for the earliest moment that keeps one hour from
every published or scheduled video. Claimed videos go two years ahead, on the
first free hour, and monetization is not attempted for them.

Studio (claim check and monetization) runs Brave headless with a copy of the
signed-in profile (`.runtime_browser_profiles/agente`). It must launch without
Playwright's `--password-store=basic` / `--use-mock-keychain` flags (so Brave
can read the cookies from KWallet) and with a normal Chrome user agent; see
`abrir_contexto_studio`. Claim detection reads `/video/<id>/claims`; an unclear
page returns `reclamo_desconocido` and the service asks the user instead of
guessing, and a missing session returns `sin_sesion_en_studio`.

The claim rows carry no claimant name. `extract_claimants_from_video`
(`11. apelacion.py`) waits up to 30 s for the "Ver detalles" buttons, opens each
one (up to 20 claims), waits for the visible "Reclamantes" header and reads the
block under it. Studio keeps every closed dialog in the DOM, hidden, and appends a
new one per click, so only the visible header counts; a navigation during the wait
(`/copyright` redirects to `/claims`) is treated as "not ready yet". It is read-only (it never clicks "Tomar medidas") and
a browser error leaves that claim out instead of stopping the scheduling.

`lista_negra_reclamantes.txt` (project root, one label per line, `#` lines are
ignored) is the claimant blacklist. For a claimed video, `programar` checks each
claimant line against it, case-insensitively and by containment ("umg" matches
"Believe Music, UMG"), and returns the matches in `reclamantes_en_lista_negra`.
The match never changes the schedule: the video still goes two years ahead. A
missing or unreadable file means no matches.

When `programar` returns matches, the service sends one Telegram question for
that video with its title, the matching claimants, the `/video/<id>/claims` link
and the Studio screenshot (`captura_studio`). "Conservar" only closes the
question. "🗑️ Eliminar video" runs `pipeline_agente.py eliminar_video`, which
checks the video exists, calls `videos.delete` through `ejecutar_api` and prints
one JSON line; the service reports the result. A video that is already gone
(`no_existe`) is reported as already deleted, not as a failure. Deleting cannot be undone, so
nothing deletes a video without that button press. The question is persisted in
`data/agente/servicio.json` like the others, so it survives restarts.

Disputing (`impugnar`) reuses the recipe of `10. inpunar_video.py`: for each claim
whose "Tomar medidas" is visible and enabled it opens the form, picks "Licencia",
ticks the permission and the three declarations, fills `mensaje` and `firma`, and
presses "Enviar". It runs inside the `programar` lock (same Studio profile copy)
and never waits for a 2FA (the service is headless): a verification page, a missing
form or any unexpected page fails that claim without sending and dumps evidence to
`data/impugnar_evidencia/`. It never sends more disputes than there were
disputable claims at the start. A dispute counts as done only when `/claims`,
reloaded, shows exactly one enabled "Tomar medidas" fewer; otherwise that claim is
`no_confirmado` and the run stops, with no retry. Any failure stops the run, and the
JSON reports `reclamos`, `impugnados`, `fallos` and a `captura` of `/claims`. A
video with no disputable claims returns `motivo = sin_reclamos_impugnables`.

Monetization (`studio_monetizacion.py`) turns ads on at
`/video/<id>/monetization`. Studio then asks for the ad-suitability
questionnaire; its answers cannot be changed afterwards. The user's decision
(`RESPUESTA_CUESTIONARIO`) is to always answer "Nada de lo anterior" (his videos
are art) and select no radios. The step sends it (Enviar, then Guardar, then
reloads to confirm ads are on) only if the side panel reads "Adecuado para
anuncios"; with anything else, or no checkbox, it cancels and the video is left
as it was. A failure after Enviar returns `motivo = cuestionario_ya_enviado`.
Any monetization failure never stops the video: the service sends a Telegram
notice with the `/monetization` link of that video.

### Common commands

```bash
python "2. ffmpeg_render.py"                     # render all (parallel by default)
python "2. ffmpeg_render.py" --test              # render 1 folder as smoke test
python "2. ffmpeg_render.py" --test "/path/Band - Album"  # render one specific folder (moves it like a real run)

python "3. subir_video_API.py" --limite 1        # upload 1 video
python "3. subir_video_API.py" --todo --cantidad-lote 24 --gap-horas 1
python "3. subir_video_API.py" --modo-inmediato --limite 1

python "12. mapear_playlists.py" --limite 200
python "10. inpunar_video.py" --aprender         # record new selectors
```

### Tests

```bash
env/bin/python -m pytest tests/ subir_video/ -q  # pipeline, publishing service and helpers
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
7. Muxes audio and moves the finished album folder to `DIR_RENDERIZANDO` (`Renderizando/`); `3. subir_video_API.py` reads `DIR_UPLOAD`

### Upload pipeline (`3. subir_video_API.py`, ~109 KB)

- Authenticates via `subir_video/authenticate.py` (rotates through multiple credential sets in `credentials/`)
- Builds rich video descriptions with streaming links (Bandcamp, Spotify, etc.) and social links
- Schedules uploads in batches at configurable time slots
- Creates/manages playlists per band and genre
- Handles quota exceeded errors by rotating credentials

### Credential rotation (`subir_video/authenticate.py`)

Multiple OAuth credential pairs in `credentials/`:
- `client_secrets_{prefix}_{n}.json` / `token_{prefix}_{n}.json`
- Prefixes: `upload` and `playlists`, several sets each
- Auto-rotates on quota exhaustion via `authenticate_next(prefix)`

### Configuration (`config.py`)

Central config for all scripts. Key settings:
- `BASE_DIR`: root data directory (external drive at `/mnt/Entretenimiento/01_edicion_automatizada`)
- `RENDER_SSD_DIR`: SSD (`/mnt/Render`, btrfs) that holds `Renderizando/`; rendered albums wait there for upload, and nothing renders while it is unmounted
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
- **Commits**: Spanish, Gitmoji + Conventional Commits, following `templates/commit-template.git.txt`
- **Pull requests**: body follows `.github/pull_request_template.md`

## Persistent State Files

- `mapear_playlists_checkpoint.txt`: tracks which videos have been mapped to playlists
- `data/playlist_links_cache.json`: cached playlist IDs per band/genre
- `Ya subidos/` (under `BASE_DIR`): album folders moved there after a successful upload
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
- `12. mapear_playlists.py` runs as a systemd service (`systemd/mapear_playlists.service`, installed as a copy in `/etc/systemd/system/`) with auto-restart
