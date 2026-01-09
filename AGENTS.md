# Repository Guidelines

## Project Structure & Module Organization
- Root contains numbered pipeline scripts like `0. limpieza_de_impurezas.py` and `5. ffmpeg_render.py`. Keep new pipeline steps in this convention.
- `limpieza/` hosts cleanup helpers; `effects/` contains image/VHS utilities; `subir_video/` holds upload/API helpers; `vhs_effect/` is a standalone VHS effect library with examples/tests.
- `cpp/` stores the C++/CUDA VHS renderer; `cpp/build/` is generated output.
- `content/` holds intro/overlay assets; `Comentarios/` stores text templates; `bandas-subidas-al-canal.txt` tracks uploaded titles.
- `config.py` centralizes paths and runtime options (external drive, FFmpeg, GPU settings).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs core dependencies.
- `pip install -r vhs_effect/requirements.txt` installs VHS effect dependencies (OpenCV/SciPy).
- `playwright install chromium` is required for Playwright-based scraping.
- `python config.py` validates mount points and creates required folders.
- Run pipeline scripts directly, for example:
  - `python "0. limpieza_de_impurezas.py"`
  - `python "5. ffmpeg_render.py" --test`
- C++/CUDA build: `cmake -S cpp -B cpp/build` then `cmake --build cpp/build`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, no tabs; use snake_case for functions/vars and PascalCase for classes.
- Keep user-facing logs/prompts in Spanish to match existing scripts.
- Maintain the numbered script naming pattern (`N. descripcion.py`) for pipeline steps.
- C++/CUDA: follow existing file layout (`.hpp`, `.cpp`, `.cu`) and keep formatting consistent with nearby files. No formatter is enforced.

## Testing Guidelines
- No unified test runner. For VHS module smoke tests: `python vhs_effect/test_vhs.py` (requires OpenCV).
- Use `vhs_effect/examples/` for manual visual checks.
- For pipeline changes, run the affected step on a small sample folder and verify output files.

## Commit & Pull Request Guidelines
- Commit messages are short, Spanish, and descriptive (e.g., `Barra de progreso`); avoid Conventional Commits prefixes.
- PRs should include: summary, commands run, any updated paths/coordinates, and screenshots or short clips for UI/visual changes. Link issues when applicable.

## Configuration & Automation Notes
- GUI automation uses fixed screen coordinates (see `8. subir_video_coordenadas.py`, `10. inpunar_video.py`, `11. apelacion.py`); test at the target resolution and update coordinates carefully.
- Keep machine-specific paths in `config.py` rather than hard-coding them in scripts.
