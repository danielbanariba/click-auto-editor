## AGENTS Handbook
Bienvenido/a. Este archivo es para agentes (humanos o LLMs) que trabajen en este repositorio. Sigue estas reglas para no romper el pipeline ni los assets.

## TL;DR Operativo
- Instala dependencias base: `pip install -r requirements.txt`.
- Instala dependencias VHS opcionales: `pip install -r vhs_effect/requirements.txt`.
- Instala navegadores Playwright: `playwright install chromium`.
- Valida y crea carpetas: `python config.py`.
- Render rápido de prueba: `python "2. ffmpeg_render.py" --test`.
- Subida mínima por API: `python "3. subir_video_API.py" --limite 1`.
- No hay reglas Cursor ni Copilot en este repo.

## Mapa del Repo
- Scripts numerados en la raíz (`1. verificacion_previa.py`, `2. ffmpeg_render.py`, `3. subir_video_API.py`, `10. inpunar_video.py`, `11. apelacion.py`, `12. mapear_playlists.py`). Respeta el patrón `N. descripcion.py`.
- `config.py`: rutas (disco externo, staging NVMe), flags GPU/NVENC, opciones de render.
- `limpieza/`: helpers de normalización y limpieza de carpetas/archivos.
- `effects/`: utilidades de imagen y VHS (`sombra`, overlays).
- `subir_video/`: autenticación y YouTube Data API.
- `vhs_effect/`: librería VHS autocontenida + tests y ejemplos manuales.
- `cpp/`: renderer VHS C++/CUDA; `cpp/build/` es generado.
- `content/`: intro y overlays (`0000000000000000.mp4`, `vhs_noise.mp4`).
- `Comentarios/`: plantillas de textos legacy.
- `analisis_canal/`, `selectores/`, `old/`: pipelines de análisis, selectores Playwright, scripts legacy de coordenadas.

## Setup Detallado
- Python 3.x, sin gestor especial; venv opcional (`python -m venv env && source env/bin/activate`).
- Dependencias principales en `requirements.txt`; no uses `pip_install.ps1` antiguo.
- Dependencias VHS (OpenCV/SciPy) opcionales en `vhs_effect/requirements.txt`.
- Playwright requiere instalación de navegador tras `pip install playwright` (`playwright install chromium`).
- C++/CUDA: `cmake -S cpp -B cpp/build && cmake --build cpp/build`; bin esperado en `cpp/build/vhs_render`.
- Ejecuta `python config.py` para verificar rutas montadas y crear directorios en disco externo/NVMe.

## Comandos Frecuentes
- Render principal: `python "2. ffmpeg_render.py"` (usa GPU NVENC si disponible, respeta `MAX_PARALLEL_RENDERS`).
- Render de smoke: `python "2. ffmpeg_render.py" --test` procesa 1 carpeta.
- Subir por API: `python "3. subir_video_API.py" --todo --cantidad-lote 24 --gap-horas 1` o `--limite 1` para prueba.
- Impugnación manual asistida: `python "10. inpunar_video.py" --aprender`.
- Apelación: `python "11. apelacion.py"`.
- Mapear playlists: `python "12. mapear_playlists.py" --limite 200`.
- VHS python tests (todo el archivo): `python vhs_effect/test_vhs.py`.
- VHS con pytest y test único (si tienes pytest): `pytest vhs_effect/test_vhs.py -k color_processor`.

## Tests y QA
- No hay suite unificada; `vhs_effect/test_vhs.py` ejecuta asserts y genera `test_output.jpg` para inspección manual.
- Requiere OpenCV y SciPy (instala `vhs_effect/requirements.txt`).
- Para cambios en render, corre `python "2. ffmpeg_render.py" --test` sobre 1 carpeta y revisa output.
- Para Playwright, valida credenciales y perfil en `.playwright_profile` antes de ejecutar scripts de estudio.
- No hay linting configurado (sin black/flake8); revisa estilo a mano.

## Estilo Python
- Indentación de 4 espacios; sin tabs.
- Nombres snake_case para funciones/variables; PascalCase para clases; constantes en MAYÚSCULAS.
- Orden de imports: stdlib, terceros, locales; usa `pathlib.Path` en vez de strings cuando toques rutas.
- Logs y prompts siempre en español; evita mezclar idiomas en UX.
- Prefiere mensajes claros con emojis/✔️ ya usados; conserva tono imperativo y breve.
- Evita comentarios innecesarios; solo explica bloques no obvios.
- Evita prints silenciados; usa excepciones con contexto en español en lugar de `pass` silencioso.
- No uses type hints pesados; cuando añadas, usa `Optional`, `Path`, `list[str]`, y conserva el estilo existente (muchos archivos sin tipos).
- No introduzcas frameworks de logging; sigue prints sencillos.
- Reutiliza helpers de `config.py` para rutas y flags; no dupliques constantes.

## Patrones de Código
- Prefiere funciones pequeñas y puras; evita lógica duplicada entre scripts numerados.
- Para rutas usa `Path` y métodos (`.exists()`, `.mkdir()`) en lugar de `os.path` cuando toques código nuevo.
- Evita globales mutables; si necesitas estado, pásalo como argumentos o usa dataclasses ligeras.
- Cuando uses multiprocessing (`ProcessPoolExecutor`), protege entrada con `if __name__ == "__main__":`.
- Para comandos externos usa `subprocess.run` con `check=True` y listas, no cadenas con `shell=True` salvo que sea imprescindible.
- Normaliza texto con `unidecode` donde se hace matching de títulos; respeta el comportamiento existente.
- Mantén los mensajes en consola consistentes con los emojis y formatos actuales.
- Si agregas JSON/YAML, mantén UTF-8 y evita valores nulos innecesarios.
- No mezcles PIL y OpenCV sin convertir BGR↔RGB explícitamente.
- Cuando abras archivos, usa `encoding="utf-8"` y `newline=""` si escribes CSV.

## Manejo de Errores y Retries
- No uses `except Exception:` vacío; captura errores específicos y loguea en español.
- Mantén reintentos y backoff para API DeathGrind (`DELAY_BASE_429`, `MAX_RETRIES_429`).
- Cuando el renderer CUDA falle, respeta flags de fallback (`ALLOW_FFMPEG_FALLBACK`, `DISABLE_CPP_ON_CUDA`, `CUDA_FAIL_FAST`).
- Para I/O, comprueba existencia de rutas antes de operar; crea carpetas con `mkdir(parents=True, exist_ok=True)` como en `config.py`.
- No borres archivos de usuario ni muevas a producción sin copia; usa staging NVMe (`STAGING_*`) si ya está activado.

## Datos y Credenciales
- No subas `client_secrets.json`, `token.json` ni `.env`; están presentes localmente.
- DeathGrind usa `.env` con `DEATHGRIND_EMAIL` y `DEATHGRIND_PASSWORD`; no los escribas en código ni en logs.
- Paths a discos externos vienen de `BASE_DIR` y `FAST_BASE_DIR`; evita hardcodear rutas absolutas nuevas.
- `bandas-subidas-al-canal.txt` controla duplicados; edítalo con cuidado y siempre en UTF-8.

## Playwright y Automatización
- `selectores/` contiene JSON de acciones; respeta su forma si agregas pasos.
- Scripts de estudio (`10. inpunar_video.py`, `11. apelacion.py`) asumen un perfil en `.playwright_profile`; no lo limpies.
- Ajusta tiempos con cautela; sleeps demasiado cortos rompen la automatización.

## VHS Effect (Python)
- `vhs_effect` es independiente; imports relativos ya configurados en tests con `sys.path.insert`.
- Usa OpenCV en BGR; no mezcles con PIL sin convertir.
- Guarda salidas temporales fuera de assets (`test_output.jpg` en raíz está bien para smoke).

## Renderer C++/CUDA
- Código vive en `cpp/` con headers `.hpp` y fuentes `.cpp/.cu`.
- Mantén formato existente; no hay clang-format configurado.
- Bin esperado: `cpp/build/vhs_render`; scripts Python lo invocan cuando `USE_CPP_VHS=True`.
- Si cambias flags NVENC/CUDA, refleja el cambio en `config.py` y en las llamadas del renderer.

## Formato y Naming de Archivos
- Nuevos pasos del pipeline siguen patrón `NN. descripcion.py` sin huecos.
- Evita caracteres raros en nombres; mantén ASCII por defecto.
- Recursos van en `content/`; no los dupliques.

## Commits y PRs
- Mensajes cortos en español, descriptivos (`Barra de progreso`, `Arreglo render NVENC`). Sin prefijos tipo Conventional Commits.
- Incluye en PR: resumen, comandos ejecutados, cambios en rutas/coords, capturas o clips si tocas UI/visuales.
- No empujes credenciales ni cambios en assets binarios pesados sin aviso.

## Seguridad y Limpieza
- No uses `git reset --hard` ni toques cambios del usuario.
- No añadas nuevas dependencias sin necesidad; prioriza stdlib.
- Evita dormir hilos largos en código nuevo; usa tiempos existentes como referencia.

## Documentos Existentes
- README.md cubre flujo completo y comandos rápidos; mantenlo sincronizado.
- CLAUDE.md contiene contexto antiguo (Windows + Adobe); útil para histórico pero el pipeline activo es FFmpeg/Playwright.
- No hay `.cursor/rules` ni `.github/copilot-instructions.md` a la fecha.

## Cómo Pedir Ayuda (para agentes)
- Si una instrucción es ambigua y afecta producción (rutas, credenciales, costos), pregunta con una sola duda concreta y propone tu default.
- En tareas simples, actúa sin pedir permiso y resume lo que hiciste.

## Checklist Rápido Antes de Subir Cambios
- [ ] Dependencias instaladas (`requirements.txt`, opcional `vhs_effect/requirements.txt`).
- [ ] Playwright instalado (`playwright install chromium`) si tocas scripts de estudio.
- [ ] `python config.py` corre sin errores en tu entorno.
- [ ] Prueba mínima del cambio (render `--test`, pytest -k si aplica, o smoke manual en VHS).
- [ ] Sin secretos nuevos en git status.
- [ ] Mensaje de commit en español.

## Fricción Conocida
- Coordinadas de GUI son frágiles; no ajustes sin probar en la resolución objetivo.
- Render 4K es pesado; `MAX_PARALLEL_RENDERS=4` es el techo sugerido con 3090 Ti.
- Fallback FFmpeg usa NVENC si `USE_GPU` es True y hay soporte; de lo contrario cae a libx264.

## Para Ejecutar Un Test Único (ejemplo)
- Con pytest instalado: `pytest vhs_effect/test_vhs.py -k position_jitter`.
- Sin pytest: duplica/ajusta la función en `test_vhs.py` y ejecútala con `python vhs_effect/test_vhs.py` (llama todas las pruebas secuencialmente).
- Limpia `test_output.jpg` si no lo necesitas; no se versiona.

## Última Nota
Todo lo que sea cara al usuario debe permanecer en español. Cambios en rutas o coordenadas deben documentarse en el PR. Mantén los assets donde están y evita romper los scripts numerados.
