# AGENTS.md
Handbook para agentes (humanos o LLMs) que trabajen en este repositorio.

## Quick Start
```bash
pip install -r requirements.txt                    # Dependencias base
pip install -r vhs_effect/requirements.txt         # VHS (OpenCV/SciPy)
playwright install chromium                        # Navegadores Playwright
python config.py                                   # Validar entorno y crear carpetas
```

## Comandos de Test

### Test único (recomendado)
```bash
pytest vhs_effect/test_vhs.py -k color_processor    # Test específico por nombre
pytest vhs_effect/test_vhs.py -k position_jitter    # Otro test específico
pytest vhs_effect/test_vhs.py                       # Todos los tests VHS
```

### Sin pytest
```bash
python vhs_effect/test_vhs.py                       # Ejecuta todas las pruebas
```

### Smoke tests del pipeline
```bash
python "2. ffmpeg_render.py" --test                 # 1 carpeta de prueba
python "3. subir_video_API.py" --limite 1          # Subida mínima por API
```

### Build C++/CUDA
```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
# Binario: cpp/build/vhs_render
```

## Estructura del Repositorio
- **Scripts numerados**: `1. verificacion_previa.py`, `2. ffmpeg_render.py`, etc.
- `config.py`: rutas, flags GPU/NVENC, opciones de render
- `limpieza/`: normalización y limpieza de carpetas
- `effects/`: utilidades de imagen (`sombra`)
- `subir_video/`: autenticación y YouTube Data API
- `vhs_effect/`: librería VHS autocontenida + tests
- `cpp/`: renderer VHS C++/CUDA
- `content/`: intro y overlays
- `selectores/`: JSON de acciones Playwright

## Estilo de Código

### Formato
- Indentación: 4 espacios
- Nombres: `snake_case` funciones/variables, `PascalCase` clases, `MAYUSCULAS` constantes
- Líneas: máximo ~100 caracteres (flexible)

### Imports (orden obligatorio)
```python
import os                                # 1. stdlib
import subprocess
from pathlib import Path

import numpy as np                       # 2. terceros
from PIL import Image

from config import BASE_DIR, USE_GPU     # 3. locales
from effects.sombra import add_shadow
```

### Type hints (opcional)
```python
def procesar_carpeta(ruta: Path, limite: int = 10) -> list[str]:
    ...
```

### Rutas - SIEMPRE pathlib
```python
from pathlib import Path
carpeta = Path("/ruta/al/archivo")
carpeta.mkdir(parents=True, exist_ok=True)
# NO: os.path.join(), strings con "/"
```

### Subprocesos
```python
subprocess.run(["ffmpeg", "-i", str(input_path), str(output_path)], check=True)
# Evita shell=True
```

### Archivos
```python
with open(archivo, "r", encoding="utf-8") as f:
    ...
with open(csv_file, "w", encoding="utf-8", newline="") as f:
    ...
```

## Manejo de Errores
- Captura errores específicos, NO `except Exception:` vacío
- Loguea errores en español con contexto
- Respeta reintentos existentes: `DELAY_BASE_429`, `MAX_RETRIES_429`
- Fallbacks CUDA: `ALLOW_FFMPEG_FALLBACK`, `DISABLE_CPP_ON_CUDA`
- Verifica existencia de rutas antes de operar

## UX y Mensajes
- Logs/prompts en **español**
- Emojis consistentes: ✓ ✔️ ❌ (ya en el proyecto)
- Tono imperativo y breve

## Credenciales y Secretos
**NUNCA subas a git:**
- `client_secrets.json`, `token.json`, `.env`
- `.playwright_profile`, `PASS.py`

Variables de entorno en `.env`:
- `DEATHGRIND_EMAIL`, `DEATHGRIND_PASSWORD`

## VHS Effect
- Ubicación: `vhs_effect/` (librería independiente)
- OpenCV usa **BGR**; convierte explícitamente si usas PIL (RGB)
- Output temporal: `test_output.jpg` (no versionado)

## C++/CUDA Renderer
- Activado con `USE_CPP_VHS=True` en `config.py`
- Si falla CUDA, respeta `ALLOW_FFMPEG_FALLBACK`

## Playwright/Automatización
- Perfil persistente en `.playwright_profile`
- Selectores en `selectores/*.json`
- Ajusta sleeps con cautela; tiempos cortos rompen automatización

## Commits
- Mensajes cortos en **español**, descriptivos
- Ejemplos: `Barra de progreso`, `Arreglo render NVENC`
- Sin prefijos Conventional Commits
- No empujes credenciales ni binarios pesados

## Checklist Pre-Commit
- [ ] `pip install -r requirements.txt` sin errores
- [ ] `python config.py` valida entorno
- [ ] Smoke test: `pytest vhs_effect/test_vhs.py` o `python "2. ffmpeg_render.py" --test`
- [ ] Sin secretos en `git status`

## Fricción Conocida
- Coordenadas GUI son frágiles; prueba en resolución objetivo
- Render 4K pesado; `MAX_PARALLEL_RENDERS=4` techo con 3090 Ti
- Fallback FFmpeg usa NVENC si `USE_GPU=True`, sino libx264

## Notas para Agentes
- Instrucción ambigua que afecta producción → pregunta con duda concreta, propón default
- Tareas simples → actúa sin pedir permiso, resume lo que hiciste
- Todo cara al usuario en **español**
- No hay `.cursor/rules` ni `.github/copilot-instructions.md` en este repo
