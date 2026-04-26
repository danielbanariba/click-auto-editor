#!/usr/bin/env bash
# Wrapper para impugnar copyright en YouTube Studio en modo "preparar pestañas + loop".
#
# Comportamiento:
#   - Detecta TODOS los videos del canal con reclamos de copyright.
#   - Por cada reclamo abre una pestaña, completa el formulario hasta firma,
#     y deja el botón "Enviar" SIN clickear (vos marcás los 3 checks + Enviar).
#   - Filtra reclamos ya enviados ("en proceso de revisión") — no los re-toca.
#   - Loop de hasta 30 vueltas, con 30 min entre vueltas para que envíes las tabs.
#
# Uso:
#   ./impugnar.sh                    # corre con defaults
#   ./impugnar.sh --max 5            # solo los primeros 5 videos
#   ./impugnar.sh --headless         # sin ventana (NO recomendado, no podés enviar)
#
# Cualquier flag extra que pases se inyecta al comando.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="./env/bin/python"
SCRIPT="10. inpunar_video.py"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no encuentro $PYTHON. Activá el venv o reinstalá."
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: no encuentro '$SCRIPT' en $(pwd)"
    exit 1
fi

mkdir -p data

exec "$PYTHON" -u "$SCRIPT" \
    --auto-detect \
    --no-esperar \
    --preparar-pestanas \
    --loop \
    --max-loops 30 \
    --pausa-interactiva-s 1800 \
    --delay 1.2 \
    --espera-envio 6 \
    --espera-entre 1 \
    --reintentos-modal 2 \
    --cola data/impugnar_claims_queue.json \
    --checkpoint data/impugnar_claims_checkpoint.txt \
    --evidencia-dir data/impugnar_evidencia \
    "$@"
