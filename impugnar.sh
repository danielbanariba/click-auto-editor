#!/usr/bin/env bash
# Wrapper para impugnar copyright en YouTube Studio en modo "preparar pestañas + loop".
#
# Comportamiento:
#   - Detecta TODOS los videos del canal con reclamos de copyright.
#   - Por cada reclamo abre una pestaña, completa el formulario hasta firma,
#     y deja el botón "Enviar" SIN clickear (vos marcás los 3 checks + Enviar).
#   - OMITE videos con fecha marcador 2028-04-30 (los que ya impugnaste y querés
#     saltear). Cambia la fecha pasando --omitir-fecha YYYY-MM-DD.
#   - Loop manual: al terminar cada vuelta, ENTER arranca la siguiente.
#   - Monitor de memoria: si la RAM libre baja de 2GB, PAUSA automaticamente
#     hasta tener al menos 3.5GB libres (cerrá tabs enviadas para liberar).
#
# Uso:
#   ./impugnar.sh                    # corre con defaults
#   ./impugnar.sh --max 5            # solo los primeros 5 videos
#   ./impugnar.sh --omitir-fecha 2028-05-15   # cambiar fecha marcador
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
    --omitir-fecha 2028-04-30 \
    --mem-pausa-mb 2000 \
    --mem-reanudar-mb 3500 \
    --delay 1.2 \
    --espera-envio 6 \
    --espera-entre 1 \
    --reintentos-modal 2 \
    --cola data/impugnar_claims_queue.json \
    --checkpoint data/impugnar_claims_checkpoint.txt \
    --evidencia-dir data/impugnar_evidencia \
    "$@"
