#!/usr/bin/env bash
# Wrapper para impugnar copyright en YouTube Studio en modo "armar + auto-enviar + loop infinito".
#
# Comportamiento:
#   - Detecta TODOS los videos del canal con reclamos de copyright.
#   - Arma hasta 10 tabs por vuelta, cada una con el formulario completado.
#   - Después de armar el batch, AUTO-ENVÍA cada tab via Playwright trusted clicks
#     (marca los 3 checks + clickea "Enviar"). El usuario NO toca nada.
#   - Si una tab falla, queda abierta para envío manual; las exitosas se cierran.
#   - OMITE videos con fechas marcador 2028-04-30 y 2028-05-01 (CSV soportado).
#     Cambialas con --omitir-fecha "YYYY-MM-DD,YYYY-MM-DD".
#   - Loop hasta 1000 vueltas (efectivamente infinito hasta que no queden reclamos).
#   - Monitor de memoria: pausa automáticamente si la RAM libre baja de 2GB.
#
# Uso:
#   ./impugnar.sh                    # arma + auto-envía TODO en bucle
#   ./impugnar.sh --max-tabs 5       # batches más chicos (menos RAM)
#   ./impugnar.sh --probar-envio 1   # solo enviar 1 tab (debug)
#   ./impugnar.sh --max 1            # solo procesar 1 video
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
    --auto-enviar-listas \
    --max-tabs 10 \
    --loop \
    --max-loops 1000 \
    --omitir-fecha "2028-04-30,2028-05-01" \
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
