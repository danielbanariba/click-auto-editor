#!/usr/bin/env bash
# Wrapper para apelar reclamos de copyright en YouTube Studio.
#
# Comportamiento:
#   - Detecta videos del canal cuya fila tenga fecha+hora EXACTAS: 2028-04-30 23:45
#     (la combinacion que usaste como marcador en los videos ya impugnados pendientes
#     de apelar). En impugnar.sh esa misma fecha se OMITE; aca es lo opuesto.
#   - NO usa filtro de reclamante por default (--filtrar-reclamante vacio = procesa todo).
#     Si querés filtrar por reclamante, agregalo con --filtrar-reclamante "CD Baby".
#   - Ejecuta el flujo de apelacion (NO impugnacion): click "Apelar", llenar
#     formulario completo con datos personales + mensaje + firma, enviar.
#
# IMPORTANTE: este wrapper usa el flow lineal viejo de "11. apelacion.py" (NO el sistema
# de pestañas paralelas + auto-envio trusted que tiene impugnar.sh). Si fallan los clicks
# por bot detection de YouTube, hay que portar los patterns modernos de inpunar_video.py.
#
# Uso:
#   ./apelar.sh                                    # filtro default: fecha 2028-04-30
#   ./apelar.sh --solo-fecha ""                    # sin filtro fecha (todos los videos)
#   ./apelar.sh --solo-fecha 2028-05-15            # cambiar fecha marcador
#   ./apelar.sh --solo-hora 23:45                  # agregar filtro por hora (opcional, suele no estar en el scrape)
#   ./apelar.sh --filtrar-reclamante "CD Baby"     # agregar filtro por reclamante
#   ./apelar.sh --max 1                            # solo 1 video
#   ./apelar.sh --solo-detectar                    # escanea y guarda cola sin enviar
#
# Cualquier flag extra que pases se inyecta al comando.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="./env/bin/python"
SCRIPT="11. apelacion.py"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no encuentro $PYTHON. Activá el venv o reinstalá."
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: no encuentro '$SCRIPT' en $(pwd)"
    exit 1
fi

mkdir -p data

FECHA_DEFAULT="2028-04-30"

USER_PASSED_FECHA=0
for arg in "$@"; do
    if [[ "$arg" == "--solo-fecha"* ]]; then
        USER_PASSED_FECHA=1
    fi
done

EXTRA_ARGS=()
if [[ "$USER_PASSED_FECHA" -eq 0 ]]; then
    EXTRA_ARGS+=(--solo-fecha "$FECHA_DEFAULT")
fi

exec "$PYTHON" -u "$SCRIPT" \
    --auto-detect \
    --no-esperar \
    --loop \
    --max-loops 1000 \
    --espera-vuelta 3 \
    --delay 1.2 \
    --espera-envio 6 \
    --espera-entre 1 \
    --reintentos-modal 2 \
    --cola data/apelar_claims_queue.json \
    --checkpoint data/apelar_claims_checkpoint.txt \
    "${EXTRA_ARGS[@]}" \
    "$@"
