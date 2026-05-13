#!/usr/bin/env bash
# Wrapper para ESCANEAR (read-only) todo el canal y reportar que videos
# tienen como reclamante de copyright a uno o varios nombres dados.
#
# Por default usa la "lista negra" de sellos en
#   /home/banar/Desktop/scrapper-deathgrind/lista_sello.txt
# (120 sellos discograficos problematicos).
#
# Si pasas un nombre como primer argumento, busca solo ese (modo legacy).
#
# Comportamiento:
#   - Escanea TODOS los videos del canal con restriccion "Derechos de autor".
#   - Sin filtro de fecha, sin limite de cantidad: canal completo.
#   - Para cada video, abre /copyright, clickea "Ver detalles" de cada claim,
#     lee la seccion "Reclamantes" del modal, cierra el modal.
#   - NO clickea "Tomar medidas", NO envia nada. Read-only desde el lado de YT.
#
# Uso:
#   ./escanear-reclamante.sh                                       # usa lista_sello.txt
#   ./escanear-reclamante.sh --archivo /ruta/otra_lista.txt        # archivo distinto
#   ./escanear-reclamante.sh "Hostile Media"                       # un solo reclamante
#   ./escanear-reclamante.sh "Hostile Media,CD Baby CO"            # CSV de nombres
#   ./escanear-reclamante.sh "Hostile Media" --max 5               # debug: 5 videos
#
# Output:
#   - data/escaneo_<slug>_queue.json     -> cola con status + reclamantes por video
#   - data/escaneo_<slug>_matches.txt    -> solo los que matchean el filtro
#                                           (titulo \t video_id \t reclamantes \t url)

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="./env/bin/python"
SCRIPT="11. apelacion.py"
LISTA_DEFAULT="/home/banar/Desktop/scrapper-deathgrind/lista_sello.txt"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no encuentro $PYTHON. Activa el venv o reinstala."
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: no encuentro '$SCRIPT' en $(pwd)"
    exit 1
fi

mkdir -p data

# --- Resolver modo: archivo vs CSV ---
ARCHIVO=""
RECLAMANTE_CSV=""
EXTRA_ARGS=()

# Si el usuario pasa --archivo PATH explicitamente, usar eso.
# Si pasa un primer arg que no empieza con "--", tratarlo como CSV de nombres.
# Si no pasa nada relevante, default = lista_sello.txt.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archivo)
            ARCHIVO="$2"
            shift 2
            ;;
        --archivo=*)
            ARCHIVO="${1#*=}"
            shift
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [[ -z "$RECLAMANTE_CSV" ]]; then
                RECLAMANTE_CSV="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$ARCHIVO" && -z "$RECLAMANTE_CSV" ]]; then
    ARCHIVO="$LISTA_DEFAULT"
fi

# --- Generar slug para nombres de archivo ---
if [[ -n "$ARCHIVO" ]]; then
    if [[ ! -f "$ARCHIVO" ]]; then
        echo "ERROR: archivo de reclamantes no existe: $ARCHIVO"
        exit 1
    fi
    SLUG_BASE=$(basename "$ARCHIVO" .txt)
    SLUG=$(echo "$SLUG_BASE" | tr '[:upper:]' '[:lower:]' | tr ' ,' '__' | tr -cd 'a-z0-9_-')
else
    SLUG=$(echo "$RECLAMANTE_CSV" | tr '[:upper:]' '[:lower:]' | tr ' ,' '__' | tr -cd 'a-z0-9_-')
fi

QUEUE="data/escaneo_${SLUG}_queue.json"
MATCHES="data/escaneo_${SLUG}_matches.txt"
CHECKPOINT="data/escaneo_${SLUG}_checkpoint.txt"

echo "==========================================="
if [[ -n "$ARCHIVO" ]]; then
    N_NOMBRES=$(grep -cvE '^\s*$|^\s*#' "$ARCHIVO" || true)
    echo "ESCANEO con archivo: $ARCHIVO"
    echo "Nombres en lista:   $N_NOMBRES"
fi
if [[ -n "$RECLAMANTE_CSV" ]]; then
    echo "ESCANEO con CSV:    $RECLAMANTE_CSV"
fi
echo "Cola:    $QUEUE"
echo "Matches: $MATCHES"
echo "==========================================="
echo ""

CMD=("$PYTHON" -u "$SCRIPT"
    --auto-detect
    --no-esperar
    --solo-detectar
    --matches-output "$MATCHES"
    --delay 1.2
    --espera-entre 0.5
    --cola "$QUEUE"
    --checkpoint "$CHECKPOINT")

if [[ -n "$ARCHIVO" ]]; then
    CMD+=(--filtrar-reclamante-archivo "$ARCHIVO")
fi
if [[ -n "$RECLAMANTE_CSV" ]]; then
    CMD+=(--filtrar-reclamante "$RECLAMANTE_CSV")
fi

exec "${CMD[@]}" "${EXTRA_ARGS[@]}"
