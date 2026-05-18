#!/usr/bin/env python3
"""auto_enviar.py — auto-clicker con ydotool (Wayland) para enviar pestañas armadas.

Uso:
    ./env/bin/python auto_enviar.py record   # input manual de las 4 coords
    ./env/bin/python auto_enviar.py run      # ejecuta el ciclo

Cómo medir coords (GNOME Wayland no permite leer cursor desde CLI):
  1) En la pestaña con el form, abrí DevTools (F12) → Console.
  2) Tipeá: monitorEvents(window, 'click')
  3) Hacé click sobre el check 1. La consola imprime un evento con clientX, clientY.
  4) Esos son los coords RELATIVOS a la viewport. Necesitás SUMAR el offset de la ventana
     del navegador (top-left de la viewport en el escritorio).
  5) Para el offset: en una terminal, ydotool no lee posición. Usá GNOME Screenshot área
     y mirá donde empieza la viewport del browser, o asumí offsets típicos
     (ej. fullscreen sin barra: 0,90 si el browser tiene tab bar de 90px).
  Más simple: corré 'run' y observá donde cae el primer click; ajustá si está corrido.

Comportamiento de 'run':
  - Por cada pestaña: click en las 4 coords (los 3 checks + Enviar).
  - Después: Ctrl+W cierra la pestaña actual (Brave avanza a la siguiente automáticamente).
  - Cada 50 pestañas pausa, espera ENTER.
  - Ctrl-C en la terminal aborta.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "auto_enviar_coords.json"
LOTE_PESTANAS = 50
DELAY_ENTRE_CLICKS = 0.35
DELAY_ANTES_DE_CERRAR = 1.2
DELAY_TRAS_CERRAR = 0.8


def _check_ydotool():
    if shutil.which("ydotool") is None:
        print("ERROR: ydotool no instalado. sudo pacman -S ydotool && sudo systemctl --user enable --now ydotoold")
        sys.exit(1)


def _ydotool_click(x, y):
    subprocess.run(["ydotool", "mousemove", "-a", "-x", str(x), "-y", str(y)], check=True)
    time.sleep(0.05)
    subprocess.run(["ydotool", "click", "0xC0"], check=True)


def _ydotool_ctrl_w():
    # KEY_LEFTCTRL=29, KEY_W=17. Formato ydotool: <code>:1 (down) / <code>:0 (up)
    subprocess.run(["ydotool", "key", "29:1", "17:1", "17:0", "29:0"], check=True)


def record():
    print("Modo record: vas a tipear las coordenadas X Y de los 4 puntos.\n")
    print("Tip rápido para medir:")
    print("  - Abrí DevTools (F12) en la pestaña, Console: monitorEvents(window, 'click')")
    print("  - Hacé click sobre cada elemento: te imprime clientX/clientY (viewport).")
    print("  - Sumá el offset Y del top de la viewport (alto del header de tabs ~90px).")
    print("  - O bien: probá run con coords aproximadas y ajustá viendo dónde caen.\n")
    coords = []
    etiquetas = ["check 1 (Mi vídeo no infringe)",
                 "check 2 (Entiendo que el reclamante)",
                 "check 3 (Soy consciente)",
                 "botón Enviar"]
    for i, etiqueta in enumerate(etiquetas, 1):
        line = input(f"  Posición {i}/4 [{etiqueta}] (formato: X Y): ").strip()
        try:
            x, y = line.split()
            coords.append([int(x), int(y)])
        except ValueError:
            print("Formato inválido. Esperaba dos números separados por espacio.")
            sys.exit(1)
        print(f"     guardado ({x}, {y})")
    CONFIG_PATH.write_text(json.dumps({"clicks": coords}, indent=2), encoding="utf-8")
    print(f"\nOK, guardado en {CONFIG_PATH}")


def run():
    _check_ydotool()
    if not CONFIG_PATH.exists():
        print(f"No existe {CONFIG_PATH}. Corré primero: python auto_enviar.py record")
        sys.exit(1)
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    clicks = data["clicks"]
    if len(clicks) != 4:
        print(f"El JSON tiene {len(clicks)} posiciones, esperaba 4. Re-grabá.")
        sys.exit(1)
    print(f"Cargadas {len(clicks)} posiciones. Lotes de {LOTE_PESTANAS}.")
    print("Tenés 5 segundos para enfocar la PRIMERA pestaña que vas a enviar...")
    for s in range(5, 0, -1):
        print(f"  ...{s}", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20)

    procesadas = 0
    try:
        while True:
            for x, y in clicks:
                _ydotool_click(x, y)
                time.sleep(DELAY_ENTRE_CLICKS)
            procesadas += 1
            print(f"[{procesadas}] pestaña enviada")
            time.sleep(DELAY_ANTES_DE_CERRAR)
            _ydotool_ctrl_w()
            time.sleep(DELAY_TRAS_CERRAR)
            if procesadas % LOTE_PESTANAS == 0:
                input(f"\n>>> {procesadas} pestañas enviadas. ENTER para seguir... ")
    except KeyboardInterrupt:
        print(f"\nABORT (Ctrl-C). Pestañas enviadas: {procesadas}")
    except subprocess.CalledProcessError as exc:
        print(f"\nydotool falló: {exc}. Pestañas enviadas: {procesadas}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "record":
        record()
    elif cmd == "run":
        run()
    else:
        print(__doc__)
        sys.exit(1)
