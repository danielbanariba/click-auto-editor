#!/usr/bin/env python3
"""Muestra los videos programados en YouTube agrupados por dia.

Detecta huecos (horas sin video) y duplicados (mas de un video en la misma
hora), asumiendo el patron normal de 1 video por hora (24 por dia).

Reutiliza authenticate de subir_video.authenticate y parse_rfc3339 de
"3. subir_video_API.py" para mantener la misma logica de zonas horarias y
parseo que el script de subida.

Uso:
    python ver_programados.py
    python ver_programados.py --scan 3000
    python ver_programados.py --solo-anomalias
"""
import argparse
import importlib.util
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from subir_video.authenticate import authenticate


SCRIPT_DIR = Path(__file__).resolve().parent
SUBIR_PATH = SCRIPT_DIR / "3. subir_video_API.py"


def cargar_modulo_subida():
    spec = importlib.util.spec_from_file_location("subir_video_api", SUBIR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {SUBIR_PATH}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["subir_video_api"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def fetch_scheduled_por_dia(youtube, subir, max_items=2000):
    """Devuelve {date: [(datetime_local, video_id, titulo), ...]} ordenado."""
    playlist_id = subir.get_uploads_playlist_id(youtube)
    if not playlist_id:
        return {}

    tz_local = datetime.now().astimezone().tzinfo
    por_dia = defaultdict(list)
    fetched = 0
    page_token = None

    while True:
        resp = (
            youtube.playlistItems()
            .list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            )
            .execute()
        )
        items = resp.get("items", [])
        if not items:
            break

        ids_y_titulos = {}
        for item in items:
            snippet = item.get("snippet", {})
            video_id = snippet.get("resourceId", {}).get("videoId")
            if video_id:
                ids_y_titulos[video_id] = snippet.get("title", "")

        video_ids = list(ids_y_titulos.keys())
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            vids = (
                youtube.videos()
                .list(part="status", id=",".join(chunk))
                .execute()
            )
            for video in vids.get("items", []):
                vid = video.get("id")
                publish_at = video.get("status", {}).get("publishAt")
                if not publish_at:
                    continue
                dt = subir.parse_rfc3339(publish_at)
                if not dt:
                    continue
                local_dt = dt.astimezone(tz_local)
                if local_dt < datetime.now(tz_local):
                    continue
                slot = local_dt.replace(second=0, microsecond=0)
                por_dia[slot.date()].append(
                    (slot, vid, ids_y_titulos.get(vid, ""))
                )

        fetched += len(items)
        page_token = resp.get("nextPageToken")
        if max_items is not None and fetched >= max_items:
            if page_token:
                print(
                    "Aviso: se alcanzo el limite de escaneo. Podrian faltar fechas. "
                    "Subi --scan si necesitas mas."
                )
            break
        if not page_token:
            break

    for fecha in por_dia:
        por_dia[fecha].sort(key=lambda x: x[0])
    return dict(por_dia)


def analizar_dia(slots):
    """Devuelve (horas_presentes, huecos, duplicados).

    - horas_presentes: set de horas 0-23 que tienen al menos 1 video
    - huecos: lista ordenada de horas sin video (0-23)
    - duplicados: dict {hora: [datetime, datetime, ...]} con horas repetidas
    """
    por_hora = defaultdict(list)
    for slot, _vid, _titulo in slots:
        por_hora[slot.hour].append(slot)

    horas_presentes = set(por_hora.keys())
    huecos = sorted(set(range(24)) - horas_presentes)
    duplicados = {h: ts for h, ts in por_hora.items() if len(ts) > 1}
    return horas_presentes, huecos, duplicados


def formatear_huecos(huecos):
    """Compacta rangos contiguos de huecos: [3,4,5,8] -> '03-05, 08'."""
    if not huecos:
        return ""
    rangos = []
    inicio = anterior = huecos[0]
    for h in huecos[1:]:
        if h == anterior + 1:
            anterior = h
            continue
        rangos.append((inicio, anterior))
        inicio = anterior = h
    rangos.append((inicio, anterior))
    partes = []
    for ini, fin in rangos:
        if ini == fin:
            partes.append(f"{ini:02d}")
        else:
            partes.append(f"{ini:02d}-{fin:02d}")
    return ", ".join(partes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scan",
        type=int,
        default=2000,
        help="Maximo de videos a escanear de la playlist 'uploads' (default: 2000).",
    )
    parser.add_argument(
        "--prefix",
        default="upload",
        help="Prefijo de credenciales (default: upload).",
    )
    parser.add_argument(
        "--solo-anomalias",
        action="store_true",
        help="Mostrar solo dias con huecos o duplicados.",
    )
    parser.add_argument(
        "--esperado",
        type=int,
        default=24,
        help="Videos esperados por dia (default: 24, uno por hora).",
    )
    args = parser.parse_args()

    subir = cargar_modulo_subida()

    print(f"Autenticando con prefijo '{args.prefix}'...")
    youtube = authenticate(args.prefix)

    print(f"Escaneando hasta {args.scan} videos...")
    por_dia = fetch_scheduled_por_dia(youtube, subir, max_items=args.scan)

    if not por_dia:
        print("\nNo hay videos programados en el rango escaneado.")
        return

    dias_ordenados = sorted(por_dia.items())
    total_videos = sum(len(s) for _, s in dias_ordenados)
    dias_con_problema = 0

    print(
        f"\nDias programados: {len(dias_ordenados)} | "
        f"Videos totales: {total_videos} | "
        f"Esperado por dia: {args.esperado}"
    )
    print("-" * 70)

    for fecha, slots in dias_ordenados:
        cantidad = len(slots)
        _horas, huecos, duplicados = analizar_dia(slots)

        tiene_problema = bool(huecos) or bool(duplicados) or cantidad != args.esperado
        if tiene_problema:
            dias_con_problema += 1
        if args.solo_anomalias and not tiene_problema:
            continue

        if tiene_problema:
            estado = "❌"
        else:
            estado = "✅"
        print(f"{estado} {fecha.isoformat()}: {cantidad} videos")

        if huecos:
            print(f"     🕳  Horas faltantes: {formatear_huecos(huecos)}")
        for hora in sorted(duplicados):
            momentos = ", ".join(t.strftime("%H:%M") for t in duplicados[hora])
            print(f"     ⚠️  Hora {hora:02d} duplicada ({len(duplicados[hora])}x): {momentos}")

    print("-" * 70)
    print(f"Resumen: {dias_con_problema} dia(s) con anomalias de {len(dias_ordenados)}.")

    ultima = max(slot for slots in por_dia.values() for slot, _, _ in slots)
    print(f"Ultima programacion detectada: {subir.format_local_datetime(ultima)}")


if __name__ == "__main__":
    main()
