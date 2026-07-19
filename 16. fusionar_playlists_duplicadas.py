#!/usr/bin/env python3
"""Fusiona playlists de pais duplicadas en una sola, con nombre canonico en ingles.

Agrupa por codigo ISO (via paises.canonicalizar_pais), elige una playlist
destino, mueve los videos de las demas hacia ella y borra las extras.

Por defecto corre en --dry-run (solo muestra el plan y estima la cuota).
Usar --aplicar para ejecutar de verdad. Toma el mismo lock que mapear_playlists
para no correr en simultaneo con el servicio.

Costo de cuota YouTube Data API: list=1, playlistItems.insert=50,
playlists.update=50, playlists.delete=50 (limite diario tipico: 10000).
"""
import argparse
import importlib.util
from pathlib import Path

from paises import canonicalizar_pais, nombre_pais_en

# El script principal tiene espacio y numero en el nombre: se carga por ruta.
_MAPEAR_PATH = Path(__file__).resolve().parent / "12. mapear_playlists.py"
_spec = importlib.util.spec_from_file_location("mapear_playlists", _MAPEAR_PATH)
mapear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mapear)

COSTO_INSERT = 50
COSTO_DELETE = 50
COSTO_UPDATE = 50


def agrupar_paises_duplicados(playlists):
    """Agrupa playlists por codigo de pais; devuelve solo los grupos con >1."""
    grupos = {}
    for pl in playlists:
        code = canonicalizar_pais(pl.get("title"))
        if not code:
            continue
        grupos.setdefault(code, []).append(pl)
    return {code: pls for code, pls in grupos.items() if len(pls) > 1}


def elegir_destino(code, playlists):
    """Devuelve (destino, nombre_canonico, extras, necesita_rename).

    Prioriza una playlist que ya tenga el nombre canonico en ingles; si no,
    elige la que tenga mas videos y habra que renombrarla.
    """
    nombre = nombre_pais_en(code)
    exactos = [p for p in playlists if (p.get("title") or "") == nombre]
    if exactos:
        destino = max(exactos, key=lambda p: p.get("count", 0) or 0)
        necesita_rename = False
    else:
        destino = max(playlists, key=lambda p: p.get("count", 0) or 0)
        necesita_rename = True
    extras = [p for p in playlists if p.get("id") != destino.get("id")]
    return destino, nombre, extras, necesita_rename


def estimar_cuota(grupos):
    total = 0
    for code, pls in grupos.items():
        _, _, extras, necesita_rename = elegir_destino(code, pls)
        # peor caso: cada extra mueve sus 'count' videos + se borra
        for ex in extras:
            total += (ex.get("count", 0) or 0) * COSTO_INSERT + COSTO_DELETE
        if necesita_rename:
            total += COSTO_UPDATE
    return total


def _mover_videos(youtube, origen_id, destino_id, ya_presentes):
    movidos = 0
    video_ids, _ = mapear.get_playlist_video_ids(youtube, origen_id)
    for vid in video_ids:
        if vid in ya_presentes:
            continue
        body = {
            "snippet": {
                "playlistId": destino_id,
                "resourceId": {"kind": "youtube#video", "videoId": vid},
            }
        }
        mapear.run_with_backoff(
            lambda: youtube.playlistItems()
            .insert(part="snippet", body=body)
            .execute(),
            descripcion=f"mover video {vid}",
        )
        ya_presentes.add(vid)
        movidos += 1
    return movidos


def fusionar(youtube, grupos, aplicar=False):
    for code, pls in sorted(grupos.items()):
        destino, nombre, extras, necesita_rename = elegir_destino(code, pls)
        titulos_extra = ", ".join(f"'{e['title']}'" for e in extras)
        print(f"\n[{code}] -> '{nombre}' (destino: {destino['id']})")
        print(f"  fusiona: {titulos_extra}")
        if necesita_rename:
            print(f"  + renombra destino '{destino['title']}' -> '{nombre}'")
        if not aplicar:
            continue

        if necesita_rename:
            mapear.run_with_backoff(
                lambda: youtube.playlists()
                .update(
                    part="snippet",
                    body={"id": destino["id"], "snippet": {"title": nombre}},
                )
                .execute(),
                descripcion=f"renombrar playlist a '{nombre}'",
            )

        ya_presentes, _ = mapear.get_playlist_video_ids(youtube, destino["id"])
        ya_presentes = set(ya_presentes)
        for ex in extras:
            movidos = _mover_videos(youtube, ex["id"], destino["id"], ya_presentes)
            print(f"    movidos {movidos} de '{ex['title']}'")
            mapear.run_with_backoff(
                lambda: youtube.playlists().delete(id=ex["id"]).execute(),
                descripcion=f"borrar playlist '{ex['title']}'",
            )
            print(f"    borrada '{ex['title']}'")


def main():
    parser = argparse.ArgumentParser(
        description="Fusiona playlists de pais duplicadas (nombre canonico en ingles)."
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Ejecuta de verdad. Sin esto corre en dry-run (solo muestra el plan).",
    )
    args = parser.parse_args()

    if not mapear.adquirir_lock():
        print(
            "Hay otra instancia (mapear/fusionar) corriendo "
            f"(lock: {mapear.DEFAULT_LOCK}). Se aborta."
        )
        return

    youtube = mapear.authenticate(prefix="playlists")
    # Bucle de rotacion de credenciales: si una agota cuota, se pasa a la
    # siguiente. Re-lista y re-agrupa en cada vuelta, asi que es resumible:
    # las ya fusionadas quedan como playlist unica y se saltan solas.
    while True:
        try:
            _, _, todas = mapear.list_playlists(youtube, include_counts=True)
            grupos = agrupar_paises_duplicados(todas)

            if not grupos:
                print("No hay playlists de pais duplicadas. Nada que fusionar.")
                return

            print(f"Paises con playlists duplicadas: {len(grupos)}")
            cuota = estimar_cuota(grupos)
            print(
                f"Cuota estimada (peor caso): ~{cuota} unidades (limite diario ~10000)."
            )
            if not args.aplicar:
                print(
                    "\n=== DRY-RUN (no se modifica nada). Use --aplicar para ejecutar. ==="
                )

            fusionar(youtube, grupos, aplicar=args.aplicar)
            break
        except mapear.QuotaExceededError:
            print("\nCuota agotada. Intentando con siguiente credencial...")
            youtube = mapear.authenticate_next(prefix="playlists")
            if youtube is None:
                print(
                    "Todas las credenciales agotadas. Reintenta manana con --aplicar; "
                    "el script retoma solo (re-lista y salta lo ya fusionado)."
                )
                return

    if args.aplicar:
        print("\nFusion completada.")
    else:
        print("\nDry-run terminado. Nada se modifico.")


if __name__ == "__main__":
    main()
