#!/usr/bin/env python3
"""Ordena las playlists de la seccion 'Generos Musicales/Musical Genres'
por cantidad de videos en orden descendente (mas videos primero).
"""
import argparse
import importlib.util
import time
from pathlib import Path

from googleapiclient.errors import HttpError

from subir_video.authenticate import (
    authenticate,
    authenticate_next,
)

# ---------------------------------------------------------------------------
# Importar utilidades de "12. mapear_playlists.py"
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "mapear_playlists",
    Path(__file__).resolve().parent / "12. mapear_playlists.py",
)
_mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mp)

run_with_backoff = _mp.run_with_backoff
RateLimitError = _mp.RateLimitError
QuotaExceededError = _mp.QuotaExceededError
CredentialNotConfiguredError = _mp.CredentialNotConfiguredError
get_error_reasons = _mp.get_error_reasons
clean_title = _mp.clean_title
normalize_title = _mp.normalize_title

SECCION_GENEROS = "géneros musicales/musical genres"


# ---------------------------------------------------------------------------
# Funciones de la API
# ---------------------------------------------------------------------------


def list_channel_sections(youtube):
    """Lista todas las secciones del canal autenticado."""
    resp = run_with_backoff(
        lambda: youtube.channelSections()
        .list(part="id,snippet,contentDetails", mine=True)
        .execute(),
        descripcion="listar secciones del canal",
    )
    return resp.get("items", [])


def get_playlist_info(youtube, playlist_ids):
    """Obtiene titulo y cantidad de videos de cada playlist (bloques de 50)."""
    info = {}
    for i in range(0, len(playlist_ids), 50):
        chunk = playlist_ids[i : i + 50]
        ids_str = ",".join(chunk)
        resp = run_with_backoff(
            lambda ids=ids_str: youtube.playlists()
            .list(
                part="snippet,contentDetails",
                id=ids,
                maxResults=50,
                fields="items(id,snippet/title,contentDetails/itemCount)",
            )
            .execute(),
            descripcion="obtener info de playlists",
        )
        for item in resp.get("items", []):
            pid = item.get("id")
            title = clean_title(
                item.get("snippet", {}).get("title", "")
            ) or ""
            count = item.get("contentDetails", {}).get("itemCount", 0) or 0
            if pid:
                info[pid] = {"title": title, "count": int(count)}
    return info


def update_section_playlists(youtube, section):
    """Actualiza una seccion del canal con las playlists reordenadas."""
    snippet = section.get("snippet", {})
    content_details = {}
    if "playlists" in section.get("contentDetails", {}):
        content_details["playlists"] = section["contentDetails"]["playlists"]
    if "channels" in section.get("contentDetails", {}):
        content_details["channels"] = section["contentDetails"]["channels"]

    body = {
        "id": section["id"],
        "snippet": {
            "type": snippet["type"],
            "title": snippet.get("title", ""),
        },
        "contentDetails": content_details,
    }
    position = snippet.get("position")
    if position is not None:
        body["snippet"]["position"] = position

    run_with_backoff(
        lambda: youtube.channelSections()
        .update(part="snippet,contentDetails", body=body)
        .execute(),
        descripcion=f"actualizar seccion '{snippet.get('title', '')}'",
    )


def delete_section(youtube, section_id):
    """Elimina una seccion del canal."""
    run_with_backoff(
        lambda: youtube.channelSections()
        .delete(id=section_id)
        .execute(),
        descripcion=f"eliminar seccion {section_id}",
    )


def insert_section(youtube, channel_id, title, playlist_ids, position=None):
    """Crea una nueva seccion de tipo multiplePlaylists.

    Si la API rechaza por demasiadas playlists, reintenta con la mitad
    hasta encontrar el maximo aceptado. Retorna la cantidad insertada.
    """
    attempt_ids = list(playlist_ids)

    while attempt_ids:
        body = {
            "snippet": {
                "type": "multiplePlaylists",
                "title": title,
                "channelId": channel_id,
            },
            "contentDetails": {
                "playlists": attempt_ids,
            },
        }
        if position is not None:
            body["snippet"]["position"] = position

        try:
            run_with_backoff(
                lambda b=body: youtube.channelSections()
                .insert(part="snippet,contentDetails", body=b)
                .execute(),
                descripcion=f"crear seccion '{title}' ({len(attempt_ids)} playlists)",
            )
            return len(attempt_ids)
        except HttpError as e:
            error_str = str(e)
            is_too_large = (
                "maxPlaylistExceeded" in error_str
                or (e.resp.status == 400 and "invalid argument" in error_str.lower())
            )
            if not is_too_large:
                raise
            # Reducir a la mitad y reintentar
            new_size = len(attempt_ids) // 2
            if new_size == 0:
                raise
            print(f"  Insert con {len(attempt_ids)} playlists fallo. "
                  f"Reintentando con {new_size}...")
            attempt_ids = attempt_ids[:new_size]

    return 0


MAX_SECCIONES_CANAL = 10


def _is_update_too_large_error(error):
    """Determina si un HttpError indica que la seccion es demasiado grande para update."""
    error_str = str(error)
    if "maxPlaylistExceeded" in error_str:
        return True
    if error.resp.status == 400 and "invalid argument" in error_str.lower():
        return True
    return False


def is_empty_undefined_section(section):
    """Detecta secciones legacy/rotas vacias que solo ocupan slot."""
    snippet = section.get("snippet", {})
    content_details = section.get("contentDetails", {})
    if snippet.get("type") != "channelsectiontypeundefined":
        return False
    if content_details.get("playlists"):
        return False
    if content_details.get("channels"):
        return False
    return True


def free_slots_if_needed(youtube, sections, slots_needed, pausa):
    """Elimina secciones undefined vacias hasta liberar los slots necesarios.

    Retorna la cantidad de slots liberados.
    """
    total = len(sections)
    free = MAX_SECCIONES_CANAL - total
    if free >= slots_needed:
        return 0

    need_to_free = slots_needed - free
    undefined_vacias = [s for s in sections if is_empty_undefined_section(s)]

    if len(undefined_vacias) < need_to_free:
        print(f"  Se necesitan {need_to_free} slots pero solo hay "
              f"{len(undefined_vacias)} secciones vacias eliminables.")
        return 0

    freed = 0
    for section in undefined_vacias[:need_to_free]:
        delete_section(youtube, section["id"])
        freed += 1
        print(f"  Seccion undefined vacia eliminada: {section['id']}")
        if pausa:
            time.sleep(pausa)

    return freed


def main(youtube=None):
    parser = argparse.ArgumentParser(
        description=(
            "Ordena playlists de 'Generos Musicales/Musical Genres' "
            "por cantidad de videos (descendente)."
        ),
    )
    parser.add_argument(
        "--solo-mostrar",
        action="store_true",
        help="Solo muestra las playlists y sus conteos sin modificar nada.",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
        help="Segundos de espera tras la actualizacion (default: 1.0).",
    )
    parser.add_argument(
        "--seccion",
        type=str,
        default=None,
        help=(
            "Nombre exacto de la seccion a ordenar. "
            "Por defecto: 'Generos Musicales/Musical Genres'."
        ),
    )
    parser.add_argument(
        "--restaurar",
        type=str,
        default=None,
        metavar="ARCHIVO_IDS",
        help=(
            "Restaurar la seccion desde un archivo con IDs de playlists "
            "(uno por linea). Limpia secciones vacias si es necesario."
        ),
    )
    args = parser.parse_args()

    if youtube is None:
        youtube = authenticate(prefix="playlists")

    print("Obteniendo secciones del canal...")
    sections = list_channel_sections(youtube)
    if not sections:
        print("No se encontraron secciones en el canal.")
        if not args.restaurar:
            return

    # Diagnostico: mostrar secciones
    total_sections = len(sections)
    print(f"Secciones encontradas: {total_sections}")
    for s in sections:
        snippet = s.get("snippet", {})
        stype = snippet.get("type", "???")
        stitle = snippet.get("title", "(sin titulo)")
        pids = s.get("contentDetails", {}).get("playlists", [])
        print(f"  [{stype}] {stitle} ({len(pids)} playlists)")

    # Buscar la seccion objetivo
    section_title = args.seccion or "Géneros Musicales/Musical Genres"
    target_key = normalize_title(section_title)

    section = None
    for s in sections:
        title = s.get("snippet", {}).get("title", "")
        if normalize_title(title) == target_key:
            section = s
            break

    # --- Modo restaurar: recrear seccion que no existe ---
    if args.restaurar:
        if section is not None:
            print(f"La seccion '{section_title}' ya existe. No hace falta restaurar.")
            return

        playlist_ids_file = Path(args.restaurar)
        if not playlist_ids_file.exists():
            print(f"Archivo no encontrado: {args.restaurar}")
            return

        restore_ids = [
            line.strip() for line in
            playlist_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not restore_ids:
            print("El archivo no contiene IDs de playlists.")
            return

        print(f"Restaurando seccion '{section_title}' con {len(restore_ids)} playlists...")

        # Liberar slots si es necesario
        freed = free_slots_if_needed(youtube, sections, 1, args.pausa)
        if freed > 0:
            print(f"  {freed} slot(s) liberado(s).")

        # Obtener channel_id de cualquier seccion existente
        channel_id = ""
        for s in sections:
            cid = s.get("snippet", {}).get("channelId", "")
            if cid:
                channel_id = cid
                break

        if not channel_id:
            print("No se pudo determinar el channel_id. Verifica las credenciales.")
            return

        inserted = insert_section(youtube, channel_id, section_title, restore_ids)
        if inserted == len(restore_ids):
            print(f"Seccion '{section_title}' restaurada con {inserted} playlists.")
        elif inserted > 0:
            print(f"Seccion '{section_title}' restaurada con {inserted}/{len(restore_ids)} playlists.")
            remaining = restore_ids[inserted:]
            print(f"  La API no acepta mas de {inserted} playlists por seccion.")
            print(f"  Quedan {len(remaining)} playlists sin agregar.")
            print("  Agrega las restantes manualmente desde YouTube Studio.")
            remaining_file = Path("_restaurar_generos_restantes.txt")
            remaining_file.write_text(
                "\n".join(remaining) + "\n", encoding="utf-8"
            )
            print(f"  IDs guardados en: {remaining_file}")
        else:
            print("No se pudo crear la seccion. Intentalo desde YouTube Studio.")
        return

    if section is None:
        print(f"\nNo se encontro la seccion '{section_title}'.")
        print("Si fue eliminada, usa --restaurar ARCHIVO con los IDs de playlists.")
        return

    title = section.get("snippet", {}).get("title", "")
    playlists = section.get("contentDetails", {}).get("playlists", [])

    if not playlists:
        print(f"La seccion '{title}' no contiene playlists.")
        return

    print(f"Seccion encontrada: {title} ({len(playlists)} playlists)")
    print(f"Obteniendo info de {len(playlists)} playlists...")

    playlist_info = get_playlist_info(youtube, playlists)

    # Ordenar por cantidad de videos (descendente), desempate alfabetico
    sorted_playlists = sorted(
        playlists,
        key=lambda pid: (
            -(playlist_info.get(pid, {}).get("count", 0)),
            normalize_title(playlist_info.get(pid, {}).get("title", pid)),
        ),
    )

    # Mostrar el orden resultante
    print(f"\nSeccion: {title}")
    print("  Playlists ordenadas por cantidad de videos:")
    for i, pid in enumerate(sorted_playlists):
        info = playlist_info.get(pid, {})
        ptitle = info.get("title", pid)
        count = info.get("count", 0)
        original_pos = playlists.index(pid)
        marker = "" if original_pos == i else f" (era pos {original_pos + 1})"
        print(f"    {i + 1}. {ptitle} ({count} videos){marker}")

    unresolved = [pid for pid in playlists if pid not in playlist_info]
    if unresolved:
        print("  Advertencia: estas playlists no devolvieron info en la API:")
        for pid in unresolved:
            print(f"    - {pid}")

    already_sorted = sorted_playlists == playlists
    if already_sorted:
        print("\n  Ya esta en el orden correcto.")
        return

    if args.solo_mostrar:
        print("\n  (modo solo-mostrar, no se actualiza)")
        return

    # Intentar update directo
    section["contentDetails"]["playlists"] = sorted_playlists
    try:
        update_section_playlists(youtube, section)
        print(f"\n  Seccion '{title}' actualizada.")
    except HttpError as e:
        if not _is_update_too_large_error(e):
            raise

        # La API no soporta update/insert con tantas playlists.
        # NO eliminamos la seccion — la API no puede recrearla.
        print(f"\n  La API de YouTube no soporta actualizar secciones con "
              f"{len(sorted_playlists)} playlists.")
        print("  La seccion NO fue modificada (se conserva el orden actual).")
        print("  Para reordenar, hacelo manualmente desde YouTube Studio.")
        print("  Orden deseado (por cantidad de videos):")
        for i, pid in enumerate(sorted_playlists):
            info = playlist_info.get(pid, {})
            print(f"    {i + 1}. {info.get('title', pid)} ({info.get('count', 0)} videos)")
        return

    if args.pausa:
        time.sleep(args.pausa)

    print("\nListo.")


if __name__ == "__main__":
    youtube_client = None
    while True:
        try:
            main(youtube=youtube_client)
            break
        except QuotaExceededError:
            print("Cuota agotada. Intentando con siguiente credencial...")
            youtube_client = authenticate_next(prefix="playlists")
            if youtube_client is None:
                print("Todas las credenciales agotadas. Intenta mas tarde.")
                break
        except CredentialNotConfiguredError:
            print(
                "Credencial sin YouTube Data API v3 habilitada. "
                "Intentando con siguiente credencial..."
            )
            youtube_client = authenticate_next(prefix="playlists")
            if youtube_client is None:
                print("Todas las credenciales agotadas.")
                break
        except RateLimitError:
            print("Rate limit. Intenta mas tarde.")
            break
