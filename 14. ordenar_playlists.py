#!/usr/bin/env python3
"""Ordena las playlists dentro de las secciones del canal en orden alfabetico.

Si una seccion no se puede actualizar via API (demasiadas playlists),
se divide automaticamente en sub-secciones alfabeticas:
  "Titulo (A-F)", "Titulo (G-M)", etc.

El tamanio de cada sub-seccion se calcula dinamicamente segun los slots
disponibles en el canal (maximo 10 secciones segun la API).
"""
import argparse
import importlib.util
import math
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

MAX_SECCIONES_CANAL = 10


# ---------------------------------------------------------------------------
# Funciones de la API channelSections
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


def get_playlist_titles(youtube, playlist_ids):
    """Obtiene los titulos de las playlists por sus IDs (en bloques de 50)."""
    titles = {}
    for i in range(0, len(playlist_ids), 50):
        chunk = playlist_ids[i : i + 50]
        ids_str = ",".join(chunk)
        resp = run_with_backoff(
            lambda ids=ids_str: youtube.playlists()
            .list(
                part="snippet",
                id=ids,
                maxResults=50,
                fields="items(id,snippet/title)",
            )
            .execute(),
            descripcion="obtener titulos de playlists",
        )
        for item in resp.get("items", []):
            playlist_id = item.get("id")
            title = item.get("snippet", {}).get("title", "")
            if playlist_id:
                titles[playlist_id] = clean_title(title) or ""
    return titles


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


def clone_section_with_playlists(section, playlist_ids):
    """Clona una seccion reemplazando solo la lista de playlists."""
    cloned = {
        "id": section["id"],
        "snippet": dict(section.get("snippet", {})),
        "contentDetails": dict(section.get("contentDetails", {})),
    }
    cloned["contentDetails"]["playlists"] = playlist_ids
    return cloned


def delete_section(youtube, section_id):
    """Elimina una seccion del canal."""
    run_with_backoff(
        lambda: youtube.channelSections()
        .delete(id=section_id)
        .execute(),
        descripcion=f"eliminar seccion {section_id}",
    )


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


def insert_section(youtube, channel_id, title, playlist_ids, position=None):
    """Crea una nueva seccion de tipo multiplePlaylists."""
    body = {
        "snippet": {
            "type": "multiplePlaylists",
            "title": title,
            "channelId": channel_id,
        },
        "contentDetails": {
            "playlists": playlist_ids,
        },
    }
    if position is not None:
        body["snippet"]["position"] = position

    run_with_backoff(
        lambda: youtube.channelSections()
        .insert(part="snippet,contentDetails", body=body)
        .execute(),
        descripcion=f"crear seccion '{title}'",
    )


def sort_key_for_title(title):
    """Clave de ordenamiento: normaliza el titulo para orden alfabetico."""
    return normalize_title(title) or ""


# ---------------------------------------------------------------------------
# Division de secciones grandes en sub-secciones alfabeticas
# ---------------------------------------------------------------------------


def _group_key(title):
    """Devuelve la clave de agrupacion: primera letra (A-Z), digito (0-9), o '#'."""
    normalized = normalize_title(title) or ""
    for ch in normalized:
        if ch.isalpha():
            return ch.upper()
        if ch.isdigit():
            return ch
    return "#"


def split_into_alpha_groups(sorted_playlists, max_sub_sections):
    """Divide una lista de (id, titulo) ordenada en grupos para sub-secciones.

    El numero de sub-secciones resultantes nunca excede max_sub_sections.
    Retorna lista de (rango_str, [(id, titulo), ...]).
    """
    if max_sub_sections <= 0:
        return []
    if max_sub_sections == 1:
        return [(_make_range_str_from_items(sorted_playlists), sorted_playlists)]

    # Agrupar por clave (letra/digito)
    groups_by_key = {}
    key_order = []
    for pid, ptitle in sorted_playlists:
        key = _group_key(ptitle)
        if key not in groups_by_key:
            groups_by_key[key] = []
            key_order.append(key)
        groups_by_key[key].append((pid, ptitle))

    # Si hay menos grupos que max_sub_sections, combinar letras adyacentes
    # Si hay mas, fusionar letras adyacentes hasta que quepan
    bins = [(k, groups_by_key[k]) for k in key_order]

    # Fusionar bins hasta que tengamos <= max_sub_sections
    while len(bins) > max_sub_sections:
        # Encontrar el par adyacente cuya fusion sea la mas pequenia
        best_idx = 0
        best_size = float("inf")
        for i in range(len(bins) - 1):
            combined = len(bins[i][1]) + len(bins[i + 1][1])
            if combined < best_size:
                best_size = combined
                best_idx = i
        # Fusionar bins[best_idx] y bins[best_idx + 1]
        merged_keys = bins[best_idx][0] + "," + bins[best_idx + 1][0]
        merged_items = bins[best_idx][1] + bins[best_idx + 1][1]
        bins[best_idx] = (merged_keys, merged_items)
        del bins[best_idx + 1]

    # Generar resultado con rangos legibles
    result = []
    for keys_str, items in bins:
        keys = keys_str.split(",")
        range_str = _range_from_keys(keys)
        result.append((range_str, items))

    return result


def _range_from_keys(keys):
    """Genera un rango legible: 'A-F', 'G', '1-3', etc."""
    if not keys:
        return "#"
    first = keys[0]
    last = keys[-1]
    if first == last:
        return first
    return f"{first}-{last}"


def _make_range_str_from_items(sorted_playlists):
    """Genera rango a partir de los items."""
    if not sorted_playlists:
        return "#"
    first_key = _group_key(sorted_playlists[0][1])
    last_key = _group_key(sorted_playlists[-1][1])
    if first_key == last_key:
        return first_key
    return f"{first_key}-{last_key}"


def _is_update_too_large_error(error):
    """Determina si un HttpError indica que la seccion es demasiado grande para update."""
    error_str = str(error)
    if "maxPlaylistExceeded" in error_str:
        return True
    if error.resp.status == 400 and "invalid argument" in error_str.lower():
        return True
    return False


def _is_invalid_section_reference_error(error):
    """Detecta referencias invalidadas dentro de una seccion."""
    reasons = set(get_error_reasons(error))
    return bool(
        reasons
        & {
            "channelNotActive",
            "playlistNotFound",
            "playlistIsPrivate",
        }
    )


def main(youtube=None):
    parser = argparse.ArgumentParser(
        description="Ordena playlists dentro de las secciones del canal en orden alfabetico."
    )
    parser.add_argument(
        "--seccion",
        type=str,
        action="append",
        help="Nombre de la seccion a ordenar (puede repetirse). Si no se indica, ordena todas.",
    )
    parser.add_argument(
        "--solo-mostrar",
        action="store_true",
        help="Solo muestra las secciones y sus playlists sin modificar nada.",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
        help="Segundos de espera entre actualizaciones de secciones (default: 1.0).",
    )
    parser.add_argument(
        "--max-por-seccion",
        type=int,
        default=0,
        help="Maximo de playlists por sub-seccion al dividir. 0 = calcular automaticamente.",
    )
    parser.add_argument(
        "--eliminar-undefined-vacias",
        action="store_true",
        help="Elimina secciones 'channelsectiontypeundefined' vacias para liberar slots.",
    )
    args = parser.parse_args()

    if youtube is None:
        youtube = authenticate(prefix="playlists")

    print("Obteniendo secciones del canal...")
    sections = list_channel_sections(youtube)
    if not sections:
        print("No se encontraron secciones en el canal.")
        return

    if args.eliminar_undefined_vacias:
        undefined_vacias = [s for s in sections if is_empty_undefined_section(s)]
        if undefined_vacias:
            print(
                "Eliminando secciones vacias 'channelsectiontypeundefined': "
                f"{len(undefined_vacias)}"
            )
            for section in undefined_vacias:
                delete_section(youtube, section["id"])
                print(f"  ✓ Seccion vacia eliminada: {section['id']}")
                if args.pausa:
                    time.sleep(args.pausa)
            print("Recargando secciones del canal...")
            sections = list_channel_sections(youtube)
            if not sections:
                print("No se encontraron secciones en el canal tras la limpieza.")
                return

    total_sections = len(sections)

    # Diagnostico: mostrar todas las secciones encontradas
    print(f"Secciones encontradas: {total_sections}")
    for s in sections:
        snippet = s.get("snippet", {})
        stype = snippet.get("type", "???")
        stitle = snippet.get("title", "(sin titulo)")
        pids = s.get("contentDetails", {}).get("playlists", [])
        print(f"  [{stype}] {stitle} ({len(pids)} playlists)")

    # Filtrar secciones que contienen playlists
    playlist_sections = [
        s
        for s in sections
        if s.get("contentDetails", {}).get("playlists")
    ]

    if not playlist_sections:
        print("No se encontraron secciones que contengan playlists.")
        return

    # Filtrar por --seccion si se indica
    if args.seccion:
        target_keys = {normalize_title(s) for s in args.seccion}
        filtered = [
            s
            for s in playlist_sections
            if normalize_title(s.get("snippet", {}).get("title", "")) in target_keys
        ]
        if not filtered:
            print("No se encontraron secciones que coincidan con los filtros.")
            print("Secciones disponibles:")
            for s in playlist_sections:
                title = s.get("snippet", {}).get("title", "")
                count = len(s.get("contentDetails", {}).get("playlists", []))
                print(f"  - {title} ({count} playlists)")
            return
        playlist_sections = filtered

    # Recolectar todos los IDs de playlists para obtener titulos en bloque
    all_playlist_ids = set()
    for section in playlist_sections:
        pids = section.get("contentDetails", {}).get("playlists", [])
        all_playlist_ids.update(pids)

    if not all_playlist_ids:
        print("Las secciones seleccionadas no contienen playlists.")
        return

    print(f"Obteniendo titulos de {len(all_playlist_ids)} playlists...")
    playlist_titles = get_playlist_titles(youtube, list(all_playlist_ids))

    updated = 0
    for section in playlist_sections:
        title = section.get("snippet", {}).get("title", "")
        channel_id = section.get("snippet", {}).get("channelId", "")
        playlists = section.get("contentDetails", {}).get("playlists", [])

        # Emparejar IDs con titulos y ordenar alfabeticamente
        playlist_with_titles = [
            (pid, playlist_titles.get(pid, pid)) for pid in playlists
        ]
        unresolved_ids = [pid for pid in playlists if pid not in playlist_titles]
        sorted_playlists = sorted(
            playlist_with_titles, key=lambda x: sort_key_for_title(x[1])
        )
        sorted_ids = [pid for pid, _ in sorted_playlists]

        already_sorted = sorted_ids == playlists

        print(f"\nSeccion: {title}")
        print(f"  Playlists ({len(playlists)}):")
        for i, (pid, ptitle) in enumerate(sorted_playlists):
            original_pos = playlists.index(pid)
            marker = "" if original_pos == i else f" (era pos {original_pos + 1})"
            print(f"    {i + 1}. {ptitle}{marker}")
        if unresolved_ids:
            print("  Advertencia: estas playlists no devolvieron titulo en la API:")
            for pid in unresolved_ids:
                print(f"    - {pid}")

        if already_sorted:
            print("  Ya esta en orden alfabetico.")
            continue

        if args.solo_mostrar:
            print("  (modo solo-mostrar, no se actualiza)")
            continue

        # --- Intentar update directo ---
        section["contentDetails"]["playlists"] = sorted_ids
        try:
            update_section_playlists(youtube, section)
            updated += 1
            print(f"  Seccion actualizada.")
            if args.pausa:
                time.sleep(args.pausa)
            continue
        except HttpError as e:
            if _is_invalid_section_reference_error(e):
                reasons = ", ".join(get_error_reasons(e)) or "sin detalle"
                if unresolved_ids:
                    valid_ids = [pid for pid in sorted_ids if pid in playlist_titles]
                    section_type = section.get("snippet", {}).get("type", "")

                    if not valid_ids:
                        print(
                            "  Todas las playlists de esta seccion quedaron invalidas "
                            f"({reasons}). Se omite."
                        )
                        continue

                    if section_type == "singlePlaylist" and len(valid_ids) != 1:
                        print(
                            "  La seccion es singlePlaylist y no se puede reparar "
                            f"automaticamente ({reasons}). Se omite."
                        )
                        continue

                    print(
                        "  La API rechazo esta seccion por playlists o canales "
                        f"inactivos/inaccesibles ({reasons}). "
                        f"Reintentando sin {len(unresolved_ids)} playlists invalidas..."
                    )
                    repaired_section = clone_section_with_playlists(section, valid_ids)
                    update_section_playlists(youtube, repaired_section)
                    updated += 1
                    print(
                        f"  Seccion actualizada tras limpiar {len(unresolved_ids)} "
                        "playlists invalidas."
                    )
                    if args.pausa:
                        time.sleep(args.pausa)
                    continue

                print(
                    "  La API rechazo esta seccion por playlists o canales "
                    f"inactivos/inaccesibles ({reasons}). Se omite."
                )
                continue
            if not _is_update_too_large_error(e):
                raise
            print(f"  Update directo fallo ({len(sorted_ids)} playlists). Intentando dividir...")

        # --- Dividir en sub-secciones ---
        # Calcular slots disponibles: eliminamos la seccion actual (-1),
        # y vemos cuantos slots quedan hasta el limite
        slots_disponibles = MAX_SECCIONES_CANAL - (total_sections - 1)
        if slots_disponibles < 2:
            print(f"  No hay slots disponibles para dividir "
                  f"({total_sections} secciones actuales, limite {MAX_SECCIONES_CANAL}). "
                  f"Saltando.")
            continue

        max_per = args.max_por_seccion
        if max_per <= 0:
            # Calcular automaticamente
            max_per = math.ceil(len(sorted_ids) / slots_disponibles)

        chunks = split_into_alpha_groups(sorted_playlists, slots_disponibles)

        print(f"  Se dividira en {len(chunks)} sub-secciones "
              f"(slots disponibles: {slots_disponibles}):")
        for range_str, chunk in chunks:
            sub_title = f"{title} ({range_str})"
            print(f"    - {sub_title} ({len(chunk)} playlists)")

        # 1. Eliminar la seccion original PRIMERO para liberar slot
        delete_section(youtube, section["id"])
        total_sections -= 1
        print(f"  Seccion original '{title}' eliminada.")
        time.sleep(args.pausa)

        # 2. Crear sub-secciones
        for range_str, chunk in chunks:
            sub_title = f"{title} ({range_str})"
            chunk_ids = [pid for pid, _ in chunk]
            try:
                insert_section(youtube, channel_id, sub_title, chunk_ids)
                total_sections += 1
                print(f"  Creada: {sub_title} ({len(chunk_ids)} playlists)")
            except HttpError as e:
                if "maxChannelSectionExceeded" in str(e):
                    print(f"  Limite de secciones alcanzado al crear '{sub_title}'. "
                          f"Quedan {len(chunk_ids)} playlists sin seccion.")
                    break
                raise
            time.sleep(args.pausa)

        updated += 1

    print(f"\nListo. Secciones procesadas: {updated}.")


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
