#!/usr/bin/env python3
import argparse
import random
import re
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

from config import DIR_LIMPIEZA, DIR_UPLOAD


def cargar_modulo_subir_api():
    script_path = Path(__file__).with_name("9. subir_video_API.py")
    if not script_path.exists():
        print(f"No se encontro el script base: {script_path}")
        return None
    loader = SourceFileLoader("subir_video_api", str(script_path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def leer_titulo_desde_txt(folder_path):
    for txt_path in sorted(folder_path.glob("*.txt")):
        try:
            with txt_path.open("r", encoding="utf-8", errors="ignore") as handle:
                line = handle.readline().replace("\0", "").strip()
                if line:
                    return line
        except Exception:
            continue
    return None


def obtener_titulo_busqueda(folder_path, repertorio, helpers):
    titulo_txt = leer_titulo_desde_txt(folder_path)
    if titulo_txt:
        return titulo_txt
    return helpers.build_preview_title(folder_path, repertorio)


def verificacion_manual(folder_path, repertorio, helpers, base_dir):
    titulo = obtener_titulo_busqueda(folder_path, repertorio, helpers)
    helpers.abrir_busqueda_youtube(titulo)
    respuesta = input("Ya esta subido? [Y/n]: ")
    normalizada = re.sub(r"[^a-z]", "", respuesta.strip().lower())
    if normalizada.startswith(("y", "s")):
        if helpers.safe_delete_folder(folder_path, base_dir):
            print(f"Carpeta eliminada: {folder_path.name}")
        else:
            print(f"No se pudo eliminar: {folder_path.name}")
        return True
    if normalizada and normalizada not in {"n", "no"}:
        print("Entrada no valida, se continua con la verificacion.")
    return False


def elegir_origen_interactivo():
    while True:
        print("Elige carpeta base para verificar:")
        print("  1) 01_limpieza_de_impurezas")
        print("  2) upload_video")
        respuesta = input("Opcion (1/2): ").strip().lower()
        if respuesta in {"1", "limpieza", "l"}:
            return "limpieza"
        if respuesta in {"2", "upload", "u"}:
            return "upload"
        print("Opcion no valida, intenta de nuevo.")


def main():
    parser = argparse.ArgumentParser(description="Verificacion manual sin eliminar carpetas.")
    parser.add_argument(
        "--origen",
        choices=["upload", "limpieza"],
        default=None,
        help="Carpeta base a verificar (upload_video o 01_limpieza_de_impurezas).",
    )
    parser.add_argument("--ruta", help="Ruta absoluta a verificar (sobrescribe --origen).")
    parser.add_argument("--carpeta", help="Carpeta especifica dentro de la carpeta base")
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Cantidad maxima de carpetas a verificar (default: todas)",
    )
    parser.add_argument("--todo", action="store_true", help="Verificar todas las carpetas disponibles")
    parser.add_argument(
        "--permitir-sin-video",
        action="store_true",
        help="Verificar carpetas aunque no haya .mp4 (util para limpieza_de_impurezas).",
    )
    args = parser.parse_args()

    helpers = cargar_modulo_subir_api()
    if not helpers:
        return

    helpers.cargar_env()
    repertorio = helpers.cargar_repertorio()

    if args.ruta:
        base_dir = Path(args.ruta)
    else:
        origen = args.origen or elegir_origen_interactivo()
        base_dir = DIR_LIMPIEZA if origen == "limpieza" else DIR_UPLOAD

    if args.carpeta:
        upload_dir = Path(args.carpeta)
        if not upload_dir.is_absolute():
            upload_dir = base_dir / args.carpeta
    else:
        upload_dir = base_dir

    if not upload_dir.exists():
        print(f"No existe la carpeta: {upload_dir}")
        return

    if args.carpeta:
        folders = [upload_dir]
    else:
        folders = [path for path in upload_dir.iterdir() if path.is_dir()]
        random.shuffle(folders)

    if not folders:
        print("No hay carpetas para verificar.")
        return

    if args.todo:
        max_items = None
    elif args.limite is None:
        max_items = None
    else:
        max_items = max(0, args.limite)
    revisados = 0
    ya_subidos = 0
    allow_no_video = args.permitir_sin_video or base_dir == DIR_LIMPIEZA

    for folder_path in folders:
        if max_items is not None and revisados >= max_items:
            break
        video_path = helpers.encontrar_video(folder_path)
        if not video_path and not allow_no_video:
            print(f"No hay video .mp4 en {folder_path}, se omite.")
            continue
        if verificacion_manual(folder_path, repertorio, helpers, base_dir):
            ya_subidos += 1
        revisados += 1

    print(f"Verificacion terminada. Revisados: {revisados} | Marcados como ya subidos: {ya_subidos}")


if __name__ == "__main__":
    main()
