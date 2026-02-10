#!/usr/bin/env python3
"""
Analiza portadas con un modelo NSFW y mueve carpetas sospechosas a Censurado.

Uso rápido (descarga NudeNet ONNX si falta):
    python "13. filtrar_portadas.py" --auto-descargar --umbral 0.35

Uso con modelo propio:
    python "13. filtrar_portadas.py" --modelo /ruta/al/modelo.onnx --umbral 0.35

Opcional:
    --base /ruta/de/escaneo (por defecto DIR_AUDIO_SCRIPTS)
    --ext png,jpg,jpeg (por defecto estos tres)
    --modelo-url URL (URL de modelo ONNX a descargar si no está)

Requisitos:
    - onnxruntime-gpu (o onnxruntime) para modelos .onnx (recomendado: GPU 3090 Ti).
    - Alternativamente, TensorFlow si usas un modelo Keras/SavedModel (.h5).
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import requests

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from nudenet import NudeClassifier
except ImportError:
    NudeClassifier = None

try:
    from nudenet import NudeDetector
except ImportError:
    NudeDetector = None

sys.path.append(str(Path(__file__).parent))
from config import DIR_AUDIO_SCRIPTS


CENSORED_DIR = Path(
    "/run/media/banar/Entretenimiento/01_edicion_automatizada/Censurado"
)
DEFAULT_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_MODEL_URL = (
    "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
)


def log(msg: str):
    print(msg, flush=True)


def get_unique_destination(base_dir: Path, folder_name: str) -> Path:
    candidate = base_dir / folder_name
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = base_dir / f"{folder_name}__{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def find_cover_image(folder_path: Path, valid_exts=DEFAULT_EXTS) -> Optional[Path]:
    preferred = [
        "cover_shadow.png",
        "cover_shadow.jpg",
        "cover_shadow.jpeg",
        "cover.png",
        "cover.jpg",
        "cover.jpeg",
    ]
    for name in preferred:
        candidate = folder_path / name
        if candidate.exists():
            return candidate
    for candidate in folder_path.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in valid_exts:
            return candidate
    return None


def move_to_censored(folder_path: Path, reason: str):
    CENSORED_DIR.mkdir(parents=True, exist_ok=True)
    destination = get_unique_destination(CENSORED_DIR, folder_path.name)
    try:
        shutil.move(str(folder_path), str(destination))
        log(f"[CENSURA] {folder_path.name} -> {destination} ({reason})")
    except Exception as exc:
        log(f"[CENSURA] No se pudo mover {folder_path}: {exc}")


def download_model_if_needed(model_path: Path, url: str) -> bool:
    if model_path.exists():
        return True
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Descargando modelo NSFW desde {url} ...")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(model_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        log(f"Modelo guardado en {model_path}")
        return True
    except Exception as exc:
        log(f"No se pudo descargar el modelo: {exc}")
        return False


def load_model(model_path: Path):
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        if ort is None:
            log("Instala 'onnxruntime-gpu' (o 'onnxruntime') para cargar modelos ONNX.")
            return None
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            log(f"Modelo ONNX cargado con providers: {session.get_providers()}")
            return session
        except Exception as exc:
            log(f"No se pudo cargar el modelo ONNX {model_path}: {exc}")
            return None
    else:
        if tf is None:
            log(
                "TensorFlow no está instalado. Instala 'tensorflow' para usar modelos Keras."
            )
            return None
        try:
            return tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as exc:
            log(f"No se pudo cargar el modelo TF/Keras {model_path}: {exc}")
            return None


def predict_nsfw(model, image_path: Path) -> Optional[float]:
    try:
        # NudeClassifier (descarga interna de modelos)
        if NudeClassifier is not None and isinstance(model, NudeClassifier):
            res = model.classify(str(image_path))
            item = res.get(str(image_path)) if isinstance(res, dict) else None
            if not item:
                return None
            # Preferimos 'unsafe' si existe
            for key in ("unsafe", "porn", "sexy", "nsfw"):
                if key in item:
                    return float(item[key])
            # Fallback: prob más alta
            return float(max(item.values())) if item else None

        # NudeDetector (detección de partes, usa modelo incluido en el paquete)
        if NudeDetector is not None and isinstance(model, NudeDetector):
            detections = model.detect(str(image_path))
            if not detections:
                return 0.0
            scores = [d.get("score", 0.0) for d in detections if isinstance(d, dict)]
            return float(max(scores)) if scores else 0.0

        # ONNXRuntime
        if ort is not None and isinstance(model, ort.InferenceSession):
            import PIL.Image as Image

            img = Image.open(image_path).convert("RGB").resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # NCHW
            arr = np.expand_dims(arr, 0)
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: arr})
            if not outputs:
                return None
            return float(np.max(outputs[0]))

        # TensorFlow fallback
        if tf is None:
            return None
        data = tf.io.read_file(str(image_path))
        img = tf.image.decode_image(data, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        preds = model(img, training=False)
        arr = preds.numpy() if hasattr(preds, "numpy") else preds
        if arr is None:
            return None
        return float(np.max(arr))
    except Exception as exc:
        log(f"Error evaluando {image_path.name}: {exc}")
        return None


def process_folder(folder_path: Path, model, threshold: float, valid_exts):
    if not folder_path.is_dir():
        return
    cover = find_cover_image(folder_path, valid_exts)
    if not cover:
        log(f"[SKIP] Sin portada en {folder_path.name}")
        return

    score = predict_nsfw(model, cover)
    if score is None:
        log(f"[SKIP] No se pudo evaluar {folder_path.name}")
        return

    if score >= threshold:
        move_to_censored(folder_path, f"NSFW score {score:.3f} >= {threshold}")
    else:
        log(f"[OK] {folder_path.name} score {score:.3f} < {threshold}")


def list_folders(base_dir: Path):
    if not base_dir.exists():
        return []
    return [p for p in base_dir.iterdir() if p.is_dir()]


def parse_exts(raw: str):
    return {
        f".{ext.strip().lower().lstrip('.')}" for ext in raw.split(",") if ext.strip()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filtra portadas con modelo NSFW y mueve carpetas a Censurado"
    )
    parser.add_argument(
        "--modelo",
        help="Ruta al modelo (.onnx recomendado; también se acepta .h5/.SavedModel)",
    )
    parser.add_argument(
        "--base", help="Carpeta base a escanear (default: DIR_AUDIO_SCRIPTS)"
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=float(os.environ.get("NSFW_THRESHOLD", "0.35")),
        help="Umbral NSFW (default 0.35)",
    )
    parser.add_argument(
        "--ext", default="png,jpg,jpeg", help="Extensiones de imagen separadas por coma"
    )
    parser.add_argument(
        "--auto-descargar",
        action="store_true",
        help="Descarga automática del modelo NudeNet ONNX si no existe",
    )
    parser.add_argument(
        "--modelo-url",
        default=DEFAULT_MODEL_URL,
        help="URL del modelo ONNX a descargar si no está presente",
    )
    args = parser.parse_args()

    base_dir = Path(args.base) if args.base else Path(DIR_AUDIO_SCRIPTS)
    if not base_dir.exists():
        log(f"No existe la carpeta base: {base_dir}")
        return

    valid_exts = parse_exts(args.ext)

    model = None
    model_path: Optional[Path] = None

    if args.modelo:
        model_path = Path(args.modelo)
        model = load_model(model_path)
    elif args.auto_descargar:
        if NudeClassifier is not None:
            try:
                model = NudeClassifier()
                log(
                    "Usando NudeClassifier (auto-descarga interna del modelo por el paquete)."
                )
            except Exception as exc:
                log(f"No se pudo inicializar NudeClassifier: {exc}")
                model = None
        if model is None and NudeDetector is not None:
            try:
                model = NudeDetector()
                log("Usando NudeDetector (modelo incluido en el paquete).")
            except Exception as exc:
                log(f"No se pudo inicializar NudeDetector: {exc}")
                model = None
        if model is None:
            log(
                "No se pudo usar NudeClassifier; proporciona --modelo con un ONNX/Keras válido."
            )
            return
    else:
        log(
            "Debes indicar --modelo o usar --auto-descargar para bajar un ONNX por defecto."
        )
        return

    if model is None:
        return

    folders = list_folders(base_dir)
    if not folders:
        log(f"No hay carpetas en {base_dir}")
        return

    log(f"Analizando {len(folders)} carpetas en {base_dir} con umbral {args.umbral}...")
    for folder in folders:
        process_folder(folder, model, args.umbral, valid_exts)


if __name__ == "__main__":
    main()
