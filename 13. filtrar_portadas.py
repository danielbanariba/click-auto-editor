#!/usr/bin/env python3
"""
Filtra portadas problemáticas y mueve carpetas a revisión.

Políticas disponibles:
- `cadaver-real`: intenta detectar portadas con posible fotografía de cadáver
  humano real. Prioriza backend local vía Ollama para usar tu GPU; OpenAI queda
  como fallback opcional. Combina visión y un score heurístico de fotorealismo
  para reducir falsos positivos de ilustraciones.
- `nsfw`: mantiene el flujo legacy basado en NudeNet / ONNX / TensorFlow.

Ejemplos:
    python "13. filtrar_portadas.py" --politica cadaver-real --backend ollama --limite 20 --dry-run
    python "13. filtrar_portadas.py" --politica cadaver-real --destino /ruta/revision
    python "13. filtrar_portadas.py" --politica nsfw --auto-descargar --umbral 0.35

Notas:
- `cadaver-real` mueve por defecto a `DIR_NEED_CENSURED` para revisión manual.
- `cadaver-real` usa `ollama` si está disponible en `http://localhost:11434`
  o en `OLLAMA_HOST`. Si no, puede usar OpenAI como fallback.
- El score de fotorealismo es heurístico. La idea es ser conservador y apartar
  sospechosos, no asumir precisión perfecta.
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

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
from config import DIR_AUDIO_SCRIPTS, DIR_NEED_CENSURED


LEGACY_CENSORED_DIR = Path(
    "/run/media/banar/Entretenimiento/01_edicion_automatizada/Censurado"
)
DEFAULT_REVIEW_DIR = DIR_NEED_CENSURED
DEFAULT_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_MODEL_URL = (
    "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
)
DEFAULT_MODERATION_MODEL = os.environ.get(
    "OPENAI_MODERATION_MODEL",
    "omni-moderation-latest",
)
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "filtrar_portadas.jsonl"
MODERATION_URL = "https://api.openai.com/v1/moderations"
OLLAMA_TAGS_PATH = "/api/tags"
OLLAMA_CHAT_PATH = "/api/chat"
OLLAMA_CADAVER_SCHEMA = {
    "type": "object",
    "properties": {
        "contains_real_human_corpse": {"type": "boolean"},
        "real_human_corpse_score": {"type": "number"},
        "graphic_violence_score": {"type": "number"},
        "photographic_realism_score": {"type": "number"},
        "reason_short": {"type": "string"},
    },
    "required": [
        "contains_real_human_corpse",
        "real_human_corpse_score",
        "graphic_violence_score",
        "photographic_realism_score",
        "reason_short",
    ],
}
OLLAMA_CADAVER_PROMPT = (
    "Classify this album cover for YouTube safety. Return JSON only using the "
    "provided schema. Detect only real photographic human corpses or real dead "
    "human remains. Ignore drawings, paintings, zombies, monsters, skeleton art, "
    "medical illustrations, statues, gore art, or fictional scenes. "
    "real_human_corpse_score should be high only when a real dead human body is "
    "visibly present in a photograph. photographic_realism_score measures whether "
    "the image looks like a real photo rather than artwork. "
    "graphic_violence_score measures visible gore or disturbing injury."
)


@dataclass(frozen=True)
class CadaverThresholds:
    graphic_threshold: float
    photo_threshold: float
    combined_threshold: float


@dataclass
class AnalysisResult:
    folder: str
    cover: Optional[str]
    policy: str
    detector: str
    status: str
    flagged: bool
    moved_to: Optional[str]
    reason: str
    primary_score: Optional[float]
    scores: dict[str, float]


def log(msg: str):
    print(msg, flush=True)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def append_result_log(log_path: Path, result: AnalysisResult):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "folder": result.folder,
        "cover": result.cover,
        "policy": result.policy,
        "detector": result.detector,
        "status": result.status,
        "flagged": result.flagged,
        "moved_to": result.moved_to,
        "reason": result.reason,
        "primary_score": result.primary_score,
        "scores": result.scores,
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    for candidate in sorted(folder_path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in valid_exts:
            return candidate
    return None


def move_to_destination(
    folder_path: Path,
    destination_root: Path,
    reason: str,
    dry_run: bool = False,
) -> Optional[Path]:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = get_unique_destination(destination_root, folder_path.name)
    if dry_run:
        log(f"[DRY-RUN] {folder_path.name} -> {destination} ({reason})")
        return destination
    try:
        shutil.move(str(folder_path), str(destination))
        log(f"[CENSURA] {folder_path.name} -> {destination} ({reason})")
        return destination
    except Exception as exc:
        log(f"[CENSURA] No se pudo mover {folder_path}: {exc}")
        return None


def list_folders(base_dir: Path, limit: Optional[int] = None) -> list[Path]:
    if not base_dir.exists():
        return []
    folders = [path for path in base_dir.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.name.lower())
    if limit is not None and limit > 0:
        return folders[:limit]
    return folders


def parse_exts(raw: str):
    return {
        f".{ext.strip().lower().lstrip('.')}" for ext in raw.split(",") if ext.strip()
    }


def parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_rgb_image(image_path: Path, size: int = 256) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def encode_image_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue()


def encode_image_base64(image_path: Path) -> str:
    encoded = base64.b64encode(encode_image_bytes(image_path)).decode("ascii")
    return encoded


def encode_image_as_data_url(image_path: Path) -> str:
    encoded = encode_image_base64(image_path)
    return f"data:image/jpeg;base64,{encoded}"


def normalize_base_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def load_openai_api_key(env_name: str) -> Optional[str]:
    raw = os.environ.get(env_name, "").strip()
    return raw or None


def load_ollama_api_key(env_name: str) -> Optional[str]:
    raw = os.environ.get(env_name, "").strip()
    return raw or None


def build_ollama_headers(api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def ollama_is_available(
    host: str,
    timeout: int = 10,
    api_key: Optional[str] = None,
) -> bool:
    base_url = normalize_base_url(host)
    if not base_url:
        return False
    try:
        response = requests.get(
            f"{base_url}{OLLAMA_TAGS_PATH}",
            headers=build_ollama_headers(api_key),
            timeout=min(timeout, 10),
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def extract_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError("La respuesta del modelo no es texto JSON.")
    payload = text.strip()
    if not payload:
        raise ValueError("La respuesta del modelo llegó vacía.")
    return json.loads(payload)


def read_score(payload: dict[str, Any], key: str) -> float:
    return clamp01(float(payload.get(key) or 0.0))


def extract_moderation_score(score_map: dict[str, Any], key: str) -> float:
    value = score_map.get(key)
    if value is None and "/" in key:
        value = score_map.get(key.replace("/", "_"))
    return clamp01(float(value or 0.0))


def moderate_image_with_openai(
    image_path: Path,
    api_key: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_as_data_url(image_path),
                },
            }
        ],
    }
    response = requests.post(
        MODERATION_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("La respuesta de moderación no trae resultados.")

    first = results[0]
    raw_scores = first.get("category_scores") or {}
    raw_categories = first.get("categories") or {}
    scores = {
        "violence": extract_moderation_score(raw_scores, "violence"),
        "violence_graphic": extract_moderation_score(raw_scores, "violence/graphic"),
    }
    flags = {
        "violence": bool(raw_categories.get("violence", False)),
        "violence_graphic": bool(raw_categories.get("violence/graphic", False)),
    }
    return {
        "flagged": bool(first.get("flagged", False)),
        "scores": scores,
        "flags": flags,
    }


def estimate_photographic_score(image_path: Path) -> tuple[float, dict[str, float]]:
    arr = load_rgb_image(image_path, size=256)
    gray = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])

    hist, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    entropy_norm = clamp01(entropy / np.log2(64))

    quantized = np.floor(arr * 15.0).astype(np.uint8).reshape(-1, 3)
    unique_ratio = clamp01(len(np.unique(quantized, axis=0)) / 800.0)

    diff_h = np.linalg.norm(arr[:, 1:] - arr[:, :-1], axis=2)
    diff_v = np.linalg.norm(arr[1:, :] - arr[:-1, :], axis=2)
    flat_ratio = float(
        (
            np.mean(diff_h < 0.035) +
            np.mean(diff_v < 0.035)
        )
        / 2.0
    )
    detail_score = clamp01(1.0 - flat_ratio)

    channel_std = clamp01(float(arr.std()) / 0.25)

    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    saturation = np.where(rgb_max > 0, (rgb_max - rgb_min) / rgb_max, 0.0)
    saturation_var = clamp01(float(np.std(saturation)) / 0.28)

    photo_score = clamp01(
        (entropy_norm * 0.30)
        + (unique_ratio * 0.30)
        + (detail_score * 0.25)
        + (channel_std * 0.10)
        + (saturation_var * 0.05)
    )

    details = {
        "photo_score": photo_score,
        "entropy": entropy_norm,
        "unique_ratio": unique_ratio,
        "detail_score": detail_score,
        "channel_std": channel_std,
        "saturation_var": saturation_var,
    }
    return photo_score, details


def calculate_cadaver_score(
    violence_graphic: float,
    violence: float,
    photo_score: float,
) -> float:
    graphic_component = max(violence_graphic, violence * 0.45)
    return clamp01(graphic_component * photo_score)


def should_flag_cadaver_real(
    violence_graphic: float,
    violence: float,
    photo_score: float,
    thresholds: CadaverThresholds,
) -> tuple[bool, float, str]:
    cadaver_score = calculate_cadaver_score(violence_graphic, violence, photo_score)
    flagged = (
        violence_graphic >= thresholds.graphic_threshold
        and photo_score >= thresholds.photo_threshold
        and cadaver_score >= thresholds.combined_threshold
    )
    reason = (
        f"graphic={violence_graphic:.3f}, "
        f"violence={violence:.3f}, "
        f"photo={photo_score:.3f}, "
        f"cadaver={cadaver_score:.3f}"
    )
    return flagged, cadaver_score, reason


def analyze_cadaver_real_cover(
    image_path: Path,
    api_key: str,
    model: str,
    timeout: int,
    thresholds: CadaverThresholds,
) -> tuple[bool, float, str, dict[str, float], str]:
    moderation = moderate_image_with_openai(image_path, api_key, model, timeout)
    photo_score, photo_details = estimate_photographic_score(image_path)

    violence = moderation["scores"]["violence"]
    violence_graphic = moderation["scores"]["violence_graphic"]
    flagged, cadaver_score, reason = should_flag_cadaver_real(
        violence_graphic=violence_graphic,
        violence=violence,
        photo_score=photo_score,
        thresholds=thresholds,
    )

    scores = {
        "cadaver_score": cadaver_score,
        "violence": violence,
        "violence_graphic": violence_graphic,
        "photo_score": photo_details["photo_score"],
        "entropy": photo_details["entropy"],
        "unique_ratio": photo_details["unique_ratio"],
        "detail_score": photo_details["detail_score"],
        "channel_std": photo_details["channel_std"],
        "saturation_var": photo_details["saturation_var"],
    }

    moderation_flags = moderation.get("flags") or {}
    detector = f"openai-moderation:{model}"
    if moderation_flags.get("violence_graphic"):
        reason += " | categoria violence/graphic=true"
    elif moderation_flags.get("violence"):
        reason += " | categoria violence=true"
    return flagged, cadaver_score, reason, scores, detector


def analyze_cadaver_real_cover_ollama(
    image_path: Path,
    host: str,
    model: str,
    timeout: int,
    thresholds: CadaverThresholds,
    api_key: Optional[str] = None,
) -> tuple[bool, float, str, dict[str, float], str]:
    payload = {
        "model": model,
        "stream": False,
        "format": OLLAMA_CADAVER_SCHEMA,
        "messages": [
            {
                "role": "user",
                "content": OLLAMA_CADAVER_PROMPT,
                "images": [encode_image_base64(image_path)],
            }
        ],
        "options": {
            "temperature": 0,
        },
    }
    response = requests.post(
        f"{normalize_base_url(host)}{OLLAMA_CHAT_PATH}",
        headers=build_ollama_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    parsed = extract_json_object(message.get("content"))

    corpse_score = read_score(parsed, "real_human_corpse_score")
    graphic_score = read_score(parsed, "graphic_violence_score")
    model_photo_score = read_score(parsed, "photographic_realism_score")
    contains_real = bool(parsed.get("contains_real_human_corpse", False))
    reason_short = str(parsed.get("reason_short") or "").strip()

    heuristic_photo_score, photo_details = estimate_photographic_score(image_path)
    photo_score = clamp01(max(model_photo_score, heuristic_photo_score * 0.90))

    real_signal = contains_real or corpse_score >= max(
        0.65, thresholds.combined_threshold + 0.35
    )
    flagged = (
        real_signal
        and graphic_score >= thresholds.graphic_threshold
        and photo_score >= thresholds.photo_threshold
        and corpse_score >= thresholds.combined_threshold
    )
    reason = (
        f"corpse={corpse_score:.3f}, "
        f"graphic={graphic_score:.3f}, "
        f"photo={photo_score:.3f}, "
        f"photo_model={model_photo_score:.3f}, "
        f"photo_heur={heuristic_photo_score:.3f}"
    )
    if reason_short:
        reason += f" | {reason_short}"

    scores = {
        "cadaver_score": corpse_score,
        "violence_graphic": graphic_score,
        "photo_score": photo_score,
        "photo_model_score": model_photo_score,
        "photo_heuristic_score": heuristic_photo_score,
        "entropy": photo_details["entropy"],
        "unique_ratio": photo_details["unique_ratio"],
        "detail_score": photo_details["detail_score"],
        "channel_std": photo_details["channel_std"],
        "saturation_var": photo_details["saturation_var"],
    }
    return flagged, corpse_score, reason, scores, f"ollama:{model}"


def download_model_if_needed(model_path: Path, url: str) -> bool:
    if model_path.exists():
        return True
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Descargando modelo NSFW desde {url} ...")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(model_path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        log(f"Modelo guardado en {model_path}")
        return True
    except requests.RequestException as exc:
        log(f"No se pudo descargar el modelo: {exc}")
        return False


def load_nsfw_model(model_path: Path):
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

    if tf is None:
        log("TensorFlow no está instalado. Instala 'tensorflow' para modelos Keras.")
        return None
    try:
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        log(f"No se pudo cargar el modelo TF/Keras {model_path}: {exc}")
        return None


def predict_nsfw(model, image_path: Path) -> Optional[float]:
    try:
        if NudeClassifier is not None and isinstance(model, NudeClassifier):
            result = model.classify(str(image_path))
            item = result.get(str(image_path)) if isinstance(result, dict) else None
            if not item:
                return None
            for key in ("unsafe", "porn", "sexy", "nsfw"):
                if key in item:
                    return float(item[key])
            return float(max(item.values())) if item else None

        if NudeDetector is not None and isinstance(model, NudeDetector):
            detections = model.detect(str(image_path))
            if not detections:
                return 0.0
            scores = [d.get("score", 0.0) for d in detections if isinstance(d, dict)]
            return float(max(scores)) if scores else 0.0

        if ort is not None and isinstance(model, ort.InferenceSession):
            arr = load_rgb_image(image_path, size=224)
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, 0).astype(np.float32)
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: arr})
            if not outputs:
                return None
            return float(np.max(outputs[0]))

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


def build_nsfw_model(args) -> Optional[Any]:
    if args.modelo:
        return load_nsfw_model(Path(args.modelo))

    if not args.auto_descargar:
        log(
            "Para politica nsfw debes indicar --modelo o usar --auto-descargar."
        )
        return None

    model = None
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

    if model is not None:
        return model

    fallback_path = Path("models") / "nsfw_classifier.onnx"
    if not download_model_if_needed(fallback_path, args.modelo_url):
        return None
    return load_nsfw_model(fallback_path)


def analyze_nsfw_cover(
    image_path: Path,
    model: Any,
    threshold: float,
) -> tuple[bool, float, str, dict[str, float], str]:
    score = predict_nsfw(model, image_path)
    if score is None:
        raise RuntimeError("No se pudo calcular el score NSFW.")
    flagged = score >= threshold
    reason = f"nsfw_score={score:.3f} umbral={threshold:.3f}"
    return flagged, score, reason, {"nsfw_score": score}, "nsfw"


def build_default_cadaver_thresholds() -> CadaverThresholds:
    return CadaverThresholds(
        graphic_threshold=parse_float_env("CADAVER_GRAPHIC_THRESHOLD", 0.14),
        photo_threshold=parse_float_env("CADAVER_PHOTO_THRESHOLD", 0.58),
        combined_threshold=parse_float_env("CADAVER_COMBINED_THRESHOLD", 0.12),
    )


def resolve_cadaver_backend(
    backend: str,
    ollama_host: str,
    timeout: int,
    api_key: Optional[str],
    api_key_env: str,
    ollama_api_key: Optional[str] = None,
) -> Optional[str]:
    if backend == "ollama":
        return "ollama" if ollama_is_available(ollama_host, timeout, ollama_api_key) else None
    if backend == "openai":
        return "openai" if (api_key or load_openai_api_key(api_key_env)) else None
    if ollama_is_available(ollama_host, timeout, ollama_api_key):
        return "ollama"
    if api_key or load_openai_api_key(api_key_env):
        return "openai"
    return None


def scan_cover_path(
    image_path: Path,
    policy: str = "cadaver-real",
    backend: str = "auto",
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    moderation_model: str = DEFAULT_MODERATION_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_api_key: Optional[str] = None,
    timeout: int = 60,
    thresholds: Optional[CadaverThresholds] = None,
    nsfw_model: Any = None,
    nsfw_threshold: float = 0.35,
) -> dict[str, Any]:
    if policy == "cadaver-real":
        thresholds = thresholds or build_default_cadaver_thresholds()
        resolved_backend = resolve_cadaver_backend(
            backend=backend,
            ollama_host=ollama_host,
            timeout=timeout,
            api_key=api_key,
            api_key_env=api_key_env,
            ollama_api_key=ollama_api_key,
        )
        if not resolved_backend:
            return {
                "status": "skip",
                "flagged": False,
                "detector": "cadaver-real",
                "reason": (
                    "No hay backend disponible. Inicia Ollama en "
                    f"{normalize_base_url(ollama_host)}. {api_key_env} es solo "
                    "fallback opcional."
                ),
                "primary_score": None,
                "scores": {},
            }

        try:
            if resolved_backend == "ollama":
                flagged, primary_score, reason, scores, detector = (
                    analyze_cadaver_real_cover_ollama(
                        image_path=image_path,
                        host=ollama_host,
                        model=ollama_model,
                        timeout=timeout,
                        thresholds=thresholds,
                        api_key=ollama_api_key,
                    )
                )
            else:
                api_key = api_key or load_openai_api_key(api_key_env)
                flagged, primary_score, reason, scores, detector = (
                    analyze_cadaver_real_cover(
                        image_path=image_path,
                        api_key=api_key,
                        model=moderation_model,
                        timeout=timeout,
                        thresholds=thresholds,
                    )
                )
        except requests.RequestException as exc:
            return {
                "status": "error",
                "flagged": False,
                "detector": resolved_backend,
                "reason": f"Error HTTP moderando portada: {exc}",
                "primary_score": None,
                "scores": {},
            }
        except Exception as exc:
            return {
                "status": "error",
                "flagged": False,
                "detector": "cadaver-real",
                "reason": f"Error evaluando portada: {exc}",
                "primary_score": None,
                "scores": {},
            }

        return {
            "status": "flagged" if flagged else "ok",
            "flagged": flagged,
            "detector": detector,
            "reason": reason,
            "primary_score": primary_score,
            "scores": scores,
        }

    if nsfw_model is None:
        return {
            "status": "skip",
            "flagged": False,
            "detector": "nsfw",
            "reason": "No se proporcionó modelo NSFW",
            "primary_score": None,
            "scores": {},
        }

    try:
        flagged, primary_score, reason, scores, detector = analyze_nsfw_cover(
            image_path=image_path,
            model=nsfw_model,
            threshold=nsfw_threshold,
        )
    except Exception as exc:
        return {
            "status": "error",
            "flagged": False,
            "detector": "nsfw",
            "reason": f"Error evaluando portada: {exc}",
            "primary_score": None,
            "scores": {},
        }

    return {
        "status": "flagged" if flagged else "ok",
        "flagged": flagged,
        "detector": detector,
        "reason": reason,
        "primary_score": primary_score,
        "scores": scores,
    }


def build_skip_result(
    folder_path: Path,
    cover: Optional[Path],
    policy: str,
    reason: str,
) -> AnalysisResult:
    return AnalysisResult(
        folder=folder_path.name,
        cover=str(cover) if cover else None,
        policy=policy,
        detector="none",
        status="skip",
        flagged=False,
        moved_to=None,
        reason=reason,
        primary_score=None,
        scores={},
    )


def process_folder(folder_path: Path, args, state: dict[str, Any]) -> AnalysisResult:
    cover = find_cover_image(folder_path, state["valid_exts"])
    if not cover:
        return build_skip_result(folder_path, None, args.politica, "Sin portada")

    try:
        result = scan_cover_path(
            image_path=cover,
            policy=args.politica,
            backend=state.get("cadaver_backend", args.backend),
            api_key=state.get("api_key"),
            api_key_env=args.api_key_env,
            moderation_model=args.modelo_moderacion,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            ollama_api_key=state.get("ollama_api_key"),
            timeout=args.timeout,
            thresholds=state["cadaver_thresholds"],
            nsfw_model=state.get("nsfw_model"),
            nsfw_threshold=args.umbral,
        )
    except UnidentifiedImageError:
        return build_skip_result(folder_path, cover, args.politica, "Portada inválida")
    except Exception as exc:
        return AnalysisResult(
            folder=folder_path.name,
            cover=str(cover),
            policy=args.politica,
            detector=args.politica,
            status="error",
            flagged=False,
            moved_to=None,
            reason=f"Error evaluando portada: {exc}",
            primary_score=None,
            scores={},
        )

    if result["status"] != "flagged":
        return AnalysisResult(
            folder=folder_path.name,
            cover=str(cover),
            policy=args.politica,
            detector=result["detector"],
            status=result["status"],
            flagged=False,
            moved_to=None,
            reason=result["reason"],
            primary_score=result["primary_score"],
            scores=result["scores"],
        )

    destination = move_to_destination(
        folder_path=folder_path,
        destination_root=state["destination_root"],
        reason=result["reason"],
        dry_run=args.dry_run,
    )
    return AnalysisResult(
        folder=folder_path.name,
        cover=str(cover),
        policy=args.politica,
        detector=result["detector"],
        status="flagged",
        flagged=True,
        moved_to=str(destination) if destination else None,
        reason=result["reason"],
        primary_score=result["primary_score"],
        scores=result["scores"],
    )


def resolve_destination_root(args) -> Path:
    if args.destino:
        return Path(args.destino)
    if args.politica == "cadaver-real":
        return DEFAULT_REVIEW_DIR
    return LEGACY_CENSORED_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Filtra portadas riesgosas y mueve carpetas a revisión."
    )
    parser.add_argument(
        "--politica",
        choices=("cadaver-real", "nsfw"),
        default=os.environ.get("PORTADA_POLICY", "cadaver-real"),
        help="Regla de filtrado a aplicar.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ollama", "openai"),
        default=os.environ.get("CADAVER_BACKEND", "auto"),
        help="Backend para politica cadaver-real. `auto` prioriza Ollama local.",
    )
    parser.add_argument(
        "--base",
        help="Carpeta base a escanear (default: DIR_AUDIO_SCRIPTS).",
    )
    parser.add_argument(
        "--destino",
        help="Carpeta destino para carpetas marcadas.",
    )
    parser.add_argument(
        "--ext",
        default="png,jpg,jpeg",
        help="Extensiones de imagen separadas por coma.",
    )
    parser.add_argument(
        "--limite",
        type=int,
        help="Número máximo de carpetas a revisar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No mueve carpetas; solo reporta qué haría.",
    )
    parser.add_argument(
        "--registro",
        default=str(DEFAULT_LOG_PATH),
        help="Ruta JSONL donde guardar resultados.",
    )

    parser.add_argument(
        "--api-key-env",
        default=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Variable de entorno para fallback OpenAI.",
    )
    parser.add_argument(
        "--modelo-moderacion",
        default=DEFAULT_MODERATION_MODEL,
        help="Modelo de moderación para backend OpenAI.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Host/base URL de Ollama para backend local.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Modelo vision en Ollama para backend local.",
    )
    parser.add_argument(
        "--ollama-api-key-env",
        default=os.environ.get("OLLAMA_API_KEY_ENV", "OLLAMA_API_KEY"),
        help="Variable de entorno para Authorization Bearer en Ollama si aplica.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PORTADA_FILTER_TIMEOUT", "60")),
        help="Timeout HTTP en segundos para moderación.",
    )
    parser.add_argument(
        "--umbral-grafico",
        type=float,
        default=parse_float_env("CADAVER_GRAPHIC_THRESHOLD", 0.14),
        help="Mínimo de violence/graphic para marcar sospecha.",
    )
    parser.add_argument(
        "--umbral-foto",
        type=float,
        default=parse_float_env("CADAVER_PHOTO_THRESHOLD", 0.58),
        help="Mínimo de score fotográfico para asumir foto realista.",
    )
    parser.add_argument(
        "--umbral-cadaver",
        type=float,
        default=parse_float_env("CADAVER_COMBINED_THRESHOLD", 0.12),
        help="Umbral combinado final para cadaver-real.",
    )

    parser.add_argument(
        "--modelo",
        help="Ruta al modelo NSFW (.onnx recomendado; también .h5/.SavedModel).",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=parse_float_env("NSFW_THRESHOLD", 0.35),
        help="Umbral NSFW para politica nsfw.",
    )
    parser.add_argument(
        "--auto-descargar",
        action="store_true",
        help="Descarga automática del modelo NudeNet si aplica.",
    )
    parser.add_argument(
        "--modelo-url",
        default=DEFAULT_MODEL_URL,
        help="URL del modelo ONNX de fallback para politica nsfw.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base) if args.base else Path(DIR_AUDIO_SCRIPTS)
    if not base_dir.exists():
        log(f"No existe la carpeta base: {base_dir}")
        return

    state: dict[str, Any] = {
        "valid_exts": parse_exts(args.ext),
        "destination_root": resolve_destination_root(args),
        "nsfw_model": None,
        "api_key": None,
        "ollama_api_key": None,
        "cadaver_backend": args.backend,
        "cadaver_thresholds": CadaverThresholds(
            graphic_threshold=args.umbral_grafico,
            photo_threshold=args.umbral_foto,
            combined_threshold=args.umbral_cadaver,
        ),
    }

    if args.politica == "cadaver-real":
        state["api_key"] = load_openai_api_key(args.api_key_env)
        state["ollama_api_key"] = load_ollama_api_key(args.ollama_api_key_env)
        if args.backend == "ollama":
            if not ollama_is_available(
                args.ollama_host,
                timeout=args.timeout,
                api_key=state["ollama_api_key"],
            ):
                log(
                    f"No se detectó Ollama en {normalize_base_url(args.ollama_host)}. "
                    "Levanta el servicio o usa otro backend."
                )
                return
            state["cadaver_backend"] = "ollama"
            log("Backend cadaver-real seleccionado: ollama (local, sin OPENAI_API_KEY).")
        elif args.backend == "openai" and not state["api_key"]:
            log(
                f"Falta la variable {args.api_key_env}. "
                "Defínela o usa --backend ollama."
            )
            return
        elif args.backend == "auto":
            resolved_backend = resolve_cadaver_backend(
                backend="auto",
                ollama_host=args.ollama_host,
                timeout=args.timeout,
                api_key=state["api_key"],
                api_key_env=args.api_key_env,
                ollama_api_key=state["ollama_api_key"],
            )
            if not resolved_backend:
                log(
                    "No hay backend disponible para cadaver-real. "
                    f"Inicia Ollama en {normalize_base_url(args.ollama_host)}. "
                    f"{args.api_key_env} es solo fallback opcional."
                )
                return
            state["cadaver_backend"] = resolved_backend
            if resolved_backend == "ollama":
                log(
                    "Backend cadaver-real seleccionado: ollama "
                    "(local, sin OPENAI_API_KEY)."
                )
            else:
                log("Backend cadaver-real seleccionado: openai (fallback remoto).")
        else:
            state["cadaver_backend"] = args.backend
    else:
        nsfw_model = build_nsfw_model(args)
        if nsfw_model is None:
            return
        state["nsfw_model"] = nsfw_model

    folders = list_folders(base_dir, args.limite)
    if not folders:
        log(f"No hay carpetas en {base_dir}")
        return

    log(
        f"Analizando {len(folders)} carpetas en {base_dir} "
        f"con politica {args.politica}..."
    )

    counts = {
        "ok": 0,
        "flagged": 0,
        "skip": 0,
        "error": 0,
    }
    log_path = Path(args.registro)

    for folder in folders:
        result = process_folder(folder, args, state)
        counts[result.status] = counts.get(result.status, 0) + 1
        append_result_log(log_path, result)

        if result.status == "ok":
            score_text = (
                f"{result.primary_score:.3f}" if result.primary_score is not None else "n/a"
            )
            log(f"[OK] {result.folder} score {score_text} | {result.reason}")
        elif result.status == "skip":
            log(f"[SKIP] {result.folder} | {result.reason}")
        elif result.status == "error":
            log(f"[ERROR] {result.folder} | {result.reason}")

    log(
        "Resumen: "
        f"ok={counts['ok']} "
        f"flagged={counts['flagged']} "
        f"skip={counts['skip']} "
        f"error={counts['error']}"
    )
    log(f"Registro guardado en {log_path}")


if __name__ == "__main__":
    main()
