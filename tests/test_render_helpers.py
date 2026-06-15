"""
Tests de los helpers de optimización de "2. ffmpeg_render.py".

Ejecutar: pytest tests/test_render_helpers.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "2. ffmpeg_render.py"


@pytest.fixture(scope="module")
def render_mod():
    spec = importlib.util.spec_from_file_location("ffmpeg_render", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ffmpeg_render"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wav_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "tono.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostats",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(path),
        ],
        check=True,
    )
    return path


def test_get_audio_info_duracion_y_bitrate(render_mod, wav_file):
    duration, bitrate = render_mod.get_audio_info(wav_file)
    assert 1.9 <= duration <= 2.1
    assert bitrate > 0


def test_get_audio_info_usa_cache(render_mod, wav_file):
    primera = render_mod.get_audio_info(wav_file)
    # Si el archivo desaparece, el caché debe seguir respondiendo igual.
    wav_file.rename(wav_file.with_suffix(".movido"))
    try:
        assert render_mod.get_audio_info(wav_file) == primera
    finally:
        wav_file.with_suffix(".movido").rename(wav_file)


def test_wrappers_de_duracion_y_bitrate(render_mod, wav_file):
    duration, bitrate = render_mod.get_audio_info(wav_file)
    assert render_mod.get_audio_duration(wav_file) == duration
    assert render_mod.get_audio_bitrate_kbps(wav_file) == bitrate


def test_pick_output_audio_bitrate_respeta_tope(render_mod, wav_file):
    # WAV PCM reporta bitrate altísimo; el tope de salida es 384k.
    assert render_mod.pick_output_audio_bitrate([wav_file]) == "384k"


def test_deathgrind_cache_key_normaliza(render_mod):
    key1 = render_mod._deathgrind_cache_key("Banda Ñu", "Álbum (2020)")
    key2 = render_mod._deathgrind_cache_key("BANDA ñu", "álbum")
    assert "|" in key1
    # Mayúsculas y el año entre paréntesis se descartan; acentos se preservan
    # (mismo criterio que normalize_name usa para el matching con la API).
    assert key1 == key2


def test_render_job_defaults(render_mod):
    job = render_mod.RenderJob(
        original_folder_path=Path("/x"),
        folder_name="x",
        folder_path=Path("/x"),
    )
    assert job.error is None
    assert job.stop_result is None
    assert job.prep_seconds == 0.0


def test_has_enough_disk_space_descuenta_pendientes(render_mod, monkeypatch):
    monkeypatch.setattr(render_mod, "get_free_space_gb", lambda path: 15.0)
    monkeypatch.setattr(render_mod, "PENDING_FINALIZE_BYTES", 0)
    assert render_mod.has_enough_disk_space(10.0) is True
    # 10 GB en cola de finalización: 15 libres ya no alcanzan para mínimo 10.
    monkeypatch.setattr(render_mod, "PENDING_FINALIZE_BYTES", 10 * 1024**3)
    assert render_mod.has_enough_disk_space(10.0) is False


def test_check_finalize_failures_detecta_no_space(render_mod, monkeypatch):
    class FuturoFallido:
        def done(self):
            return True

        def exception(self):
            return OSError(28, "No space left on device")

    monkeypatch.setattr(
        render_mod, "FINALIZE_FUTURES", [("Carpeta X", FuturoFallido())]
    )
    assert render_mod.check_finalize_failures() == "no_space"
    assert render_mod.FINALIZE_FUTURES == []
