"""
Tests de los helpers de optimización de "2. ffmpeg_render.py".

Ejecutar: pytest tests/test_render_helpers.py
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

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
    monkeypatch.setattr(render_mod, "RENDER_SSD_DIR", Path("/"))
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


def test_sin_el_ssd_montado_no_hay_espacio_para_renderizar(render_mod, monkeypatch, tmp_path):
    # Sin el SSD, Renderizando/ caería en el disco del sistema.
    monkeypatch.setattr(render_mod, "RENDER_SSD_DIR", tmp_path)
    monkeypatch.setattr(render_mod, "get_free_space_gb", lambda path: 5000.0)
    assert render_mod.has_enough_disk_space(10.0) is False


@pytest.fixture
def album_renderizado(render_mod, monkeypatch, tmp_path):
    """Álbum en la cola, con su video final todavía en la carpeta temporal."""
    cola = tmp_path / "cola" / "Banda - Disco"
    cola.mkdir(parents=True)
    (cola / "01 - Tema.mp3").write_bytes(b"audio")
    (cola / "cover.jpg").write_bytes(b"jpg")
    temporal = tmp_path / "temp" / "Banda - Disco"
    temporal.mkdir(parents=True)
    video = temporal / "Banda - Disco.mp4"
    video.write_bytes(b"video")
    renderizando = tmp_path / "ssd" / "Renderizando"
    monkeypatch.setattr(render_mod, "DIR_RENDERIZANDO", renderizando)
    monkeypatch.setattr(render_mod, "SSD_TEMP_DIR", tmp_path / "temp")
    return cola, temporal, video, renderizando


def finalizar(render_mod, cola, temporal, video):
    render_mod.finalize_render_output(
        final_video=video,
        temp_folder_path=temporal,
        track_overlays_path=None,
        original_folder_path=cola,
        folder_name=cola.name,
        show_progress=False,
        start_time=0.0,
    )


def test_finalize_lleva_el_video_directo_a_renderizando(render_mod, monkeypatch, album_renderizado):
    cola, temporal, video, renderizando = album_renderizado
    movidos = []
    mover = render_mod.shutil.move

    def espiar(origen, destino, *args, **kwargs):
        movidos.append((str(origen), str(destino)))
        return mover(origen, destino, *args, **kwargs)

    monkeypatch.setattr(render_mod.shutil, "move", espiar)

    finalizar(render_mod, cola, temporal, video)

    destino = renderizando / "Banda - Disco"
    assert sorted(p.name for p in destino.iterdir()) == ["01 - Tema.mp3", "Banda - Disco.mp4", "cover.jpg"]
    assert not cola.exists()
    assert not temporal.exists()
    # El video nunca se escribe en el disco de la cola (el NTFS lento).
    assert [d for o, d in movidos if o == str(video)] == [str(destino / video.name)]


@pytest.mark.parametrize("falla", ["video", "carpeta"])
def test_si_falla_pasar_al_ssd_el_album_vuelve_a_la_cola_y_el_video_se_rescata(
    render_mod, monkeypatch, album_renderizado, falla
):
    cola, temporal, video, renderizando = album_renderizado
    origen_que_falla = str(video) if falla == "video" else str(cola)
    mover = render_mod.shutil.move

    def falla_al_pasar_al_ssd(origen, destino, *args, **kwargs):
        if str(origen) == origen_que_falla and str(renderizando) in str(destino):
            raise OSError("sin espacio")
        return mover(origen, destino, *args, **kwargs)

    monkeypatch.setattr(render_mod.shutil, "move", falla_al_pasar_al_ssd)

    with pytest.raises(OSError):
        finalizar(render_mod, cola, temporal, video)

    assert sorted(p.name for p in cola.iterdir()) == ["01 - Tema.mp3", "cover.jpg"]
    assert not (renderizando / "Banda - Disco").exists()
    assert [p.name for p in (temporal.parent / "_rescate").iterdir()] == ["Banda - Disco.mp4"]


@pytest.fixture
def render_con_cpp_que_falla(render_mod, monkeypatch, tmp_path, wav_file):
    """Renderiza en modo prueba (el del servicio) con un vhs_render que falla con `mensaje`."""

    def renderizar(mensaje):
        binario = tmp_path / "vhs_render"
        binario.write_text(f"#!/bin/sh\necho '[Config] GPU: enabled'\necho '{mensaje}' >&2\nexit 1\n")
        binario.chmod(0o755)
        monkeypatch.setattr(render_mod, "VHS_CPP_BIN", binario)
        monkeypatch.setattr(render_mod, "USE_CPP_VHS", True)
        carpeta = tmp_path / "Banda - Disco"
        carpeta.mkdir()
        job = render_mod.RenderJob(
            original_folder_path=carpeta,
            folder_name=carpeta.name,
            folder_path=carpeta,
            audio_files=[wav_file],
            cover_main=tmp_path / "cover.jpg",
            audio_duration=2.0,
            total_duration=9.0,
        )
        return render_mod.render_video(carpeta, carpeta.name, show_progress=True, prepared_job=job)

    return renderizar


def test_el_render_usa_la_portada_que_muestra_el_servicio_y_no_la_propaganda_de_deathgrind(
    render_mod, monkeypatch, tmp_path, wav_file
):
    carpeta = tmp_path / "Banda - Disco (2018) [EP]"
    carpeta.mkdir()
    shutil.copy(wav_file, carpeta / "01 Tema.wav")
    for nombre, color in (("DGC - PURE HATRED.jpg", "black"), ("Cover_2018.jpg", "red")):
        Image.new("RGB", (32, 32), color).save(carpeta / nombre)
    # El render de Marcio Dentro (2026-11-09) recibió la propaganda primero del sistema de archivos.
    listar = os.listdir
    monkeypatch.setattr(render_mod.os, "listdir", lambda ruta: sorted(listar(ruta), reverse=True))
    monkeypatch.setattr(render_mod, "USE_SSD_TEMP", False)
    monkeypatch.setattr(render_mod, "USE_RAMDISK", False)
    monkeypatch.setattr(render_mod, "USE_CPP_VHS", False)
    monkeypatch.setattr(render_mod, "ensure_shadow_cover", lambda portada, *args: portada)

    job = render_mod.prepare_render_job(carpeta, carpeta.name)

    assert job.error is None
    assert job.cover_main == carpeta / "Cover_2018.jpg"


def test_sin_memoria_en_la_gpu_el_album_no_va_a_error404_y_el_lote_se_frena(
    render_mod, render_con_cpp_que_falla, capsys
):
    # Lo que imprime vhs_render cuando otro programa ocupa la VRAM (visto el 2026-11-03).
    resultado = render_con_cpp_que_falla(
        "CUDA error at cpp/src/pipeline/cover_pipeline.cpp:46: out of memory"
    )
    assert (resultado.success, resultado.reason) == (False, "gpu_sin_memoria")
    assert resultado.move_to_error is False
    assert resultado.stop_requested is True
    assert "la GPU no tiene memoria libre" in capsys.readouterr().out


def test_otra_falla_del_render_cpp_sigue_mandando_el_album_a_error404(
    render_mod, render_con_cpp_que_falla
):
    resultado = render_con_cpp_que_falla("[Error] Caught exception: invalid cover image")
    assert (resultado.success, resultado.reason) == (False, "cpp")
    assert resultado.move_to_error is True
    assert resultado.stop_requested is False


class _RespuestaApi:
    def __init__(self, status_code, datos=None):
        self.status_code = status_code
        self.datos = datos

    def json(self):
        return self.datos


class _SesionPorCursor:
    """Responde la página de /posts/search que pide el cursor.

    Pasado `limite` pedidos falla: un bucle sin guarda no cuelga la prueba, la pone en rojo.
    """

    def __init__(self, paginas, limite=10):
        self.paginas = paginas
        self.pedidas = []
        self.limite = limite

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.pedidas.append(params)
        if len(self.pedidas) > self.limite:
            raise AssertionError(f"más de {self.limite} pedidos: el bucle no corta")
        if url.endswith("/posts/search") and params.get("cursor") in self.paginas:
            return _RespuestaApi(200, self.paginas[params.get("cursor")])
        return _RespuestaApi(404)


def test_la_busqueda_del_post_sigue_el_cursor(render_mod):
    sesion = _SesionPorCursor({
        None: {"posts": [{"postId": 1, "bands": [{"name": "Otra"}], "album": "Disco"}],
               "hasMore": True, "nextCursor": "c1"},
        "c1": {"posts": [{"postId": 2, "bands": [{"name": "Banda"}], "album": "Disco"}],
               "hasMore": False, "nextCursor": None},
    })

    assert render_mod.buscar_post_id_en_api(sesion, "Banda", "Disco") == 2
    assert sesion.pedidas[1] == {"q": "Banda Disco", "cursor": "c1"}


def test_la_busqueda_del_post_corta_si_el_servidor_repite_el_cursor(render_mod):
    # Si no cortara, pediría a DeathGrind sin fin con cada álbum que se renderiza.
    trabada = {"posts": [{"postId": 1, "bands": [{"name": "Otra"}], "album": "Disco"}],
               "hasMore": True, "nextCursor": "c1"}
    sesion = _SesionPorCursor({None: trabada, "c1": trabada})

    assert render_mod.buscar_post_id_en_api(sesion, "Banda", "Disco") is None
    assert len(sesion.pedidas) == 2


def test_la_busqueda_del_post_para_en_el_tope_de_paginas_aunque_los_cursores_sean_nuevos(render_mod):
    otra = {"postId": 1, "bands": [{"name": "Otra"}], "album": "Disco"}
    cursores = [None] + [f"c{n}" for n in range(1, 9)]
    sesion = _SesionPorCursor({
        cursor: {"posts": [otra], "hasMore": True, "nextCursor": f"c{n}"}
        for n, cursor in enumerate(cursores, start=1)
    })

    assert render_mod.buscar_post_id_en_api(sesion, "Banda", "Disco") is None
    assert len(sesion.pedidas) == render_mod.PAGINAS_BUSQUEDA_POST
