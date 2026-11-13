"""
Tests del mux directo al SSD de "2. ffmpeg_render.py".

Nada de esto toca el ramdisk, el SSD ni la GPU reales: cada directorio se redirige a
`tmp_path`, el binario de la GPU es un script que escribe un video de 3 s con ffmpeg en CPU
y MemAvailable sale de una función que los tests reemplazan.

Ejecutar: pytest tests/test_render_ram.py
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


def ffmpeg_cpu(*args):
    subprocess.run(
        ["ffmpeg", "-y", "-nostats", "-loglevel", "error", *args], check=True
    )


@pytest.fixture(scope="module")
def wav_chico(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("audio") / "tono.wav"
    ffmpeg_cpu("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(ruta))
    return ruta


@pytest.fixture(scope="module")
def intro_chica(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("intro") / "intro.mp4"
    ffmpeg_cpu(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(ruta),
    )
    return ruta


# ---------------------------------------------------------------------------
# Mux directo al SSD
# ---------------------------------------------------------------------------

NOMBRE = "Banda - Disco"


@pytest.fixture
def render_cpp(render_mod, monkeypatch, tmp_path, wav_chico, intro_chica):
    """
    Corre render_video (modo del servicio, finalización síncrona) con vhs_render de mentira.

    `renderizar("ram")` trabaja en una carpeta del ramdisk de mentira y `renderizar("ssd")` en
    SSD_TEMP_DIR (el álbum que no entró en la RAM). Devuelve lo que pasó en el camino.
    """
    ram = tmp_path / "ram"
    ssd = tmp_path / "ssd"
    renderizando = ssd / "Renderizando"
    temp_ssd = ssd / "temp_render"
    ram.mkdir()
    for nombre, valor in (
        ("RAMDISK_PATH", ram), ("RAMDISK_MOUNTED", True), ("USE_RAMDISK", True),
        ("USE_SSD_TEMP", True), ("USE_CPP_VHS", True), ("RENDER_SSD_DIR", ssd),
        ("DIR_RENDERIZANDO", renderizando), ("SSD_TEMP_DIR", temp_ssd),
        ("INTRO_VIDEO", intro_chica), ("INTRO_DURATION", 1.0),
    ):
        monkeypatch.setattr(render_mod, nombre, valor)
    monkeypatch.setattr(render_mod, "resolve_gpu_usage", lambda: False)

    binario = tmp_path / "vhs_render"
    binario.write_text(
        '#!/bin/sh\n'
        'while [ $# -gt 0 ]; do [ "$1" = "--output" ] && salida="$2"; shift; done\n'
        'exec ffmpeg -y -nostats -loglevel error -f lavfi -i testsrc=duration=3:size=64x64:rate=5 '
        '-c:v libx264 -pix_fmt yuv420p "$salida"\n'
    )
    binario.chmod(0o755)
    monkeypatch.setattr(render_mod, "VHS_CPP_BIN", binario)

    visto = {"timings": None, "mux_salida": None, "al_finalizar": None}

    # Los tiempos se apendarían a data/render_times.jsonl, que está versionado.
    monkeypatch.setattr(
        render_mod, "log_render_timing", lambda carpeta, fases: visto.update(timings=dict(fases))
    )

    def hay_mp4_en_renderizando():
        return sorted(p.name for p in renderizando.rglob("*.mp4")) if renderizando.exists() else []

    def renderizar(donde, mux_falla=False):
        cola = tmp_path / "cola" / NOMBRE
        cola.mkdir(parents=True)
        trabajo = (ram if donde == "ram" else temp_ssd) / NOMBRE
        trabajo.mkdir(parents=True)
        for destino in (cola, trabajo):
            (destino / "01 - Tema.wav").write_bytes(wav_chico.read_bytes())
            (destino / "cover.jpg").write_bytes(b"jpg")
        job = render_mod.RenderJob(
            original_folder_path=cola,
            folder_name=NOMBRE,
            folder_path=trabajo,
            temp_folder_path=trabajo,
            audio_files=[trabajo / "01 - Tema.wav"],
            cover_main=trabajo / "cover.jpg",
            audio_duration=2.0,
            total_duration=3.0,
        )
        visto.update(trabajo=trabajo, cola=cola, renderizando=renderizando)

        mux_real = render_mod.run_ffmpeg_command

        def espiar_mux(cmd, *args, **kwargs):
            salida = Path(cmd[-1])
            visto["mux_salida"] = salida
            visto["mp4_en_renderizando_antes"] = hay_mp4_en_renderizando()
            if mux_falla:
                salida.write_bytes(b"video a medias")
                return False, "el mux se cortó"
            ok = mux_real(cmd, *args, **kwargs)
            visto["inodo"] = salida.stat().st_ino
            visto["mp4_en_renderizando_despues"] = hay_mp4_en_renderizando()
            return ok

        finalizar_real = render_mod.finalize_render_output

        def espiar_finalizar(**kwargs):
            visto["al_finalizar"] = {
                sufijo: (trabajo / f"{NOMBRE}{sufijo}").exists()
                for sufijo in ("__video.mp4", "__audio.m4a")
            }
            return finalizar_real(**kwargs)

        monkeypatch.setattr(render_mod, "run_ffmpeg_command", espiar_mux)
        monkeypatch.setattr(render_mod, "finalize_render_output", espiar_finalizar)
        visto["resultado"] = render_mod.normalize_render_result(
            render_mod.render_video(cola, NOMBRE, show_progress=False, prepared_job=job)
        )
        return visto

    return renderizar


@pytest.mark.parametrize("donde", ["ram", "ssd"])
def test_el_video_final_se_escribe_en_el_ssd_y_llega_a_renderizando_con_un_rename(
    render_mod, render_cpp, donde
):
    visto = render_cpp(donde)

    assert visto["resultado"].success
    salida = visto["mux_salida"]
    temp_ssd = render_mod.SSD_TEMP_DIR
    # Nunca en el ramdisk: el mux escribe en el mismo disco que Renderizando/.
    assert salida.is_relative_to(temp_ssd)
    # Mientras el mux escribe, la cola no tiene ningún .mp4 (estado, verificar y subir la leen).
    assert visto["mp4_en_renderizando_antes"] == []
    assert visto["mp4_en_renderizando_despues"] == []
    # El video entra de una vez: es el mismo archivo (mismo inodo), no una copia.
    final = visto["renderizando"] / NOMBRE / f"{NOMBRE}.mp4"
    assert final.stat().st_ino == visto["inodo"]
    assert not salida.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(final)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout) > 0
    assert not visto["trabajo"].exists()
    assert not visto["cola"].exists()


def test_al_terminar_el_mux_la_ram_se_libera_antes_de_la_finalizacion_en_segundo_plano(render_cpp):
    # La finalización puede tardar (copia el álbum del NTFS) mientras el siguiente álbum ya
    # prepara su copia: el video sin audio y la mezcla no pueden seguir ocupando la RAM.
    visto = render_cpp("ram")
    assert visto["al_finalizar"] == {"__video.mp4": False, "__audio.m4a": False}


def test_los_tiempos_por_fase_siguen_con_todos_sus_campos(render_cpp):
    visto = render_cpp("ram")
    assert {"prep_s", "gpu_s", "audio_wait_s", "mux_s", "move_s", "total_s"} <= set(visto["timings"])


def test_si_el_mux_falla_no_queda_un_video_a_medias_ni_en_el_ssd_ni_en_la_cola(render_cpp):
    visto = render_cpp("ram", mux_falla=True)

    assert (visto["resultado"].success, visto["resultado"].reason) == (False, "mux")
    assert not visto["mux_salida"].exists()
    assert not visto["trabajo"].exists()
    assert not (visto["renderizando"].exists() and list(visto["renderizando"].rglob("*")))
    # El álbum no se movió: sigue en su carpeta de origen para el próximo intento.
    assert visto["cola"].exists()


def test_si_falla_la_llegada_a_renderizando_el_video_del_ssd_se_rescata_y_el_album_vuelve_a_la_cola(
    render_mod, monkeypatch, render_cpp
):
    mover = render_mod.shutil.move

    def falla_la_llegada_del_video(origen, destino, *args, **kwargs):
        if str(origen).endswith(f"{NOMBRE}.mp4") and str(render_mod.DIR_RENDERIZANDO) in str(destino):
            raise OSError("sin espacio")
        return mover(origen, destino, *args, **kwargs)

    monkeypatch.setattr(render_mod.shutil, "move", falla_la_llegada_del_video)

    visto = render_cpp("ram")

    assert not visto["resultado"].success
    rescatados = list((render_mod.SSD_TEMP_DIR / render_mod.CARPETA_RESCATE_RENDER).iterdir())
    assert [p.name for p in rescatados] == [f"{NOMBRE}.mp4"]
    assert sorted(p.name for p in visto["cola"].iterdir()) == ["01 - Tema.wav", "cover.jpg"]
    assert not (visto["renderizando"] / NOMBRE).exists()


def test_la_caida_a_ffmpeg_sigue_dejando_el_video_en_la_cola(
    render_mod, monkeypatch, tmp_path, render_cpp
):
    # CUDA falla: el álbum se renderiza con FFmpeg dentro de su carpeta temporal y entra a la cola.
    monkeypatch.setattr(render_mod, "ALLOW_FFMPEG_FALLBACK", True)
    monkeypatch.setattr(render_mod, "CUDA_FAIL_FAST", False)
    monkeypatch.setattr(render_mod, "DISABLE_CPP_ON_CUDA", False)
    monkeypatch.setattr(render_mod, "CUDA_ERROR_FLAG", tmp_path / "flag_cuda")
    monkeypatch.setattr(render_mod, "ensure_vhs_noise_4k", lambda: tmp_path / "ruido.mp4")
    monkeypatch.setattr(render_mod, "precompute_static_background", lambda *args: None)
    render_mod.VHS_CPP_BIN.write_text(
        "#!/bin/sh\necho 'CUDA error: illegal memory access' >&2\nexit 1\n"
    )
    encodes = []
    run_real = render_mod.subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd[0] == "ffmpeg" and "-filter_complex" in cmd and "[outv]" in cmd:
            Path(cmd[-1]).write_bytes(b"video de ffmpeg")
            encodes.append(Path(cmd[-1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return run_real(cmd, *args, **kwargs)

    monkeypatch.setattr(render_mod.subprocess, "run", run)

    visto = render_cpp("ram")

    assert visto["resultado"].success
    assert encodes == [visto["trabajo"] / f"{NOMBRE}.mp4"]
    assert (visto["renderizando"] / NOMBRE / f"{NOMBRE}.mp4").read_bytes() == b"video de ffmpeg"
    assert not (render_mod.SSD_TEMP_DIR / render_mod.CARPETA_MUX_RENDER).exists()


def test_un_corte_a_mitad_del_mux_no_deja_un_video_parcial_en_el_ssd_al_limpiar(
    render_mod, monkeypatch, tmp_path
):
    # Ctrl+C durante el mux: el video parcial está en el SSD, no en el ramdisk que se vacía.
    ram = tmp_path / "ram"
    temp_ssd = tmp_path / "ssd" / "temp_render"
    parcial = temp_ssd / render_mod.CARPETA_MUX_RENDER / f"{NOMBRE}.mp4"
    rescatado = temp_ssd / render_mod.CARPETA_RESCATE_RENDER / "Otro.mp4"
    for archivo in (parcial, rescatado, ram / "Album" / "01.flac"):
        archivo.parent.mkdir(parents=True)
        archivo.write_bytes(b"x")
    monkeypatch.setattr(render_mod, "RAMDISK_PATH", ram)
    monkeypatch.setattr(render_mod, "RAMDISK_MOUNTED", True)
    monkeypatch.setattr(render_mod, "USE_RAMDISK", True)
    monkeypatch.setattr(render_mod, "SSD_TEMP_DIR", temp_ssd)

    render_mod.cleanup_temp()

    assert not parcial.exists()
    assert rescatado.exists()
    assert list(ram.iterdir()) == []


@pytest.mark.parametrize(
    "en_el_ssd, bytes_pendientes",
    [(True, 0), (False, 1234)],
    ids=["ya_esta_en_el_ssd", "esta_en_la_ram"],
)
def test_la_finalizacion_solo_cuenta_como_pendiente_el_video_que_aun_no_esta_en_el_ssd(
    render_mod, monkeypatch, tmp_path, en_el_ssd, bytes_pendientes
):
    # Un video que ya está en /mnt/Render ya está descontado del espacio libre: contarlo otra vez
    # frenaría el lote por espacio un video antes de tiempo.
    ssd = tmp_path / "ssd"
    video = (ssd if en_el_ssd else tmp_path / "ram") / "Banda - Disco.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x" * 1234)
    monkeypatch.setattr(render_mod, "RENDER_SSD_DIR", ssd)
    monkeypatch.setattr(render_mod, "FINALIZE_EXECUTOR", None)
    monkeypatch.setattr(render_mod, "FINALIZE_FUTURES", [])
    monkeypatch.setattr(render_mod, "PENDING_FINALIZE_BYTES", 0)
    contado = []
    monkeypatch.setattr(
        render_mod, "finalize_render_output",
        lambda **kwargs: contado.append(render_mod.PENDING_FINALIZE_BYTES),
    )

    render_mod.submit_finalize_job(
        final_video=video, temp_folder_path=video.parent, folder_name="Banda - Disco"
    )
    try:
        render_mod.FINALIZE_FUTURES[0][1].result(timeout=10)
    finally:
        render_mod.FINALIZE_EXECUTOR.shutdown(wait=True)

    assert contado == [bytes_pendientes]
