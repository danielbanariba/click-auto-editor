import json
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pipeline_agente as pa

TZ = timezone(timedelta(hours=-6))
HAY_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def momento(texto):
    return datetime.fromisoformat(texto).replace(tzinfo=TZ)


class TestHorarios(unittest.TestCase):
    def test_salta_horas_ocupadas_y_nunca_devuelve_una_hora_pasada(self):
        ocupadas = pa.horas_ocupadas([momento("2026-11-01 10:15"), momento("2026-11-01 11:00")])
        hora = pa.siguiente_hora_libre(momento("2026-11-01 09:30"), ocupadas)
        self.assertEqual(hora, momento("2026-11-01 12:00"))

    def test_dia_con_24_videos_pasa_al_dia_siguiente(self):
        dia = momento("2026-11-01 00:00")
        ocupadas = pa.horas_ocupadas([dia + timedelta(hours=h) for h in range(24)])
        hora = pa.siguiente_hora_libre(dia, ocupadas)
        self.assertEqual(hora, momento("2026-11-02 00:00"))

    def test_con_reclamo_se_programa_dos_anios_despues(self):
        ahora = momento("2026-11-01 13:20")
        self.assertEqual(pa.elegir_fecha(ahora, True, set()), momento("2028-11-01 14:00"))
        self.assertEqual(pa.elegir_fecha(ahora, False, set()), momento("2026-11-01 14:00"))

    def test_dos_anios_desde_29_de_febrero(self):
        self.assertEqual(
            pa.sumar_dos_anios(momento("2028-02-29 10:00")), momento("2030-02-28 10:00")
        )


class TestStatusProgramado(unittest.TestCase):
    def test_conserva_campos_editables_y_descarta_los_de_solo_lectura(self):
        actual = {
            "uploadStatus": "processed",
            "privacyStatus": "private",
            "license": "youtube",
            "embeddable": True,
            "publicStatsViewable": False,
            "madeForKids": False,
        }
        publica = momento("2026-11-02 08:00")
        self.assertEqual(
            pa.armar_status_programado(actual, publica),
            {
                "license": "youtube",
                "embeddable": True,
                "publicStatsViewable": False,
                "privacyStatus": "private",
                "publishAt": "2026-11-02T08:00:00-06:00",
            },
        )


class TestClasificarReclamo(unittest.TestCase):
    def test_pagina_sin_senal_clara_nunca_cuenta_como_sin_reclamo(self):
        self.assertEqual(pa.clasificar_reclamo(0, "Cargando...", True), "desconocido")
        self.assertEqual(
            pa.clasificar_reclamo(0, "No se encontraron problemas", False), "desconocido"
        )
        self.assertEqual(pa.clasificar_reclamo(2, "", True), "con_reclamo")
        self.assertEqual(
            pa.clasificar_reclamo(0, "No se encontraron problemas", True), "sin_reclamo"
        )


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def crear_album(carpeta, pista_s, video_s, audio="tono", tamanio="320x180"):
    carpeta.mkdir(parents=True)
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:duration={pista_s}", str(carpeta / "01. pista.flac"))
    entradas = ["-f", "lavfi", "-i", f"color=c=blue:s={tamanio}:d={video_s}"]
    if audio == "tono":
        entradas += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={video_s}"]
    elif audio == "mudo":
        entradas += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={video_s}"]
    ffmpeg(*entradas, "-shortest", "-c:v", "libx264", "-c:a", "aac", str(carpeta / f"{carpeta.name}.mp4"))
    return carpeta


@unittest.skipUnless(HAY_FFMPEG, "requiere ffmpeg y ffprobe")
class TestVerificar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.parches = [mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas")]
        for parche in self.parches:
            parche.start()

    def tearDown(self):
        for parche in self.parches:
            parche.stop()
        shutil.rmtree(self.tmp)

    def fallidos(self, resultado):
        return {c["nombre"] for c in resultado["checks"] if not c["ok"]}

    def test_video_correcto_pasa_y_genera_captura(self):
        carpeta = crear_album(self.tmp / "Banda - Album", pista_s=4, video_s=5)
        resultado = pa.verificar(carpeta, ancho=320, alto=180, intro=1.0)
        self.assertTrue(resultado["ok"], resultado["resumen"])
        self.assertEqual(self.fallidos(resultado), set())
        self.assertTrue(Path(resultado["captura"]).exists())
        self.assertIn("Banda - Album", resultado["resumen"])

    def test_video_mudo_falla(self):
        carpeta = crear_album(self.tmp / "Mudo", pista_s=4, video_s=5, audio="mudo")
        resultado = pa.verificar(carpeta, ancho=320, alto=180, intro=1.0)
        self.assertFalse(resultado["ok"])
        self.assertEqual(self.fallidos(resultado), {"no_silencio"})

    def test_video_sin_pista_de_audio_falla(self):
        carpeta = crear_album(self.tmp / "SinAudio", pista_s=4, video_s=5, audio=None)
        resultado = pa.verificar(carpeta, ancho=320, alto=180, intro=1.0)
        self.assertFalse(resultado["ok"])
        self.assertIn("tiene_audio", self.fallidos(resultado))

    def test_duracion_y_resolucion_distintas_fallan(self):
        carpeta = crear_album(self.tmp / "Corto", pista_s=4, video_s=10)
        resultado = pa.verificar(carpeta, ancho=3840, alto=2160, intro=1.0)
        self.assertFalse(resultado["ok"])
        self.assertEqual(self.fallidos(resultado), {"duracion", "resolucion"})

    def test_captura_sale_de_la_mitad_del_video(self):
        from PIL import Image

        carpeta = self.tmp / "Largo"
        carpeta.mkdir()
        tramos = "".join(
            f"color=c={color}:s=320x180:r=5:d={segundos}[{nombre}];"
            for nombre, color, segundos in (("a", "red", 90), ("b", "lime", 20), ("c", "blue", 90))
        )
        ffmpeg(
            "-filter_complex", f"{tramos}[a][b][c]concat=n=3:v=1:a=0[v]",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=200",
            "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", str(carpeta / "Largo.mp4"),
        )
        resultado = pa.verificar(carpeta, ancho=320, alto=180, intro=1.0)
        rojo, verde, azul = Image.open(resultado["captura"]).convert("RGB").getpixel((640, 360))
        self.assertGreater(verde, 150, (rojo, verde, azul))
        self.assertLess(rojo, 100, (rojo, verde, azul))


class TestRetomar(unittest.TestCase):
    """Una sesión nueva retoma lo que dejó otra: renders en curso y álbumes sin aprobar."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Un nombre propio, para no confundirse con un render real de la máquina.
        self.proceso = f"render_falso_{self.tmp.name}.py"
        self.parches = [
            mock.patch.object(pa, "PROCESO_RENDER", self.proceso),
            mock.patch.object(pa, "DIR_RENDERIZANDO", self.tmp / "Renderizando"),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
        ]
        for parche in self.parches:
            parche.start()
        self.procesos = []

    def tearDown(self):
        for proceso in self.procesos:
            proceso.kill()
            proceso.wait()
        for parche in self.parches:
            parche.stop()
        shutil.rmtree(self.tmp)

    def lanzar_render_falso(self, segundos):
        proceso = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({segundos})", str(self.tmp / self.proceso)]
        )
        self.procesos.append(proceso)
        return proceso

    def test_estado_lista_los_albumes_renderizados_que_esperan_aprobacion(self):
        listo = self.tmp / "Renderizando" / "Banda - Listo"
        listo.mkdir(parents=True)
        (listo / "Banda - Listo.mp4").write_bytes(b"video")
        (self.tmp / "Renderizando" / "Banda - Sin video").mkdir()

        resultado = pa.estado()

        self.assertEqual(resultado["por_aprobar"], [str(listo)])
        self.assertFalse(resultado["render_en_curso"])

    def test_renderizar_no_lanza_otro_render_si_ya_hay_uno(self):
        self.lanzar_render_falso(30)
        with mock.patch.object(pa.subprocess, "run", side_effect=AssertionError("lanzó un render")):
            resultado = pa.renderizar()
        self.assertEqual(resultado, {"ok": False, "motivo": "render_en_curso"})

    def test_un_editor_con_el_script_abierto_no_cuenta_como_render(self):
        editor = subprocess.Popen(["/bin/sh", "-c", "sleep 30; true", str(self.tmp / self.proceso)])
        self.procesos.append(editor)
        self.assertFalse(pa.estado()["render_en_curso"])

    def test_estado_espera_a_que_termine_el_render_de_otra_sesion(self):
        self.lanzar_render_falso(1.5)
        self.assertTrue(pa.estado()["render_en_curso"])

        inicio = time.monotonic()
        resultado = pa.estado(esperar_render=True, intervalo=0.1)

        self.assertFalse(resultado["render_en_curso"])
        self.assertGreater(time.monotonic() - inicio, 1.0)


class TestSubir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raiz = self.tmp / "Renderizando"
        self.carpeta = self.raiz / "Banda - Album"
        self.carpeta.mkdir(parents=True)
        (self.carpeta / "Banda - Album.mp4").write_bytes(b"video")
        (self.carpeta / "01. pista.flac").write_bytes(b"audio")
        self.parches = [
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
        ]
        for parche in self.parches:
            parche.start()
        self.subidas = []

    def tearDown(self):
        for parche in self.parches:
            parche.stop()
        shutil.rmtree(self.tmp)

    def uploader(self, subir_video):
        class QuotaExceededError(RuntimeError):
            pass

        class UploadLimitExceededError(RuntimeError):
            pass

        def borrar(carpeta, raiz):
            self.assertEqual(Path(raiz), self.raiz.resolve())
            shutil.rmtree(carpeta)
            return True

        return SimpleNamespace(
            QuotaExceededError=QuotaExceededError,
            UploadLimitExceededError=UploadLimitExceededError,
            cargar_env=lambda: None,
            probar_credenciales_disponibles=lambda prefix: ("yt", [], [], []),
            encontrar_video=lambda carpeta: next(carpeta.glob("*.mp4"), None),
            cargar_generos=lambda: ([], {}),
            cargar_repertorio=lambda: {},
            procesar_carpeta=lambda *a: {"title": "Banda - Album", "description": "D"},
            subir_video=subir_video,
            authenticate_next=lambda prefix: None,
            is_invalid_grant_error=lambda exc: False,
            safe_delete_folder=borrar,
        )

    def test_sube_privado_sin_fecha_registra_pendiente_y_borra_la_carpeta(self):
        def subir_video(youtube, video, titulo, descripcion, **kwargs):
            self.subidas.append(kwargs)
            return {"id": "abc123"}

        resultado = pa.subir(
            self.carpeta, uploader=self.uploader(subir_video), sin_deathgrind=True, raices=(self.raiz,)
        )
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(self.subidas[0]["privacy"], "private")
        self.assertIsNone(self.subidas[0]["publish_at"])
        self.assertFalse(self.carpeta.exists())
        pendientes = json.loads(pa.PENDIENTES_JSON.read_text(encoding="utf-8"))
        self.assertEqual([p["video_id"] for p in pendientes], ["abc123"])

    def test_limite_del_canal_conserva_la_carpeta(self):
        uploader = None

        def subir_video(*args, **kwargs):
            raise uploader.UploadLimitExceededError("limite")

        uploader = self.uploader(subir_video)
        resultado = pa.subir(self.carpeta, uploader=uploader, sin_deathgrind=True, raices=(self.raiz,))
        self.assertEqual(resultado["motivo"], "limite_canal")
        self.assertTrue(self.carpeta.exists())
        self.assertFalse(pa.PENDIENTES_JSON.exists())

    def test_carpeta_fuera_de_las_raices_no_se_sube_ni_se_borra(self):
        afuera = self.tmp / "otra" / "Banda - Album"
        shutil.copytree(self.carpeta, afuera)

        def subir_video(*args, **kwargs):
            self.subidas.append(kwargs)
            return {"id": "x"}

        resultado = pa.subir(afuera, uploader=self.uploader(subir_video), sin_deathgrind=True, raices=(self.raiz,))
        self.assertEqual(resultado["motivo"], "carpeta_fuera_de_raiz")
        self.assertEqual(self.subidas, [])
        self.assertTrue(afuera.exists())


class Llamada:
    def __init__(self, resultado):
        self.resultado = resultado

    def execute(self):
        return self.resultado


class VideosFalsos:
    def __init__(self, estados, demora_actualizacion):
        self.estados = list(estados)
        self.demora_actualizacion = demora_actualizacion
        self.actualizaciones = []

    def list(self, part, id):
        estado = self.estados.pop(0) if len(self.estados) > 1 else self.estados[0]
        status = {
            "privacyStatus": "private",
            "uploadStatus": "processed" if estado == "succeeded" else "uploaded",
            "embeddable": True,
        }
        return Llamada({"items": [{"status": status, "processingDetails": {"processingStatus": estado}}]})

    def update(self, part, body):
        time.sleep(self.demora_actualizacion)
        self.actualizaciones.append(body)
        return Llamada(body)


class YouTubeFalso:
    def __init__(self, estados, demora_actualizacion=0.0):
        self.videos_falsos = VideosFalsos(estados, demora_actualizacion)

    def videos(self):
        return self.videos_falsos


def uploader_para_programar(youtube):
    return SimpleNamespace(
        cargar_env=lambda: None,
        probar_credenciales_disponibles=lambda prefix: (youtube, [], [], []),
        fetch_scheduled_publish_times=lambda cliente, max_items: (set(), {}),
        is_quota_error=lambda exc: False,
        is_invalid_grant_error=lambda exc: False,
        authenticate_next=lambda prefix: None,
    )


class TestProgramar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.parches = [
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "PROGRAMADOS_JSONL", self.tmp / "programados.jsonl"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
        ]
        for parche in self.parches:
            parche.start()

    def tearDown(self):
        for parche in self.parches:
            parche.stop()
        shutil.rmtree(self.tmp)

    def test_margen_de_una_hora_cuenta_desde_que_termina_el_procesado(self):
        subido = datetime.now().astimezone() - timedelta(hours=3)
        pa.registrar_pendiente("V1", "Banda - Album", "Banda - Album", subido)
        youtube = YouTubeFalso(["processing", "succeeded"])
        esperas = []
        with mock.patch.object(pa.time, "sleep", esperas.append):
            resultado = pa.programar(
                "V1",
                esperar=True,
                revisar=lambda video_id: {"estado": "sin_reclamo", "reclamantes": []},
                uploader=uploader_para_programar(youtube),
            )
        self.assertTrue(resultado["ok"], resultado)
        self.assertGreaterEqual(max(esperas), 59 * 60)
        status = youtube.videos_falsos.actualizaciones[0]["status"]
        self.assertEqual(status["privacyStatus"], "private")
        self.assertEqual(status["publishAt"], resultado["publica_en"])
        self.assertEqual(pa.leer_pendientes(), [])

    def test_dos_programar_a_la_vez_no_eligen_la_misma_hora(self):
        contexto = multiprocessing.get_context("fork")
        resultados = contexto.Queue()

        def correr(video_id):
            # La demora abre la ventana entre elegir la hora y registrarla.
            uploader = uploader_para_programar(YouTubeFalso(["succeeded"], demora_actualizacion=0.5))
            resultados.put(pa.programar(video_id, reclamo=False, uploader=uploader)["publica_en"])

        procesos = [contexto.Process(target=correr, args=(v,)) for v in ("V1", "V2")]
        for proceso in procesos:
            proceso.start()
        for proceso in procesos:
            proceso.join(timeout=30)
        horas = {resultados.get(timeout=5), resultados.get(timeout=5)}
        self.assertEqual(len(horas), 2, horas)


if __name__ == "__main__":
    unittest.main()
