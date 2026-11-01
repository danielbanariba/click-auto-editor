import json
import shutil
import subprocess
import tempfile
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


if __name__ == "__main__":
    unittest.main()
