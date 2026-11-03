import contextlib
import http.client
import io
import json
import multiprocessing
import select
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httplib2
import playwright.sync_api as playwright_api
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build_from_document
from googleapiclient.errors import HttpError

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
        self.assertEqual(pa.elegir_fecha(ahora, True, []), momento("2028-11-01 14:00"))

    def test_con_reclamo_sigue_saltando_las_horas_ya_programadas(self):
        ahora = momento("2026-11-01 13:20")
        tomadas = [momento("2028-11-01 14:00"), momento("2028-11-01 15:30")]
        self.assertEqual(pa.elegir_fecha(ahora, True, tomadas), momento("2028-11-01 16:00"))

    def test_sin_reclamo_y_sin_videos_cerca_se_publica_en_el_acto(self):
        ahora = momento("2026-11-01 13:20")
        self.assertEqual(pa.elegir_fecha(ahora, False, []), ahora)
        # A una hora justa de otro video ya alcanza.
        lejanos = [ahora - timedelta(hours=1), ahora + timedelta(hours=1)]
        self.assertEqual(pa.elegir_fecha(ahora, False, lejanos), ahora)

    def test_sin_reclamo_con_un_video_a_menos_de_una_hora_espera_hasta_cumplir_la_hora(self):
        ahora = momento("2026-11-01 13:20")
        publicado_hace_20_min = ahora - timedelta(minutes=20)
        self.assertEqual(
            pa.elegir_fecha(ahora, False, [publicado_hace_20_min]), momento("2026-11-01 14:00")
        )
        programado_en_30_min = ahora + timedelta(minutes=30)
        self.assertEqual(
            pa.elegir_fecha(ahora, False, [programado_en_30_min]), momento("2026-11-01 14:50")
        )

    def test_sin_reclamo_encadena_videos_que_se_solapan_hasta_dejar_una_hora_libre(self):
        # Esperar a que pase el publicado deja el momento a menos de una hora del programado.
        ahora = momento("2026-11-01 13:20")
        tomados = [ahora - timedelta(minutes=10), ahora + timedelta(minutes=50)]
        self.assertEqual(pa.elegir_fecha(ahora, False, tomados), momento("2026-11-01 15:10"))

    def test_sin_reclamo_nunca_programa_con_menos_de_dos_minutos_de_anticipacion(self):
        ahora = momento("2026-11-01 13:20")
        casi_listo = ahora - timedelta(minutes=59, seconds=30)
        self.assertEqual(pa.elegir_fecha(ahora, False, [casi_listo]), momento("2026-11-01 13:22"))

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


# Texto real de Studio (cuenta en español de España) con la sesión ya iniciada.
PAGINA_SIN_RECLAMOS = (
    "Resumen de reclamaciones\n¿Qué ha ocurrido?\n"
    "No se han encontrado reclamaciones en este vídeo\nNo afecta a la cobertura del vídeo\n"
    "No afecta a los ingresos\nEste vídeo genera ingresos según tu configuración\n"
    "Reclamación\nTipo\n"
    "No se ha encontrado contenido protegido por derechos de autor en tu vídeo."
)


class TestReclamoEnStudioActual(unittest.TestCase):
    def test_studio_actual_sin_reclamaciones_cuenta_como_sin_reclamo(self):
        self.assertEqual(pa.clasificar_reclamo(0, PAGINA_SIN_RECLAMOS, True), "sin_reclamo")

    def test_studio_redirige_copyright_a_claims_y_no_es_una_pagina_de_otro_video(self):
        for url in ("https://studio.youtube.com/video/V1/copyright", "https://studio.youtube.com/video/V1/claims"):
            self.assertTrue(pa.es_pagina_de_reclamos(url, "V1"), url)
        self.assertFalse(pa.es_pagina_de_reclamos("https://studio.youtube.com/video/V2/claims", "V1"))
        self.assertFalse(pa.es_pagina_de_reclamos("https://accounts.google.com/v3/signin", "V1"))


class TestReclamoQueSeContradice(unittest.TestCase):
    def test_texto_de_sin_reclamos_junto_a_texto_de_reclamo_no_cuenta_como_sin_reclamo(self):
        # Un video con reclamo que se publica pierde ingresos y no se puede deshacer:
        # ante textos que se contradicen se pregunta al usuario.
        contradictoria = PAGINA_SIN_RECLAMOS + "\nSe ha encontrado contenido protegido por derechos de autor"
        self.assertEqual(pa.clasificar_reclamo(0, contradictoria, True), "desconocido")


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


PNG_1X1 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestEsperarMiniaturas(unittest.TestCase):
    """Con Chromium real: YouTube pone la URL de cada miniatura después de mostrar los resultados."""

    def capturar_con(self, html, timeout_ms):
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("sin Playwright")
        with sync_playwright() as playwright:
            try:
                navegador = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"sin Chromium: {exc}")
            try:
                pagina = navegador.new_page()
                pagina.set_content(html)
                listo = pa.esperar_miniaturas(pagina, timeout_ms=timeout_ms)
                cargada = pagina.evaluate(
                    "() => { const i = document.getElementById('m');"
                    " return i.complete && i.naturalWidth > 0; }"
                )
            finally:
                navegador.close()
        return listo, cargada

    def test_espera_a_que_cargue_una_miniatura_que_llega_tarde(self):
        html = (
            '<img id="m" style="width:240px;height:135px">'
            f"<script>setTimeout(() => {{ document.getElementById('m').src = '{PNG_1X1}'; }}, 800);</script>"
        )
        listo, cargada = self.capturar_con(html, timeout_ms=5000)
        self.assertTrue(listo)
        self.assertTrue(cargada)

    def test_si_una_miniatura_nunca_carga_sigue_sin_fallar(self):
        listo, cargada = self.capturar_con('<img id="m" style="width:240px;height:135px">', timeout_ms=500)
        self.assertFalse(listo)
        self.assertFalse(cargada)


@unittest.skipUnless(HAY_FFMPEG, "requiere ffmpeg y ffprobe")
class TestVerificar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.busqueda = mock.Mock(return_value=(None, "Banda - Album (Full Album)"))
        self.parches = [
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            # La búsqueda real abre un navegador y sale a internet.
            mock.patch.object(pa, "capturar_busqueda", self.busqueda),
        ]
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

    def test_devuelve_la_captura_de_la_busqueda_del_album(self):
        carpeta = crear_album(self.tmp / "Banda - Album", pista_s=4, video_s=5)
        captura = self.tmp / "capturas" / "busqueda.png"
        self.busqueda.return_value = (captura, "Banda - Album (Full Album)")

        resultado = pa.verificar(carpeta, ancho=320, alto=180, intro=1.0)

        self.assertEqual(Path(self.busqueda.call_args.args[0]), Path(carpeta))
        self.assertEqual(resultado["captura_busqueda"], str(captura))
        self.assertEqual(resultado["busqueda"], "Banda - Album (Full Album)")

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


class TestOrigenes(unittest.TestCase):
    """El render toma álbumes de 01_limpieza, Album/ y EP/, en ese orden."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cola, self.album, self.ep = (self.tmp / n for n in ("cola", "Album", "EP"))
        for carpeta in (self.cola, self.album, self.ep):
            carpeta.mkdir()
        parches = [
            mock.patch.object(pa, "DIRS_ORIGEN_RENDER", (self.cola, self.album, self.ep)),
            mock.patch.object(pa, "DIR_RENDERIZANDO", self.tmp / "Renderizando"),
            mock.patch.object(pa, "DIR_ERROR404", self.tmp / "Error404"),
            mock.patch.object(pa, "RENDER_SSD_DIR", Path("/")),
            mock.patch.object(pa, "MIN_FREE_SPACE_GB", 0),
            mock.patch.object(pa, "render_en_curso", return_value=False),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def test_siguiente_album_sigue_el_orden_de_las_carpetas_y_los_mas_nuevos_primero(self):
        for nombre in ("Banda B - Viejo (2019)", "Banda A - Nuevo (2021)", "Banda C - Sin anio"):
            (self.album / nombre).mkdir()
        (self.ep / "Banda Z - Corto (2025) [EP]").mkdir()
        (self.album / "notas.txt").write_text("no es un álbum", encoding="utf-8")

        self.assertEqual(pa.siguiente_album(), self.album / "Banda A - Nuevo (2021)")
        (self.cola / "Banda Q - A mano (2001)").mkdir()
        self.assertEqual(pa.siguiente_album(), self.cola / "Banda Q - A mano (2001)")

    def test_sin_albumes_en_ninguna_carpeta_no_hay_siguiente(self):
        self.assertIsNone(pa.siguiente_album())
        self.assertEqual(pa.renderizar()["motivo"], "sin_albumes")

    def test_renderizar_con_un_nombre_que_no_es_un_album_listado_no_renderiza(self):
        (self.album / "Banda - Disco (2020)").mkdir()
        with mock.patch.object(pa.subprocess, "run", side_effect=AssertionError("lanzó un render")):
            for nombre in ("..", ".", "notas.txt", "Banda - Otro (2020)", "Album/Banda - Disco (2020)",
                           str(self.album / "Banda - Disco (2020)")):
                self.assertEqual(pa.renderizar(nombre)["motivo"], "album_no_encontrado", nombre)

    def test_renderizar_le_pasa_al_render_la_ruta_del_album_de_ep(self):
        nombre = "Banda Z - Corto (2025) [EP]"
        (self.ep / nombre).mkdir()
        comandos = []

        def render_falso(comando, **kwargs):
            comandos.append(comando)
            destino = self.tmp / "Renderizando" / nombre
            destino.mkdir(parents=True)
            (destino / f"{nombre}.mp4").write_bytes(b"video")
            shutil.rmtree(self.ep / nombre)

        uploader = SimpleNamespace(encontrar_video=lambda c: next(c.glob("*.mp4"), None))
        with mock.patch.object(pa.subprocess, "run", side_effect=render_falso), mock.patch.object(
            pa, "uploader_real", return_value=uploader
        ):
            resultado = pa.renderizar()

        self.assertEqual(comandos[0][-2:], ["--test", str(self.ep / nombre)])
        self.assertTrue(resultado["ok"])
        self.assertEqual(resultado["album"], nombre)
        self.assertEqual(resultado["restantes"], 0)


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

    def test_renderizar_sin_el_ssd_montado_no_renderiza(self):
        with mock.patch.object(pa, "RENDER_SSD_DIR", self.tmp), mock.patch.object(
            pa.subprocess, "run", side_effect=AssertionError("lanzó un render")
        ):
            resultado = pa.renderizar()
        self.assertEqual(resultado["motivo"], "disco_no_disponible")

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

    def test_eliminar_borra_la_carpeta_y_su_captura(self):
        captura = pa.DIR_CAPTURAS / f"{pa.slug(self.carpeta.name)}.jpg"
        captura.parent.mkdir(parents=True)
        captura.write_bytes(b"jpg")

        resultado = pa.eliminar(self.carpeta, uploader=self.uploader(None), raices=(self.raiz,))

        self.assertTrue(resultado["ok"], resultado)
        self.assertFalse(self.carpeta.exists())
        self.assertFalse(captura.exists())

    def test_eliminar_fuera_de_las_raices_no_borra_nada(self):
        afuera = self.tmp / "otra" / "Banda - Album"
        shutil.copytree(self.carpeta, afuera)

        resultado = pa.eliminar(afuera, uploader=self.uploader(None), raices=(self.raiz,))

        self.assertEqual(resultado["motivo"], "carpeta_fuera_de_raiz")
        self.assertTrue(afuera.exists())

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


class PlaywrightFalso:
    """Registra con qué argumentos se lanza el navegador, sin abrir ninguno."""

    def __init__(self):
        self.lanzamientos = []
        self.chromium = SimpleNamespace(launch_persistent_context=self.lanzar)

    def lanzar(self, **kwargs):
        self.lanzamientos.append(kwargs)
        return "contexto"


class TestSesionDeStudio(unittest.TestCase):
    def test_el_navegador_se_lanza_con_acceso_al_llavero_y_sin_firma_de_headless(self):
        # Sin esto Brave no descifra las cookies copiadas y Studio pide iniciar sesión.
        playwright = PlaywrightFalso()
        contexto = pa.abrir_contexto_studio(playwright, Path("/perfil"), True, "/usr/bin/brave", "Default")
        self.assertEqual(contexto, "contexto")
        (lanzamiento,) = playwright.lanzamientos
        self.assertIn("--password-store=basic", lanzamiento["ignore_default_args"])
        self.assertIn("--use-mock-keychain", lanzamiento["ignore_default_args"])
        self.assertNotIn("Headless", lanzamiento["user_agent"])
        self.assertIn("--profile-directory=Default", lanzamiento["args"])
        self.assertEqual(lanzamiento["executable_path"], "/usr/bin/brave")

    def test_servicio_arrancado_antes_del_escritorio_toma_las_variables_del_gestor_de_usuario(self):
        # Con linger el servicio puede arrancar sin XDG_CURRENT_DESKTOP; sin él Brave no elige
        # el llavero de KDE y Studio vuelve a pedir iniciar sesión.
        gestor = "XDG_CURRENT_DESKTOP=KDE\nKDE_SESSION_VERSION=6\nPATH=/otro\n"
        with mock.patch.dict(pa.os.environ, {"PATH": "/usr/bin"}, clear=True), \
                mock.patch.object(pa.subprocess, "run", return_value=SimpleNamespace(stdout=gestor)):
            entorno = pa.entorno_del_navegador()
        self.assertEqual(entorno["XDG_CURRENT_DESKTOP"], "KDE")
        self.assertEqual(entorno["KDE_SESSION_VERSION"], "6")
        self.assertEqual(entorno["PATH"], "/usr/bin")

    def test_con_el_escritorio_ya_definido_no_se_consulta_al_gestor(self):
        with mock.patch.dict(pa.os.environ, {"XDG_CURRENT_DESKTOP": "GNOME"}, clear=True), \
                mock.patch.object(pa.subprocess, "run") as correr:
            entorno = pa.entorno_del_navegador()
        self.assertEqual(entorno["XDG_CURRENT_DESKTOP"], "GNOME")
        correr.assert_not_called()


class ContextoFalso:
    def __init__(self, cierra_mal=False):
        self.cierra_mal = cierra_mal
        self.pages = [SimpleNamespace(screenshot=lambda path: None)]

    def close(self):
        if self.cierra_mal:
            raise playwright_api.Error("Target closed")


class PlaywrightQueArranca:
    def __enter__(self):
        return SimpleNamespace()

    def __exit__(self, *args):
        return False


class PlaywrightQueNoArranca:
    def __enter__(self):
        raise playwright_api.Error("el driver no arranca")

    def __exit__(self, *args):
        return False


class TestActivarMonetizacionNuncaLanza(unittest.TestCase):
    """`activar_monetizacion_en_studio` promete devolver siempre un diccionario."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def activar(self, playwright, contexto=None, paso=None):
        paso = paso or (lambda pagina, video_id, captura: {"estado": "activada", "captura": None})
        with mock.patch.object(pa, "DIR_CAPTURAS", self.tmp), \
                mock.patch.object(pa, "preparar_perfil_studio", return_value=("/perfil", None, "Default")), \
                mock.patch.object(playwright_api, "sync_playwright", return_value=playwright), \
                mock.patch.object(pa, "abrir_contexto_studio", return_value=contexto or ContextoFalso()), \
                mock.patch("studio_monetizacion.activar_monetizacion", paso):
            return pa.activar_monetizacion_en_studio("V1")

    def test_un_error_al_cerrar_el_navegador_no_descarta_el_resultado(self):
        resultado = self.activar(PlaywrightQueArranca(), ContextoFalso(cierra_mal=True))
        self.assertEqual(resultado["estado"], "activada")

    def test_si_el_driver_de_playwright_no_arranca_devuelve_fallo(self):
        resultado = self.activar(PlaywrightQueNoArranca())
        self.assertEqual((resultado["estado"], resultado["motivo"]), ("fallo", "error_navegador"))

    def test_sin_ejecutable_del_navegador_se_usa_la_version_por_defecto(self):
        self.assertEqual(pa.version_del_navegador(None), pa.VERSION_STUDIO_POR_DEFECTO)


class Llamada:
    def __init__(self, resultado):
        self.resultado = resultado

    def execute(self):
        return self.resultado


class VideosFalsos:
    def __init__(self, estados, demora_actualizacion, catalogo=None):
        self.estados = list(estados)
        self.demora_actualizacion = demora_actualizacion
        self.catalogo = catalogo or {}
        self.actualizaciones = []

    def list(self, part, id):
        if "snippet" in part:  # el escaneo de publicados recientes
            return Llamada({"items": [self.catalogo[v] for v in id.split(",") if v in self.catalogo]})
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


class ListaDeSubidas:
    def __init__(self, ids):
        self.ids = ids

    def list(self, part, playlistId, maxResults):
        return Llamada({"items": [{"snippet": {"resourceId": {"videoId": v}}} for v in self.ids]})


class YouTubeFalso:
    def __init__(self, estados, demora_actualizacion=0.0, subidas=()):
        """`subidas`: (id, privacidad, publishedAt) de la lista de subidas del canal."""
        catalogo = {
            vid: {"id": vid, "snippet": {"publishedAt": cuando.isoformat()}, "status": {"privacyStatus": privacidad}}
            for vid, privacidad, cuando in subidas
        }
        self.videos_falsos = VideosFalsos(estados, demora_actualizacion, catalogo)
        self.subidas = ListaDeSubidas([vid for vid, _, _ in subidas])

    def videos(self):
        return self.videos_falsos

    def playlistItems(self):
        return self.subidas


def monetizacion_activada(video_id):
    return {"estado": "activada", "captura": "/tmp/monetizacion.png"}


def uploader_para_programar(youtube, programados=()):
    return SimpleNamespace(
        cargar_env=lambda: None,
        probar_credenciales_disponibles=lambda prefix: (youtube, [], [], []),
        get_uploads_playlist_id=lambda cliente: "UU1",
        fetch_scheduled_publish_times=lambda cliente, max_items: (set(programados), {}),
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
                activar=monetizacion_activada,
                uploader=uploader_para_programar(youtube),
            )
        self.assertTrue(resultado["ok"], resultado)
        self.assertGreaterEqual(max(esperas), 59 * 60)
        self.assertEqual(len(youtube.videos_falsos.actualizaciones), 1)
        self.assertEqual(pa.leer_pendientes(), [])

    def test_dos_programar_a_la_vez_no_eligen_la_misma_hora(self):
        contexto = multiprocessing.get_context("fork")
        resultados = contexto.Queue()

        def correr(video_id):
            # La demora abre la ventana entre elegir la hora y registrarla.
            uploader = uploader_para_programar(YouTubeFalso(["succeeded"], demora_actualizacion=0.5))
            resultados.put(
                pa.programar(video_id, reclamo=False, activar=monetizacion_activada, uploader=uploader)["publica_en"]
            )

        procesos = [contexto.Process(target=correr, args=(v,)) for v in ("V1", "V2")]
        for proceso in procesos:
            proceso.start()
        for proceso in procesos:
            proceso.join(timeout=30)
        horas = {resultados.get(timeout=5), resultados.get(timeout=5)}
        self.assertEqual(len(horas), 2, horas)


    def programar_con(self, youtube, reclamo, activar):
        return pa.programar(
            "V1", reclamo=reclamo, activar=activar, uploader=uploader_para_programar(youtube)
        )

    def test_sin_reclamo_activa_la_monetizacion_antes_de_publicar_y_lo_informa(self):
        youtube = YouTubeFalso(["succeeded"])
        llamadas = []

        def activar(video_id):
            # Todavía no se tocó la privacidad del video.
            llamadas.append((video_id, list(youtube.videos_falsos.actualizaciones)))
            return {"estado": "activada", "captura": "/tmp/m.png"}

        resultado = self.programar_con(youtube, False, activar)
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(llamadas, [("V1", [])])
        self.assertEqual(resultado["monetizacion"]["estado"], "activada")
        self.assertEqual(len(youtube.videos_falsos.actualizaciones), 1)

    def test_con_reclamo_no_intenta_monetizar_y_sigue_dos_anios_adelante(self):
        youtube = YouTubeFalso(["succeeded"])

        def activar(video_id):
            raise AssertionError("un video con reclamo no se monetiza")

        antes = datetime.now().astimezone()
        resultado = self.programar_con(youtube, True, activar)
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["monetizacion"]["estado"], "omitida")
        publica_en = datetime.fromisoformat(resultado["publica_en"])
        self.assertEqual(publica_en.year, antes.year + 2)
        status = youtube.videos_falsos.actualizaciones[0]["status"]
        self.assertEqual((status["privacyStatus"], status["publishAt"]), ("private", resultado["publica_en"]))

    def test_si_la_monetizacion_falla_el_video_se_programa_igual_y_el_fallo_viaja_en_el_resultado(self):
        youtube = YouTubeFalso(["succeeded"])
        fallo = {"estado": "fallo", "motivo": "cuestionario_de_idoneidad", "captura": "/tmp/m.png"}
        resultado = self.programar_con(youtube, False, lambda video_id: fallo)
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["monetizacion"], fallo)
        self.assertEqual(len(youtube.videos_falsos.actualizaciones), 1)


    def test_un_fallo_inesperado_del_paso_de_studio_no_frena_la_publicacion(self):
        errores = [
            playwright_api.Error("Target closed"),
            playwright_api.TimeoutError("Timeout 30000ms"),
            OSError("sin espacio"),
            ImportError("playwright"),
            subprocess.SubprocessError("brave"),
            pa.NavegadorNoDisponible("sin_perfil_de_navegador"),
        ]
        for error in errores:
            with self.subTest(error=type(error).__name__):
                youtube = YouTubeFalso(["succeeded"])

                def activar(video_id, error=error):
                    raise error

                resultado = self.programar_con(youtube, False, activar)
                self.assertTrue(resultado["ok"], resultado)
                self.assertEqual(resultado["monetizacion"]["estado"], "fallo")
                self.assertEqual(len(youtube.videos_falsos.actualizaciones), 1)

    def test_la_fecha_se_elige_despues_del_paso_de_studio_que_tarda_minutos(self):
        # El paso de Studio tarda: una fecha calculada antes puede quedar en el pasado al enviarla.
        class RelojFalso(datetime):
            adelanto = timedelta(0)

            @classmethod
            def now(cls, tz=None):
                return super().now(tz) + cls.adelanto

        def activar_lento(video_id):
            RelojFalso.adelanto += timedelta(minutes=3)
            return {"estado": "activada", "captura": None}

        publicado = datetime.now().astimezone() - timedelta(minutes=59)
        youtube = YouTubeFalso(["succeeded"], subidas=[("P1", "public", publicado)])
        with mock.patch.object(pa, "datetime", RelojFalso):
            resultado = self.programar_con(youtube, False, activar_lento)
            ahora_al_enviar = RelojFalso.now().astimezone()
        self.assertTrue(resultado["ok"], resultado)
        status = youtube.videos_falsos.actualizaciones[0]["status"]
        # Con el reloj al día, el publicado de hace 59 min ya cumplió su hora: se publica en el acto.
        self.assertEqual(status["privacyStatus"], "public", status)
        self.assertLessEqual(datetime.fromisoformat(resultado["publica_en"]), ahora_al_enviar)

    def programar_con_reclamantes(self, reclamantes, lista_negra="umg\n"):
        """`programar` revisando Studio: el video tiene reclamo de `reclamantes`."""
        archivo = self.tmp / "lista_negra_reclamantes.txt"
        if lista_negra is not None:
            archivo.write_text(lista_negra, encoding="utf-8")
        subido = datetime.now().astimezone() - timedelta(hours=3)
        pa.registrar_pendiente("V1", "Banda - Album", "Banda - Album", subido)
        youtube = YouTubeFalso(["succeeded"])
        with (
            mock.patch.object(pa, "LISTA_NEGRA_RECLAMANTES", archivo),
            mock.patch.object(pa.time, "sleep"),
        ):
            resultado = pa.programar(
                "V1",
                esperar=True,
                revisar=lambda video_id: {
                    "estado": "con_reclamo",
                    "reclamantes": reclamantes,
                    "captura_studio": "/tmp/studio_V1.png",
                },
                activar=monetizacion_activada,
                uploader=uploader_para_programar(youtube),
            )
        self.assertTrue(resultado["ok"], resultado)
        return resultado, youtube

    def assertProgramadoDosAniosAdelante(self, resultado, youtube):
        publica_en = datetime.fromisoformat(resultado["publica_en"])
        self.assertEqual(publica_en.year, datetime.now().astimezone().year + 2)
        status = youtube.videos_falsos.actualizaciones[0]["status"]
        self.assertEqual((status["privacyStatus"], status["publishAt"]), ("private", resultado["publica_en"]))

    def test_un_reclamante_de_la_lista_negra_viaja_en_el_resultado_y_el_video_sigue_dos_anios_adelante(self):
        # Defecto: no cruzar a los reclamantes con la lista negra (el servicio no se enteraría),
        # o dejar que la coincidencia cambie la fecha: borrar el video es decisión del usuario.
        resultado, youtube = self.programar_con_reclamantes(
            ["Believe Music", "Believe Music, UMG", "en nombre de Church Road Records"]
        )
        self.assertEqual(resultado["reclamantes_en_lista_negra"], ["Believe Music, UMG"])
        # El aviso de Telegram lleva la captura de Studio: tiene que viajar en el resultado.
        self.assertEqual(resultado["captura_studio"], "/tmp/studio_V1.png")
        self.assertEqual(
            resultado["reclamantes"],
            ["Believe Music", "Believe Music, UMG", "en nombre de Church Road Records"],
        )
        self.assertProgramadoDosAniosAdelante(resultado, youtube)

    def test_reclamantes_fuera_de_la_lista_negra_no_se_marcan_y_el_video_sigue_dos_anios_adelante(self):
        # Comportamiento previo: un video con reclamo de un sello que no está en la lista
        # se programa igual que siempre y no genera ningún aviso nuevo.
        resultado, youtube = self.programar_con_reclamantes(["Believe Music"])
        self.assertEqual(resultado["reclamantes_en_lista_negra"], [])
        self.assertEqual(resultado["reclamantes"], ["Believe Music"])
        self.assertProgramadoDosAniosAdelante(resultado, youtube)

    def test_sin_el_archivo_de_la_lista_negra_el_video_se_programa_igual(self):
        # Defecto: que un archivo borrado o ilegible frene la programación del video.
        resultado, youtube = self.programar_con_reclamantes(["Believe Music, UMG"], lista_negra=None)
        self.assertEqual(resultado["reclamantes_en_lista_negra"], [])
        self.assertProgramadoDosAniosAdelante(resultado, youtube)

    def programar_sin_reclamo(self, youtube, programados=()):
        resultado = pa.programar(
            "V1",
            reclamo=False,
            activar=monetizacion_activada,
            uploader=uploader_para_programar(youtube, programados),
        )
        self.assertTrue(resultado["ok"], resultado)
        return resultado, youtube.videos_falsos.actualizaciones[-1]["status"]

    def test_sin_reclamo_ni_videos_cerca_se_publica_en_el_acto_sin_contarse_a_si_mismo(self):
        ahora = datetime.now().astimezone()
        subidas = [
            ("V1", "private", ahora - timedelta(minutes=5)),  # el propio video, recién subido
            ("P0", "public", ahora - timedelta(hours=3)),
        ]
        resultado, status = self.programar_sin_reclamo(YouTubeFalso(["succeeded"], subidas=subidas))
        self.assertEqual(status["privacyStatus"], "public")
        self.assertNotIn("publishAt", status)
        self.assertTrue(resultado["publicado_ahora"])
        self.assertIn("publicado ahora", resultado["resumen"])

    def test_publicado_hace_20_minutos_el_video_espera_hasta_cumplir_la_hora(self):
        publicado = (datetime.now().astimezone() - timedelta(minutes=20)).replace(microsecond=0)
        youtube = YouTubeFalso(["succeeded"], subidas=[("P1", "public", publicado)])
        resultado, status = self.programar_sin_reclamo(youtube)
        self.assertEqual(status["privacyStatus"], "private")
        self.assertEqual(status["publishAt"], (publicado + timedelta(hours=1)).isoformat())
        self.assertFalse(resultado["publicado_ahora"])
        self.assertIn("programado para", resultado["resumen"])

    def test_un_video_programado_a_menos_de_una_hora_empuja_la_fecha(self):
        programado = (datetime.now().astimezone() + timedelta(minutes=30)).replace(microsecond=0)
        resultado, status = self.programar_sin_reclamo(YouTubeFalso(["succeeded"]), [programado])
        self.assertEqual(status["publishAt"], (programado + timedelta(hours=1)).isoformat())

    def test_un_publicado_viejo_o_un_privado_no_frenan_la_publicacion(self):
        ahora = datetime.now().astimezone()
        subidas = [
            ("PRIV", "private", ahora - timedelta(minutes=10)),
            ("VIEJO", "public", ahora - timedelta(minutes=61)),
        ]
        resultado, status = self.programar_sin_reclamo(YouTubeFalso(["succeeded"], subidas=subidas))
        self.assertTrue(resultado["publicado_ahora"])

    def test_el_segundo_video_espera_una_hora_al_primero_aunque_la_api_aun_no_lo_liste(self):
        primero, _ = self.programar_sin_reclamo(YouTubeFalso(["succeeded"]))
        segundo, status = self.programar_sin_reclamo(YouTubeFalso(["succeeded"]))
        self.assertTrue(primero["publicado_ahora"])
        publicado_el_primero = datetime.fromisoformat(primero["publica_en"])
        self.assertEqual(
            datetime.fromisoformat(status["publishAt"]), publicado_el_primero + timedelta(hours=1)
        )

    def test_el_resumen_dice_publicado_ahora_o_programado_para(self):
        cuando = momento("2026-11-02 18:00")
        ahora = pa.armar_resumen_programado("T", cuando, False, [], publicado_ahora=True)
        self.assertEqual(ahora, "✓ T\nSin reclamos: publicado ahora.")
        luego = pa.armar_resumen_programado("T", cuando, False, [], publicado_ahora=False)
        self.assertEqual(luego, "✓ T\nSin reclamos: programado para 2026-11-02 18:00.")


class ServidorKeepAliveQueMuere:
    """Servidor HTTPS/1.1 local que cierra la conexión keep-alive tras el primer pedido.

    Cierra con FIN y luego RST (SO_LINGER 0), como un servidor que descartó una
    conexión ociosa: el siguiente envío del cliente falla con EPIPE.
    """

    def __init__(self, directorio):
        self.certificado = directorio / "certificado.pem"
        clave = directorio / "clave.pem"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                "-nodes", "-keyout", str(clave), "-out", str(self.certificado), "-days", "2",
                "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        self.pedidos = 0
        servidor = self

        class Manejador(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                servidor.pedidos += 1
                cuerpo = json.dumps({"items": [], "pedido": servidor.pedidos}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
                if servidor.pedidos == 1:
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    self.close_connection = True

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
        self.http.daemon_threads = True
        contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        contexto.load_cert_chain(self.certificado, clave)
        self.http.socket = contexto.wrap_socket(self.http.socket, server_side=True)
        threading.Thread(target=self.http.serve_forever, args=(0.01,), daemon=True).start()

    def cerrar(self):
        self.http.shutdown()
        self.http.server_close()

    def cliente_youtube(self):
        """Mismo armado que en producción: googleapiclient + AuthorizedHttp + httplib2."""
        raiz = f"https://127.0.0.1:{self.http.server_address[1]}/"
        documento = {
            "kind": "discovery#restDescription",
            "name": "prueba",
            "version": "v1",
            "protocol": "rest",
            "rootUrl": raiz,
            "servicePath": "",
            "baseUrl": raiz,
            "basePath": "/",
            "parameters": {},
            "resources": {
                "videos": {
                    "methods": {
                        "list": {
                            "id": "prueba.videos.list",
                            "path": "videos",
                            "httpMethod": "GET",
                            "parameters": {},
                            "response": {"$ref": "Lista"},
                        }
                    }
                }
            },
            "schemas": {"Lista": {"id": "Lista", "type": "object", "properties": {}}},
        }
        http = AuthorizedHttp(Credentials(token="falso"), http=httplib2.Http(ca_certs=str(self.certificado)))
        return build_from_document(documento, http=http)


class ClienteQueFalla:
    """Cliente YouTube falso: lanza los errores indicados antes de responder."""

    def __init__(self, errores, respuesta=None):
        self.errores = list(errores)
        self.respuesta = respuesta if respuesta is not None else {"items": []}
        self.llamadas = 0
        self.cierres = 0

    def close(self):
        self.cierres += 1

    def videos(self):
        return self

    def list(self, **kwargs):
        return self._ejecutable()

    def update(self, part, body):
        return self._ejecutable()

    def _ejecutable(self):
        cliente = self

        class Ejecutable:
            def execute(self):
                cliente.llamadas += 1
                if cliente.errores:
                    raise cliente.errores.pop(0)
                return cliente.respuesta

        return Ejecutable()


class LlamadaQueFalla:
    def __init__(self, error):
        self.error = error

    def execute(self):
        raise self.error


class VideosConConexionMuerta(VideosFalsos):
    """La 2.ª consulta y la 1.ª actualización fallan como en una conexión keep-alive cerrada."""

    def __init__(self, estados):
        super().__init__(estados, 0.0)
        self.consultas = 0
        self.intentos_de_actualizar = 0

    def list(self, part, id):
        self.consultas += 1
        llamada = super().list(part, id)
        return LlamadaQueFalla(BrokenPipeError(32, "Broken pipe")) if self.consultas == 2 else llamada

    def update(self, part, body):
        self.intentos_de_actualizar += 1
        if self.intentos_de_actualizar == 1:
            return LlamadaQueFalla(BrokenPipeError(32, "Broken pipe"))
        return super().update(part, body)


class YouTubeConConexionMuerta(YouTubeFalso):
    def __init__(self, estados):
        self.videos_falsos = VideosConConexionMuerta(estados)
        self.subidas = ListaDeSubidas([])
        self.cierres = 0

    def close(self):
        self.cierres += 1


class TestEjecutarApi(unittest.TestCase):
    def setUp(self):
        self.esperas = []
        parche = mock.patch.object(pa.time, "sleep", self.esperas.append)
        parche.start()
        self.addCleanup(parche.stop)

    def uploader(self, **cambios):
        return SimpleNamespace(**{**vars(uploader_para_programar(None)), **cambios})

    @unittest.skipUnless(shutil.which("openssl"), "falta openssl para el certificado de prueba")
    def test_conexion_keep_alive_muerta_se_reintenta_con_una_conexion_nueva(self):
        directorio = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directorio)
        servidor = ServidorKeepAliveQueMuere(directorio)
        self.addCleanup(servidor.cerrar)
        cliente = servidor.cliente_youtube()
        self.addCleanup(cliente.close)
        yt = {"cliente": cliente}
        listar = lambda c: c.videos().list().execute()
        # El socket del primer pedido, para saber cuándo le llegó el cierre del servidor.
        sockets = []
        enviar_real = http.client.HTTPConnection.send

        def enviar(conexion, datos):
            sockets.append(conexion.sock)
            return enviar_real(conexion, datos)

        with mock.patch.object(http.client.HTTPConnection, "send", enviar):
            self.assertEqual(pa.ejecutar_api(self.uploader(), yt, listar)["pedido"], 1)
            # Esperar a que el RST del servidor llegue al cliente, como tras 5 minutos de espera.
            vigilante = select.poll()
            vigilante.register(sockets[0], select.POLLERR)
            self.assertTrue(vigilante.poll(2000), "el servidor no cerró la conexión")
            respuesta = pa.ejecutar_api(self.uploader(), yt, listar)

        self.assertEqual(respuesta["pedido"], 2)
        # httplib2 no cierra el socket que falló con EPIPE: reintentar sobre él vuelve a fallar
        # (SSLError al leer) y recién el tercer intento abre otra conexión. Con las conexiones
        # descartadas, un solo reintento alcanza.
        self.assertEqual(len(self.esperas), 1)

    def test_errores_de_conexion_transitorios_se_reintentan(self):
        errores = [
            BrokenPipeError(32, "Broken pipe"),
            ConnectionResetError(104, "Connection reset by peer"),
            TimeoutError("timed out"),
            ssl.SSLError("EOF occurred in violation of protocol"),
            httplib2.ServerNotFoundError("Unable to find the server"),
            TransportError("no hay conexion"),
        ]
        for error in errores:
            with self.subTest(error=type(error).__name__):
                cliente = ClienteQueFalla([error], respuesta={"items": ["ok"]})
                respuesta = pa.ejecutar_api(
                    self.uploader(), {"cliente": cliente}, lambda c: c.videos().list().execute()
                )
                self.assertEqual(respuesta, {"items": ["ok"]})
                self.assertEqual((cliente.llamadas, cliente.cierres), (2, 1))

    def test_agotados_los_reintentos_vuelve_a_lanzar_el_error_original(self):
        cliente = ClienteQueFalla([BrokenPipeError(32, "Broken pipe")] * 10)
        with self.assertRaises(BrokenPipeError):
            pa.ejecutar_api(self.uploader(), {"cliente": cliente}, lambda c: c.videos().list().execute())
        self.assertEqual(cliente.llamadas, 1 + pa.REINTENTOS_CONEXION)
        self.assertEqual(len(self.esperas), pa.REINTENTOS_CONEXION)

    def test_error_de_cuota_sigue_rotando_la_credencial(self):
        cuota = HttpError(httplib2.Response({"status": 403}), b"{}")
        agotado = ClienteQueFalla([cuota])
        siguiente = ClienteQueFalla([], respuesta={"items": ["otra credencial"]})
        uploader = self.uploader(is_quota_error=lambda exc: True, authenticate_next=lambda prefix: siguiente)
        yt = {"cliente": agotado}
        respuesta = pa.ejecutar_api(uploader, yt, lambda c: c.videos().list().execute())
        self.assertEqual(respuesta, {"items": ["otra credencial"]})
        self.assertIs(yt["cliente"], siguiente)
        self.assertEqual(self.esperas, [])

    def test_token_revocado_sigue_rotando_la_credencial(self):
        revocado = ClienteQueFalla([RefreshError("invalid_grant")])
        siguiente = ClienteQueFalla([], respuesta={"items": ["otra credencial"]})
        uploader = self.uploader(is_invalid_grant_error=lambda exc: True, authenticate_next=lambda prefix: siguiente)
        yt = {"cliente": revocado}
        respuesta = pa.ejecutar_api(uploader, yt, lambda c: c.videos().list().execute())
        self.assertEqual(respuesta, {"items": ["otra credencial"]})
        self.assertIs(yt["cliente"], siguiente)

    def test_error_http_que_no_es_de_cuota_se_lanza_sin_reintentar(self):
        cliente = ClienteQueFalla([HttpError(httplib2.Response({"status": 500}), b"{}")])
        with self.assertRaises(HttpError):
            pa.ejecutar_api(self.uploader(), {"cliente": cliente}, lambda c: c.videos().list().execute())
        self.assertEqual((cliente.llamadas, cliente.cierres, self.esperas), (1, 0, []))


class VideosParaBorrar:
    """`videos()` de un canal: lo que `eliminar_video` consulta y borra."""

    def __init__(self, titulos, errores_al_borrar=()):
        self.titulos = titulos  # id -> título
        self.errores_al_borrar = list(errores_al_borrar)
        self.borrados = []

    def list(self, part, id):
        items = [{"id": id, "snippet": {"title": self.titulos[id]}}] if id in self.titulos else []
        return Llamada({"items": items})

    def delete(self, id):
        videos = self

        class Borrado:
            def execute(self):
                if videos.errores_al_borrar:
                    raise videos.errores_al_borrar.pop(0)
                videos.borrados.append(id)

        return Borrado()


class CanalParaBorrar:
    def __init__(self, titulos, errores_al_borrar=()):
        self.videos_falsos = VideosParaBorrar(titulos, errores_al_borrar)

    def videos(self):
        return self.videos_falsos


class TestEliminarVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        parche = mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs")
        parche.start()
        self.addCleanup(parche.stop)

    def test_borra_el_video_pedido_y_devuelve_su_titulo(self):
        # Defecto: borrar otro id, o no borrar y responder ok.
        canal = CanalParaBorrar({"V1": "Banda - Disco", "V2": "Otra - Cosa"})
        resultado = pa.eliminar_video("V1", uploader=uploader_para_programar(canal))
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual((resultado["video_id"], resultado["titulo"]), ("V1", "Banda - Disco"))
        self.assertEqual(canal.videos_falsos.borrados, ["V1"])

    def test_un_video_que_no_existe_no_se_borra_a_ciegas(self):
        # Defecto: dar por borrado, o llamar a delete, con un id que el canal no tiene.
        canal = CanalParaBorrar({"V2": "Otra - Cosa"})
        resultado = pa.eliminar_video("V1", uploader=uploader_para_programar(canal))
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "no_existe"))
        self.assertEqual(canal.videos_falsos.borrados, [])

    def test_con_la_cuota_agotada_borra_con_la_credencial_siguiente(self):
        # Defecto: llamar a la API sin pasar por ejecutar_api y quedar sin borrar por cuota.
        cuota = HttpError(httplib2.Response({"status": 403}), b"{}")
        agotada = CanalParaBorrar({"V1": "Banda - Disco"}, errores_al_borrar=[cuota])
        siguiente = CanalParaBorrar({"V1": "Banda - Disco"})
        uploader = uploader_para_programar(agotada)
        uploader.is_quota_error = lambda exc: True
        uploader.authenticate_next = lambda prefix: siguiente
        resultado = pa.eliminar_video("V1", uploader=uploader)
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual((agotada.videos_falsos.borrados, siguiente.videos_falsos.borrados), ([], ["V1"]))

    def test_un_error_de_la_api_se_responde_como_json_sin_lanzar(self):
        # Defecto: dejar escapar la excepción; el servicio solo ve «sin_respuesta» y no sabe si se borró.
        prohibido = HttpError(httplib2.Response({"status": 403}), b"{}")
        canal = CanalParaBorrar({"V1": "Banda - Disco"}, errores_al_borrar=[prohibido])
        resultado = pa.eliminar_video("V1", uploader=uploader_para_programar(canal))
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "error_api"))
        self.assertEqual(canal.videos_falsos.borrados, [])

    def test_el_subcomando_corre_eliminar_video_y_no_otro_paso(self):
        # Defecto: main manda cualquier subcomando desconocido a `estado`; sin su rama,
        # el servicio recibiría el estado de los pendientes en vez de borrar el video.
        salida = io.StringIO()
        with (
            mock.patch.object(pa, "eliminar_video", return_value={"ok": True, "video_id": "V1"}) as borrar,
            mock.patch.object(pa, "estado", side_effect=AssertionError("corrió estado")),
            contextlib.redirect_stdout(salida),
        ):
            codigo = pa.main(["eliminar_video", "--video-id", "V1"])
        borrar.assert_called_once_with("V1")
        self.assertEqual(codigo, 0)
        self.assertEqual(json.loads(salida.getvalue()), {"ok": True, "video_id": "V1"})


class TestProgramarSinConexion(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.parches = [
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "PROGRAMADOS_JSONL", self.tmp / "programados.jsonl"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
            mock.patch.object(pa.time, "sleep"),
        ]
        for parche in self.parches:
            parche.start()

    def tearDown(self):
        for parche in self.parches:
            parche.stop()
        shutil.rmtree(self.tmp)

    def test_se_agotan_los_reintentos_y_programar_responde_sin_conexion(self):
        cliente = ClienteQueFalla([BrokenPipeError(32, "Broken pipe")] * 10)
        resultado = pa.programar(
            "V1", reclamo=False, activar=monetizacion_activada, uploader=uploader_para_programar(cliente)
        )
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["motivo"], "sin_conexion")
        self.assertEqual(resultado["video_id"], "V1")
        self.assertIn("Broken pipe", resultado["detalle"])
        self.assertEqual(cliente.llamadas, 1 + pa.REINTENTOS_CONEXION)

    def test_conexion_muerta_tras_la_espera_no_aborta_la_programacion(self):
        # Caso real: la consulta tras la espera de 5 min usa una conexión keep-alive ya cerrada.
        youtube = YouTubeConConexionMuerta(["processing", "succeeded"])
        resultado = pa.programar(
            "V1",
            esperar=True,
            reclamo=False,
            activar=monetizacion_activada,
            uploader=uploader_para_programar(youtube),
        )
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(youtube.cierres, 2)
        self.assertEqual(len(youtube.videos_falsos.actualizaciones), 1)


if __name__ == "__main__":
    unittest.main()
