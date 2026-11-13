import contextlib
import errno
import fcntl
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
from urllib.parse import parse_qs, quote, urlparse

import httplib2
import playwright.sync_api as playwright_api
import requests
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build_from_document
from googleapiclient.errors import HttpError
from PIL import Image

import censura_portadas as cp
import pipeline_agente as pa
import registro_publicados as rp
from tests.test_censura_portadas import OllamaFalso, respuesta_con

TZ = timezone(timedelta(hours=-6))
# Antes de que los tests la reemplacen por un falso: la prueba de verdad la busca en /proc.
ANALISIS_EN_CURSO_REAL = pa.analisis_en_curso
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
            # renderizar borra lo que hay ahí: nunca el temporal de un render real.
            mock.patch.object(pa, "SSD_TEMP_DIR", self.tmp / "temp_render"),
            mock.patch.object(pa, "MIN_FREE_SPACE_GB", 0),
            mock.patch.object(pa, "render_en_curso", return_value=False),
            mock.patch.object(pa, "analisis_en_curso", return_value=False),
            mock.patch.object(pa, "REGISTRO_PUBLICADOS", self.tmp / "ya_publicados.tsv"),
            mock.patch.object(pa, "DIR_YA_PUBLICADOS", self.tmp / "Ya publicados"),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def renderizar(self, carpeta=None):
        """Corre renderizar con un render falso; devuelve el resultado y los álbumes renderizados."""
        renderizados = []

        def render_falso(comando, **kwargs):
            origen = Path(comando[-1])
            renderizados.append(origen)
            destino = self.tmp / "Renderizando" / origen.name
            destino.mkdir(parents=True)
            (destino / f"{origen.name}.mp4").write_bytes(b"video")
            shutil.rmtree(origen)

        uploader = SimpleNamespace(encontrar_video=lambda c: next(c.glob("*.mp4"), None))
        with mock.patch.object(pa.subprocess, "run", side_effect=render_falso), mock.patch.object(
            pa, "uploader_real", return_value=uploader
        ):
            return pa.renderizar(carpeta), renderizados

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

    def test_renderizar_borra_lo_que_dejan_los_renders_y_conserva_los_videos_rescatados(self):
        temporal = self.tmp / "temp_render"
        cortado = temporal / "Banda V - Cortado (2010)"
        cortado.mkdir(parents=True)
        (cortado / "01 - Tema.mp3").write_bytes(b"audio")
        (temporal / "suelto.tmp").write_bytes(b"x")
        rescatado = temporal / pa.CARPETA_RESCATE_RENDER / "Banda R - Rescatado (2011).mp4"
        rescatado.parent.mkdir()
        rescatado.write_bytes(b"video")
        album = self.album / "Banda A - Largo (2024)"
        album.mkdir()
        al_lanzar = []

        def render_sin_espacio(comando, **kwargs):
            al_lanzar.append(sorted(p.name for p in temporal.iterdir()))
            # Un render que se queda sin espacio deja su copia y el video a medias.
            copia = temporal / album.name
            copia.mkdir()
            (copia / f"{album.name}__video.mp4").write_bytes(b"medio video")

        with mock.patch.object(pa.subprocess, "run", side_effect=render_sin_espacio):
            resultado = pa.renderizar()

        self.assertEqual(resultado["motivo"], "render_fallido")
        self.assertEqual(al_lanzar, [[pa.CARPETA_RESCATE_RENDER]])
        self.assertEqual([p.name for p in temporal.iterdir()], [pa.CARPETA_RESCATE_RENDER])
        self.assertEqual(rescatado.read_bytes(), b"video")
        self.assertTrue(album.is_dir())

    def test_un_render_lanzado_a_mano_durante_el_del_servicio_conserva_su_temporal(self):
        temporal = self.tmp / "temp_render"
        album = self.album / "Banda A - Largo (2024)"
        album.mkdir()
        a_mano = temporal / "Banda M - A mano (2024)"

        def render_con_otro_a_mano(comando, **kwargs):
            (temporal / album.name).mkdir(parents=True)
            a_mano.mkdir()
            (a_mano / "Banda M - A mano (2024)__video.mp4").write_bytes(b"medio video")

        # Antes del render no corre ninguno; al terminar, sigue corriendo el lanzado a mano.
        with mock.patch.object(pa, "render_en_curso", side_effect=[False, True]), mock.patch.object(
            pa.subprocess, "run", side_effect=render_con_otro_a_mano
        ):
            self.assertEqual(pa.renderizar()["motivo"], "render_fallido")
        self.assertEqual((a_mano / "Banda M - A mano (2024)__video.mp4").read_bytes(), b"medio video")

    def test_con_un_render_corriendo_no_borra_su_temporal(self):
        del_render = self.tmp / "temp_render" / "Banda A - Largo (2024)"
        del_render.mkdir(parents=True)
        (self.album / "Banda B - Siguiente (2024)").mkdir()
        with mock.patch.object(pa, "render_en_curso", return_value=True), mock.patch.object(
            pa.subprocess, "run", side_effect=AssertionError("lanzó un render")
        ):
            self.assertEqual(pa.renderizar()["motivo"], "render_en_curso")
        self.assertTrue(del_render.is_dir())

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

    def test_renderizar_aparta_lo_ya_publicado_y_renderiza_el_siguiente(self):
        # Defecto: se renderiza 12 minutos un álbum que ya está en el canal, o al apartarlo
        # se mete adentro de la carpeta con el mismo nombre que ya estaba en Ya publicados/.
        titulo = "🇺🇸 BANDA A - NUEVO (2021) • [Death Metal] ⟨FULL ALBUM⟩"
        rp.anotar(pa.REGISTRO_PUBLICADOS, [(titulo, "canal", "vid00000001")])
        for nombre in ("Banda A - Nuevo (2021)", "Banda B - Viejo (2019)"):
            (self.album / nombre).mkdir()
        anterior = pa.DIR_YA_PUBLICADOS / "Banda A - Nuevo (2021)"
        anterior.mkdir(parents=True)

        resultado, renderizados = self.renderizar()

        self.assertEqual(renderizados, [self.album / "Banda B - Viejo (2019)"])
        self.assertTrue(resultado["ok"], resultado)
        apartado = pa.DIR_YA_PUBLICADOS / "Banda A - Nuevo (2021)__2"
        self.assertTrue(apartado.is_dir())
        self.assertEqual(list(anterior.iterdir()), [])
        self.assertFalse((self.album / "Banda A - Nuevo (2021)").exists())
        self.assertEqual(resultado["ya_publicados"], [
            {"album": "Banda A - Nuevo (2021)", "origen": "canal", "como": titulo, "destino": str(apartado)},
        ])

    def test_renderizar_con_carpeta_renderiza_aunque_este_en_el_registro(self):
        # Defecto: un álbum pedido a mano con --carpeta se aparta en lugar de renderizarse.
        nombre = "Banda A - Nuevo (2021)"
        rp.anotar(pa.REGISTRO_PUBLICADOS, [(nombre, "eliminar", "")])
        (self.album / nombre).mkdir()

        resultado, renderizados = self.renderizar(nombre)

        self.assertEqual(renderizados, [self.album / nombre])
        self.assertTrue(resultado["ok"], resultado)
        self.assertFalse(pa.DIR_YA_PUBLICADOS.exists())

    def test_si_todo_esta_publicado_lo_aparta_y_no_renderiza(self):
        # Defecto: con todo lo que queda ya publicado, se renderiza igual uno o no se aparta nada
        # y el servicio vuelve a lanzar el render en cada vuelta.
        nombres = ("Banda A - Nuevo (2021)", "Banda B - Viejo (2019)")
        rp.anotar(pa.REGISTRO_PUBLICADOS, [(n, "subir", "") for n in nombres])
        (self.cola / nombres[0]).mkdir()
        (self.ep / nombres[1]).mkdir()

        with mock.patch.object(pa.subprocess, "run", side_effect=AssertionError("lanzó un render")):
            resultado = pa.renderizar()

        self.assertEqual(resultado["motivo"], "sin_albumes")
        self.assertEqual([a["album"] for a in resultado["ya_publicados"]], list(nombres))
        self.assertEqual(pa.listar_carpetas(pa.DIR_YA_PUBLICADOS), sorted(nombres))
        self.assertIsNone(pa.siguiente_album())

    def test_un_album_publicado_que_no_se_puede_apartar_no_se_renderiza_ni_se_copia(self):
        # Defecto: si moverlo falla, se renderiza igual el álbum ya publicado; o el movimiento
        # cae en copiarlo y, cuando el original no se puede borrar, cada vuelta deja otra copia
        # entera en Ya publicados/.
        rp.anotar(pa.REGISTRO_PUBLICADOS, [("Banda A - Nuevo (2021)", "canal", "")])
        for nombre in ("Banda A - Nuevo (2021)", "Banda B - Viejo (2019)"):
            (self.album / nombre).mkdir()
        (self.album / "Banda A - Nuevo (2021)" / "01. pista.flac").write_bytes(b"audio")

        with mock.patch.object(pa.os, "rename", side_effect=PermissionError("sin permiso")):
            resultado, renderizados = self.renderizar()

        self.assertEqual(renderizados, [self.album / "Banda B - Viejo (2019)"])
        self.assertTrue((self.album / "Banda A - Nuevo (2021)" / "01. pista.flac").is_file())
        self.assertEqual(pa.listar_carpetas(pa.DIR_YA_PUBLICADOS), [])
        self.assertEqual(resultado["ya_publicados"], [
            {"album": "Banda A - Nuevo (2021)", "origen": "canal", "como": "Banda A - Nuevo (2021)",
             "error": "sin permiso"},
        ])


class TestPreguntarAntesDeRenderizar(unittest.TestCase):
    """`buscar` elige el álbum sin renderizarlo; `eliminar` borra uno que sigue en su origen."""

    def setUp(self):
        TestOrigenes.setUp(self)
        for parche in (
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
        ):
            parche.start()
            self.addCleanup(parche.stop)

    def test_buscar_aparta_lo_publicado_y_devuelve_la_busqueda_y_la_portada_sin_renderizar(self):
        rp.anotar(pa.REGISTRO_PUBLICADOS, [("Banda A - Nuevo (2021)", "eliminar", "")])
        (self.album / "Banda A - Nuevo (2021)").mkdir()
        siguiente = self.album / "Banda B - Viejo (2019)"
        siguiente.mkdir()
        for archivo in ("aaa_shadow.png", "Arte.jpg", "cover.jpg", "01. pista.flac"):
            (siguiente / archivo).write_bytes(b"x")
        captura = self.tmp / "busqueda.png"
        buscadas = []

        def capturar(carpeta, uploader=None):
            buscadas.append(Path(carpeta))
            return captura, "Banda B - Viejo (2019) (Full Album)"

        with mock.patch.object(pa.subprocess, "run", side_effect=AssertionError("lanzó un render")), \
                mock.patch.object(pa, "capturar_busqueda", side_effect=capturar):
            resultado = pa.buscar()

        self.assertEqual(buscadas, [siguiente])
        self.assertEqual(resultado["ok"], True)
        self.assertEqual(resultado["album"], "Banda B - Viejo (2019)")
        self.assertEqual(resultado["carpeta"], str(siguiente))
        self.assertEqual(resultado["portada"], str(siguiente / "cover.jpg"))
        self.assertEqual(resultado["captura_busqueda"], str(captura))
        self.assertEqual(resultado["busqueda"], "Banda B - Viejo (2019) (Full Album)")
        self.assertEqual([a["album"] for a in resultado["ya_publicados"]], ["Banda A - Nuevo (2021)"])
        self.assertTrue((pa.DIR_YA_PUBLICADOS / "Banda A - Nuevo (2021)").is_dir())
        self.assertTrue(siguiente.is_dir())

    def test_buscar_al_azar_elige_entre_las_tres_carpetas_y_sigue_apartando_los_publicados(self):
        for carpeta, nombre in ((self.cola, "Banda A - Cola (2024)"), (self.album, "Banda B - Album (2023)"),
                                (self.ep, "Banda C - Corto (2010) [EP]"), (self.ep, "Banda D - Publicado (2001) [EP]")):
            (carpeta / nombre).mkdir()
        rp.anotar(pa.REGISTRO_PUBLICADOS, [("Banda D - Publicado (2001) [EP]", "canal", "")])
        captura = (self.tmp / "busqueda.png", "titulo")

        # El azar falso da vuelta la lista: lo último de EP/ queda primero.
        with mock.patch.object(pa.random, "shuffle", side_effect=lambda lista: lista.reverse()), \
                mock.patch.object(pa, "capturar_busqueda", return_value=captura):
            al_azar = pa.buscar(aleatorio=True)
            en_orden = pa.buscar()

        self.assertEqual(al_azar["album"], "Banda C - Corto (2010) [EP]")
        self.assertEqual([a["album"] for a in al_azar["ya_publicados"]], ["Banda D - Publicado (2001) [EP]"])
        self.assertTrue((pa.DIR_YA_PUBLICADOS / "Banda D - Publicado (2001) [EP]").is_dir())
        self.assertEqual(en_orden["album"], "Banda A - Cola (2024)")

    def test_buscar_salta_los_albumes_de_la_lista(self):
        # Defecto: sin esto, después de "➕ A la lista" se vuelve a preguntar por el mismo álbum.
        for nombre in ("Banda A - Primero (2024)", "Banda B - Segundo (2023)"):
            (self.album / nombre).mkdir()
        captura = (self.tmp / "busqueda.png", "titulo")

        with mock.patch.object(pa, "capturar_busqueda", return_value=captura):
            siguiente = pa.buscar(excluir=["Banda A - Primero (2024)"])
            todos = pa.buscar(excluir=["Banda A - Primero (2024)", "Banda B - Segundo (2023)"])

        self.assertEqual(siguiente["album"], "Banda B - Segundo (2023)")
        self.assertEqual(todos["motivo"], "sin_albumes")
        self.assertTrue((self.album / "Banda A - Primero (2024)").is_dir())

    def test_el_subcomando_buscar_pasa_cada_excluir(self):
        # Defecto: el servicio manda la lista por la línea de comandos; si no llega, se pierde.
        salida = io.StringIO()
        with mock.patch.object(pa, "buscar", return_value={"ok": True}) as buscar, \
                contextlib.redirect_stdout(salida):
            # Como lo manda el servicio: con "=", para un nombre que empieza con guion.
            codigo = pa.main(["buscar", "--aleatorio", "--excluir=-Ruido - Disco", "--excluir=B"])
        buscar.assert_called_once_with(aleatorio=True, excluir=["-Ruido - Disco", "B"])
        self.assertEqual(codigo, 0)

    def test_buscar_sin_albumes_no_captura_nada(self):
        with mock.patch.object(pa, "capturar_busqueda", side_effect=AssertionError("capturó")):
            self.assertEqual(pa.buscar()["motivo"], "sin_albumes")

    def album_con_pista(self, nombre, origen=None):
        carpeta = (origen or self.ep) / nombre
        carpeta.mkdir()
        (carpeta / "01. pista.flac").write_bytes(b"audio")
        return carpeta

    def test_eliminar_aparta_el_album_en_la_papelera_de_su_disco_y_lo_anota(self):
        # Defecto: un toque de más en Eliminar borraba la única copia del álbum, sin vuelta atrás.
        carpeta = self.album_con_pista("Banda - Corto (2025) [EP]")

        resultado = pa.eliminar(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        self.assertFalse(carpeta.exists())
        entrada = Path(resultado["papelera"])
        # Al lado de la raíz: en el mismo disco, mover es un rename y no una copia.
        self.assertEqual(entrada.parent, self.tmp / "Papelera")
        self.assertEqual((entrada / carpeta.name / "01. pista.flac").read_bytes(), b"audio")
        fila = rp.leer(pa.REGISTRO_PUBLICADOS).buscar("Banda - Corto (2025) [EP]")
        self.assertEqual(fila["origen"], "eliminar")

    def test_eliminar_no_aparta_la_carpeta_de_origen_ni_una_parte_de_un_album(self):
        # Defecto: con las carpetas de origen como raíces, una ruta mal pasada aparta parte de la
        # única copia de un álbum.
        album = self.album / "Banda - Doble (2020)"
        (album / "CD1").mkdir(parents=True)

        for carpeta in (self.album, album / "CD1"):
            resultado = pa.eliminar(carpeta)
            self.assertEqual(resultado["motivo"], "carpeta_fuera_de_raiz", carpeta)

        self.assertTrue((album / "CD1").is_dir())
        self.assertFalse((self.tmp / "Papelera").exists())
        self.assertFalse(pa.REGISTRO_PUBLICADOS.exists())

    def test_restaurar_devuelve_el_album_y_buscar_lo_vuelve_a_ofrecer(self):
        # Defecto: con la fila de `eliminar` en el registro, `buscar` apartaba a Ya publicados/
        # el álbum recién devuelto en vez de preguntar por él.
        carpeta = self.album_con_pista("Banda - Corto (2025) [EP]")
        entrada = pa.eliminar(carpeta)["papelera"]

        resultado = pa.restaurar(entrada)
        with mock.patch.object(pa, "capturar_busqueda", return_value=(None, "busqueda")):
            siguiente = pa.buscar()

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["carpeta"], str(carpeta))
        self.assertEqual((carpeta / "01. pista.flac").read_bytes(), b"audio")
        self.assertFalse(Path(entrada).exists())
        self.assertEqual(siguiente["album"], "Banda - Corto (2025) [EP]")
        self.assertEqual(siguiente["ya_publicados"], [])

    def test_restaurar_deja_las_filas_que_no_puso_eliminar(self):
        # Defecto: sacar todas las filas del álbum borra también la de su video en el canal.
        carpeta = self.album_con_pista("Banda - Corto (2025) [EP]")
        rp.anotar(pa.REGISTRO_PUBLICADOS, [("Banda - Corto (2025) [EP]", "canal", "VID1")])
        entrada = pa.eliminar(carpeta)["papelera"]

        pa.restaurar(entrada)

        filas = rp.leer(pa.REGISTRO_PUBLICADOS).filas
        self.assertEqual([(f["origen"], f["video_id"]) for f in filas], [("canal", "VID1")])

    def test_restaurar_no_pisa_una_carpeta_que_ya_volvio(self):
        # Defecto: devolverlo encima de otra copia (bajada de nuevo) la pisa o la mezcla.
        carpeta = self.album_con_pista("Banda - Corto (2025) [EP]")
        entrada = Path(pa.eliminar(carpeta)["papelera"])
        carpeta.mkdir()
        (carpeta / "nueva.flac").write_bytes(b"otra")

        resultado = pa.restaurar(entrada)

        self.assertEqual(resultado["motivo"], "ya_existe")
        self.assertEqual(sorted(p.name for p in carpeta.iterdir()), ["nueva.flac"])
        self.assertTrue((entrada / carpeta.name / "01. pista.flac").is_file())
        # El registro no se toca: el álbum sigue en la papelera.
        fila = rp.leer(pa.REGISTRO_PUBLICADOS).buscar(carpeta.name)
        self.assertEqual(fila["origen"], "eliminar")

    def test_restaurar_no_mueve_nada_fuera_de_la_papelera(self):
        # Defecto: una ruta cualquiera en el estado del servicio movería una carpeta ajena.
        afuera = self.tmp / "otra" / "abc"
        (afuera / "Banda - Ajena (2020)").mkdir(parents=True)
        destino = self.ep / "Banda - Ajena (2020)"
        (afuera / "origen.json").write_text(json.dumps({"carpeta": str(destino)}), encoding="utf-8")

        resultado = pa.restaurar(afuera)

        self.assertEqual(resultado["motivo"], "fuera_de_papelera")
        self.assertFalse(destino.exists())
        self.assertTrue((afuera / "Banda - Ajena (2020)").is_dir())

    def test_restaurar_lo_que_ya_se_vacio_dice_que_ya_no_esta(self):
        # Defecto: la entrada ya borrada hacía fallar el paso sin decir por qué.
        entrada = pa.eliminar(self.album_con_pista("Banda - Corto (2025) [EP]"))["papelera"]
        shutil.rmtree(entrada)

        self.assertEqual(pa.restaurar(entrada)["motivo"], "ya_no_esta")

    def test_vaciar_papelera_borra_lo_que_tiene_mas_de_un_dia_y_deja_lo_nuevo(self):
        # Defecto: una papelera que no se vacía llena el disco; una que borra todo no deja deshacer.
        vieja = Path(pa.eliminar(self.album_con_pista("Banda - Viejo (2020)"))["papelera"])
        nueva = Path(pa.eliminar(self.album_con_pista("Banda - Nuevo (2021)"))["papelera"])
        origen = json.loads((vieja / "origen.json").read_text(encoding="utf-8"))
        origen["eliminado"] = time.time() - pa.PLAZO_PAPELERA_S - 1
        (vieja / "origen.json").write_text(json.dumps(origen), encoding="utf-8")

        resultado = pa.vaciar_papelera(raices=(self.ep,))

        self.assertEqual(resultado, {"ok": True, "vaciadas": [str(vieja)], "fallos": []})
        self.assertFalse(vieja.exists())
        self.assertTrue((nueva / "Banda - Nuevo (2021)" / "01. pista.flac").is_file())

    def test_los_subcomandos_de_la_papelera_llaman_a_sus_pasos(self):
        # Defecto: el servicio lanza `restaurar` y `vaciar_papelera`; sin el subcomando, fallan.
        salida = io.StringIO()
        with mock.patch.object(pa, "restaurar", return_value={"ok": True}) as restaurar, \
                mock.patch.object(pa, "vaciar_papelera", return_value={"ok": True}) as vaciar, \
                contextlib.redirect_stdout(salida):
            self.assertEqual(pa.main(["restaurar", "--papelera=/Papelera/x"]), 0)
            self.assertEqual(pa.main(["vaciar_papelera"]), 0)
        restaurar.assert_called_once_with("/Papelera/x")
        vaciar.assert_called_once_with()


class TestRetomar(unittest.TestCase):
    """Una sesión nueva retoma lo que dejó otra: renders en curso y álbumes sin aprobar."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Un nombre propio, para no confundirse con un render real de la máquina.
        self.proceso = f"render_falso_{self.tmp.name}.py"
        self.parches = [
            mock.patch.object(pa, "PROCESO_RENDER", self.proceso),
            mock.patch.object(pa, "SSD_TEMP_DIR", self.tmp / "temp_render"),
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


class TestCensura(unittest.TestCase):
    """`a_censura`, `censurar` y `aceptar_censura`: la portada censurada de un álbum, con el modelo falso."""

    COLOR = (200, 100, 50)

    def setUp(self):
        TestPreguntarAntesDeRenderizar.setUp(self)
        self.censura = self.tmp / "Censura"
        self.datos = self.tmp / "datos_censura"
        for parche in (
            mock.patch.object(pa, "DIR_CENSURA", self.censura),
            mock.patch.object(pa, "DIR_CENSURA_DATOS", self.datos),
        ):
            parche.start()
            self.addCleanup(parche.stop)

    def album_con_portada(self, nombre, origen=None, con_portada=True):
        carpeta = (origen or self.album) / nombre
        carpeta.mkdir()
        (carpeta / "01. pista.flac").write_bytes(b"audio")
        if con_portada:
            Image.new("RGB", (400, 200), self.COLOR).save(carpeta / "cover.jpg")
        return carpeta

    def en_censura(self, nombre, origen=None, con_portada=True):
        """Un álbum de la cola ya apartado en Censura/, como lo deja `a_censura`."""
        carpeta = self.album_con_portada(nombre, origen, con_portada)
        self.assertTrue(pa.a_censura(carpeta)["ok"])
        return self.censura / nombre

    def censurar(self, carpeta, intento, *respuestas):
        OllamaFalso(list(respuestas)).parchar(self)
        return pa.censurar(carpeta, intento)

    def candidata(self, album):
        return Image.open(self.datos / pa.slug(album) / "candidata.png").convert("RGB")

    # -- a_censura ---------------------------------------------------------------

    def test_a_censura_aparta_el_album_y_buscar_no_lo_vuelve_a_ofrecer_ni_mira_esa_carpeta(self):
        # Defecto: el álbum que se mandó a censurar volvía a salir en la pregunta siguiente.
        primero = self.album_con_portada("Banda A - Primero (2024)")
        self.album_con_portada("Banda B - Segundo (2023)")

        resultado = pa.a_censura(primero)

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["carpeta"], str(self.censura / primero.name))
        self.assertFalse(primero.exists())
        self.assertEqual((self.censura / primero.name / "01. pista.flac").read_bytes(), b"audio")
        self.assertEqual(pa.origen_de_censura(primero.name), primero)
        with mock.patch.object(pa, "capturar_busqueda", return_value=(None, "busqueda")):
            self.assertEqual(pa.buscar()["album"], "Banda B - Segundo (2023)")
            shutil.rmtree(self.album / "Banda B - Segundo (2023)")
            self.assertEqual(pa.buscar()["motivo"], "sin_albumes")

    def test_a_censura_no_mueve_lo_que_no_es_un_album_de_origen_ni_pisa_otro_de_censura(self):
        # Defecto: una ruta mal pasada movería la carpeta madre entera o una parte de un álbum.
        album = self.album_con_portada("Banda - Doble (2020)")
        (album / "CD1").mkdir()
        renderizado = self.tmp / "Renderizando" / "Banda - Listo"
        renderizado.mkdir(parents=True)
        for carpeta in (self.album, album / "CD1", renderizado):
            self.assertEqual(pa.a_censura(carpeta)["motivo"], "carpeta_fuera_de_raiz", carpeta)
        (self.censura / album.name).mkdir(parents=True)

        self.assertEqual(pa.a_censura(album)["motivo"], "ya_existe")

        self.assertTrue((album / "CD1").is_dir())
        self.assertTrue(renderizado.is_dir())
        self.assertEqual(list((self.censura / album.name).iterdir()), [])

    def test_a_censura_repetido_tras_un_reinicio_a_mitad_dice_que_ya_esta_apartado(self):
        # Defecto: el servicio relanza el paso al reiniciarse; si ya movió el álbum, no puede fallar.
        carpeta = self.album_con_portada("Banda - Disco (2020)")
        pa.a_censura(carpeta)

        resultado = pa.a_censura(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["carpeta"], str(self.censura / carpeta.name))
        self.assertTrue((self.censura / carpeta.name / "01. pista.flac").is_file())

    def test_un_censurar_nuevo_empieza_de_cero_pero_conserva_las_originales_guardadas(self):
        # Defecto: las cajas de un Censurar anterior del mismo álbum tapaban de más; y borrar las
        # originales guardadas rompe la promesa de que la original nunca se pierde.
        datos = self.datos / pa.slug("Banda - Disco (2020)")
        (datos / "originales").mkdir(parents=True)
        (datos / "originales" / "vieja.jpg").write_bytes(b"original de antes")
        (datos / "cajas_1.json").write_text("[]", encoding="utf-8")
        (datos / "candidata.png").write_bytes(b"vieja")

        self.en_censura("Banda - Disco (2020)")

        self.assertFalse((datos / "cajas_1.json").exists())
        self.assertFalse((datos / "candidata.png").exists())
        self.assertEqual((datos / "originales" / "vieja.jpg").read_bytes(), b"original de antes")

    # -- censurar ----------------------------------------------------------------

    def test_censurar_pinta_de_negro_la_caja_ampliada_y_no_toca_la_carpeta_del_album(self):
        # Defecto: tocar la carpeta antes de que el usuario acepte; o pintar la caja sin ampliarla.
        carpeta = self.en_censura("Banda - Disco (2020)")
        antes = {p.name: p.read_bytes() for p in carpeta.iterdir()}

        resultado = self.censurar(carpeta, 1, respuesta_con(("penis", (250, 400, 500, 800))))

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["intento"], 1)
        self.assertEqual(resultado["encontradas"], ["pene"])
        self.assertEqual(resultado["zonas"], 1)
        self.assertIs(resultado["descargado"], True)
        # La caja del modelo es (100, 80)-(200, 160); ampliada llega a (50, 40)-(250, 200).
        candidata = self.candidata(carpeta.name)
        self.assertEqual(candidata.getpixel((150, 100)), (0, 0, 0))
        self.assertEqual(candidata.getpixel((55, 45)), (0, 0, 0))
        self.assertEqual(candidata.getpixel((245, 195)), (0, 0, 0))
        self.assertEqual(candidata.getpixel((45, 100)), self.COLOR)
        self.assertEqual(candidata.getpixel((10, 10)), self.COLOR)
        # Las dos vistas para el celular existen, y la original sigue sin censura.
        self.assertEqual(Image.open(resultado["original"]).convert("RGB").getpixel((150, 100)), self.COLOR)
        self.assertEqual(Image.open(resultado["censurada"]).convert("RGB").getpixel((150, 100)), (0, 0, 0))
        self.assertEqual({p.name: p.read_bytes() for p in carpeta.iterdir()}, antes)

    def test_un_intento_nuevo_suma_sus_cajas_a_las_de_los_anteriores(self):
        # Defecto: cada intento empezaba de cero y la censura podía encogerse al volver a analizar.
        carpeta = self.en_censura("Banda - Disco (2020)")
        self.censurar(carpeta, 1, respuesta_con(("penis", (50, 50, 100, 150))))

        # El intento 2 mira cuatro pedazos de 240 x 120; el último, abajo a la derecha, nace en (160, 80).
        resultado = self.censurar(carpeta, 2, "[]", "[]", "[]", respuesta_con(("anus", (0, 0, 1000, 1000))))

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["encontradas"], ["ano"])
        candidata = self.candidata(carpeta.name)
        self.assertEqual(candidata.getpixel((300, 150)), (0, 0, 0))
        # Lo del intento 1 sigue tapado: caja (20, 10)-(40, 30), ampliada (10, 5)-(50, 35).
        self.assertEqual(candidata.getpixel((30, 20)), (0, 0, 0))
        # Y no se tapa de más: lo que ninguna caja alcanza sigue como estaba.
        self.assertEqual(candidata.getpixel((5, 150)), self.COLOR)

    def test_repetir_el_mismo_intento_reemplaza_sus_cajas_y_no_las_duplica(self):
        # Defecto: un análisis relanzado tras un reinicio sumaba dos veces el mismo intento.
        carpeta = self.en_censura("Banda - Disco (2020)")
        self.censurar(carpeta, 1, respuesta_con(("penis", (0, 0, 100, 100))))

        resultado = self.censurar(carpeta, 1, respuesta_con(("vulva", (900, 900, 1000, 1000))))

        self.assertEqual(resultado["zonas"], 1)
        candidata = self.candidata(carpeta.name)
        self.assertEqual(candidata.getpixel((395, 195)), (0, 0, 0))
        self.assertEqual(candidata.getpixel((5, 5)), self.COLOR)

    def test_si_no_encuentra_nada_lo_dice_y_no_deja_una_candidata(self):
        # Defecto: dar por censurada una portada sin zonas, o dejar la candidata de un intento anterior.
        carpeta = self.en_censura("Banda - Disco (2020)")

        resultado = self.censurar(carpeta, 1, "[]")

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual((resultado["encontradas"], resultado["zonas"]), ([], 0))
        self.assertFalse((self.datos / pa.slug(carpeta.name) / "candidata.png").exists())
        # Un análisis repetido que ya no encuentra nada no deja la candidata del anterior.
        self.censurar(carpeta, 1, respuesta_con(("penis", (0, 0, 100, 100))))
        self.assertTrue((self.datos / pa.slug(carpeta.name) / "candidata.png").is_file())
        self.censurar(carpeta, 1, "[]")
        self.assertFalse((self.datos / pa.slug(carpeta.name) / "candidata.png").exists())

    def test_el_resultado_dice_cuantos_pedazos_no_se_pudieron_leer(self):
        # Defecto: el servicio no se enteraba de los pedazos sin leer y daba la portada por limpia.
        carpeta = self.en_censura("Banda - Disco (2020)")

        resultado = self.censurar(carpeta, 2, "no puedo", "[]", "[]", "[]")

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual((resultado["pedazos"], resultado["sin_leer"]), (4, 1))

    def test_si_el_analisis_falla_dice_por_que_descarga_el_modelo_y_no_anota_el_intento(self):
        # Defecto: un fallo que no descarga el modelo deja 17 GB en la GPU; y uno que anota el
        # intento hace que «Volver a analizar» mire con más cuidado sin haber mirado.
        carpeta = self.en_censura("Banda - Disco (2020)")
        ollama = OllamaFalso([requests.exceptions.ConnectionError("rechazada")]).parchar(self)

        resultado = pa.censurar(carpeta, 1)

        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "ollama_caido"))
        self.assertIn("descargado", resultado)
        self.assertTrue(any(ruta == "/api/generate" for ruta, _, _ in ollama.pedidos))
        self.assertEqual(list((self.datos / pa.slug(carpeta.name)).glob("cajas_*.json")), [])

    def test_si_el_modelo_se_queda_cargado_el_resultado_lo_avisa(self):
        # Defecto: callar que Qwen sigue en la GPU; el próximo render falla por memoria sin aviso.
        carpeta = self.en_censura("Banda - Disco (2020)")
        ollama = OllamaFalso(["[]"], ignora_la_descarga=True).parchar(self)
        cliente = cp.ClienteOllama(host="http://ollama.test", pausa=lambda s: None, espera_descarga_s=2)
        with mock.patch.object(cp.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            resultado = pa.censurar(carpeta, 1, cliente=cliente)
        self.assertTrue(resultado["ok"], resultado)
        self.assertTrue(ollama.cargado)
        self.assertIs(resultado["descargado"], False)

    def test_sin_portada_en_la_carpeta_no_llama_al_modelo(self):
        # Defecto: gastar la GPU en un álbum sin imagen que analizar.
        carpeta = self.en_censura("Banda - Disco (2020)", con_portada=False)
        ollama = OllamaFalso().parchar(self)

        resultado = pa.censurar(carpeta, 1)

        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "sin_portada"))
        self.assertEqual(ollama.pedidos, [])

    def test_con_un_render_corriendo_no_analiza_ni_toca_la_gpu(self):
        # Defecto: dos programas pidiendo VRAM a la vez hacen fallar el render (pasó el 2026-11-03).
        carpeta = self.en_censura("Banda - Disco (2020)")
        ollama = OllamaFalso().parchar(self)

        with mock.patch.object(pa, "render_en_curso", return_value=True):
            resultado = pa.censurar(carpeta, 1)

        self.assertEqual(resultado, {"ok": False, "motivo": "render_en_curso"})
        self.assertEqual(ollama.pedidos, [])

    def test_censurar_solo_analiza_albumes_de_censura(self):
        # Defecto: un `--carpeta` mal pasado analizaría (y escribiría datos de) cualquier carpeta.
        carpeta = self.album_con_portada("Banda - Disco (2020)")
        ollama = OllamaFalso().parchar(self)

        resultado = pa.censurar(carpeta, 1)

        self.assertEqual(resultado["motivo"], "carpeta_fuera_de_censura")
        self.assertEqual(ollama.pedidos, [])
        self.assertFalse(self.datos.exists())

    # -- aceptar_censura ---------------------------------------------------------

    def censurado(self, nombre="Banda - Disco (2020)", origen=None, respuesta=None):
        """Un álbum en Censura/ con un intento 1 ya analizado."""
        carpeta = self.en_censura(nombre, origen)
        self.censurar(carpeta, 1, respuesta or respuesta_con(("penis", (250, 400, 500, 800))))
        return carpeta

    def test_aceptar_deja_la_censurada_como_portada_guarda_la_original_y_devuelve_el_album(self):
        # Defecto: reemplazar la portada sin guardar la original; o dejar el álbum en Censura/.
        carpeta = self.censurado()
        original = (carpeta / "cover.jpg").read_bytes()
        destino = self.album / carpeta.name

        resultado = pa.aceptar_censura(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        self.assertFalse(carpeta.exists())
        self.assertEqual(resultado["carpeta"], str(destino))
        elegida = pa.elegir_portada(destino)
        self.assertEqual(elegida.name, "cover.png")
        self.assertEqual(Image.open(elegida).convert("RGB").getpixel((150, 100)), (0, 0, 0))
        guardadas = list((self.datos / pa.slug(destino.name) / "originales").iterdir())
        self.assertEqual([p.read_bytes() for p in guardadas], [original])
        self.assertEqual((destino / "01. pista.flac").read_bytes(), b"audio")
        # Y `buscar` lo vuelve a ofrecer: de ahí sale el render con la portada censurada.
        with mock.patch.object(pa, "capturar_busqueda", return_value=(None, "busqueda")):
            ofrecido = pa.buscar()
        self.assertEqual(ofrecido["album"], destino.name)
        self.assertEqual(ofrecido["portada"], str(elegida))

    def test_aceptar_sin_nada_que_tapar_devuelve_el_album_con_su_portada_como_esta(self):
        # Defecto: cambiar la portada (o fallar) cuando el análisis no encontró nada.
        carpeta = self.censurado(respuesta="[]")

        resultado = pa.aceptar_censura(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        devuelto = Path(resultado["carpeta"])
        self.assertEqual(devuelto.parent, self.album)
        self.assertEqual(sorted(p.name for p in devuelto.iterdir()), ["01. pista.flac", "cover.jpg"])

    def test_aceptar_un_album_que_nunca_se_analizo_no_mueve_nada(self):
        # Defecto: aceptar sin haber analizado devolvería el álbum a la cola creyéndolo censurado.
        carpeta = self.en_censura("Banda - Disco (2020)")

        resultado = pa.aceptar_censura(carpeta)

        self.assertEqual(resultado["motivo"], "sin_analisis")
        self.assertTrue(carpeta.is_dir())

    def test_aceptar_no_pisa_una_carpeta_que_ya_esta_en_el_origen_y_no_toca_la_portada(self):
        # Defecto: devolver el álbum encima de otra copia la pisa; y cambiar la portada antes de
        # saber que puede devolverlo deja el álbum a medias.
        carpeta = self.censurado()
        (self.album / carpeta.name).mkdir()
        antes = sorted(p.name for p in carpeta.iterdir())

        resultado = pa.aceptar_censura(carpeta)

        self.assertEqual(resultado["motivo"], "ya_existe")
        self.assertEqual(sorted(p.name for p in carpeta.iterdir()), antes)
        self.assertEqual(list((self.album / carpeta.name).iterdir()), [])

    def test_aceptar_repetido_tras_un_reinicio_no_vuelve_a_cambiar_la_portada(self):
        # Defecto: relanzar el paso guardaría la censurada como si fuera la original.
        carpeta = self.censurado()
        primero = pa.aceptar_censura(carpeta)

        segundo = pa.aceptar_censura(carpeta)

        self.assertTrue(segundo["ok"], segundo)
        self.assertEqual(segundo["carpeta"], primero["carpeta"])
        self.assertEqual(len(list((self.datos / pa.slug(carpeta.name) / "originales").iterdir())), 1)

    def test_si_no_puede_devolver_el_album_el_reintento_no_guarda_la_censurada_como_original(self):
        # Defecto: tras cambiar la portada y fallar al mover el álbum, el reintento tomaba la
        # censurada por la portada del álbum y la guardaba como «original».
        carpeta = self.censurado()
        with mock.patch.object(Path, "rename", side_effect=OSError("disco lleno")):
            fallo = pa.aceptar_censura(carpeta)
        self.assertEqual(fallo["motivo"], "no_se_pudo_aplicar")
        self.assertEqual(pa.elegir_portada(carpeta).name, "cover.png")

        resultado = pa.aceptar_censura(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        guardadas = list((self.datos / pa.slug(carpeta.name) / "originales").iterdir())
        self.assertEqual(len(guardadas), 1)
        self.assertEqual(Image.open(guardadas[0]).convert("RGB").getpixel((150, 100)), self.COLOR)

    def test_tras_un_aceptar_fallido_y_otro_analisis_el_aceptar_pone_la_candidata_nueva(self):
        # Defecto: el primer Aceptar cambió la portada y no pudo mover el álbum; su marca hacía
        # saltar la candidata del análisis siguiente, y el render salía sin las zonas nuevas que el
        # usuario aprobó. Además la lista de originales quedaba vacía.
        carpeta = self.censurado()
        with mock.patch.object(Path, "rename", side_effect=OSError("disco lleno")):
            self.assertEqual(pa.aceptar_censura(carpeta)["motivo"], "no_se_pudo_aplicar")
        # El intento 2 mira cuatro pedazos de 240 x 120; el último, abajo a la derecha, nace en (160, 80).
        self.censurar(carpeta, 2, "[]", "[]", "[]", respuesta_con(("anus", (0, 0, 1000, 1000))))

        resultado = pa.aceptar_censura(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        portada = Image.open(pa.elegir_portada(Path(resultado["carpeta"]))).convert("RGB")
        self.assertEqual(portada.getpixel((300, 150)), (0, 0, 0))
        self.assertEqual(portada.getpixel((150, 100)), (0, 0, 0))
        originales = [Image.open(o).convert("RGB").getpixel((150, 100)) for o in resultado["originales"]]
        self.assertIn(self.COLOR, originales)

    # -- eliminar y deshacer desde Censura/ ---------------------------------------

    def test_eliminar_desde_censura_lleva_la_original_a_la_papelera_y_deshacer_vuelve_al_origen(self):
        # Defecto: Deshacer devolvía el álbum a Censura/, donde nada lo vuelve a preguntar.
        carpeta = self.en_censura("Banda - Disco (2020)", self.ep)

        resultado = pa.eliminar(carpeta)

        self.assertTrue(resultado["ok"], resultado)
        entrada = Path(resultado["papelera"])
        self.assertFalse(carpeta.exists())
        self.assertTrue((entrada / carpeta.name / "cover.jpg").is_file())
        self.assertEqual(rp.leer(pa.REGISTRO_PUBLICADOS).buscar(carpeta.name)["origen"], "eliminar")
        self.assertFalse((self.datos / pa.slug(carpeta.name)).exists())

        devuelto = pa.restaurar(entrada)

        self.assertTrue(devuelto["ok"], devuelto)
        self.assertEqual(devuelto["carpeta"], str(self.ep / carpeta.name))
        with mock.patch.object(pa, "capturar_busqueda", return_value=(None, "busqueda")):
            self.assertEqual(pa.buscar()["album"], carpeta.name)

    def test_eliminar_sigue_sin_aceptar_carpetas_que_no_son_un_album_entero(self):
        # Defecto: sumar Censura/ como raíz no puede abrir la puerta a carpetas de adentro.
        carpeta = self.en_censura("Banda - Disco (2020)")
        (carpeta / "CD1").mkdir()

        self.assertEqual(pa.eliminar(carpeta / "CD1")["motivo"], "carpeta_fuera_de_raiz")
        self.assertEqual(pa.eliminar(self.censura)["motivo"], "carpeta_fuera_de_raiz")

    # -- GPU: un render no arranca mientras corre un análisis ---------------------

    def test_renderizar_no_arranca_si_hay_un_analisis_corriendo(self):
        # Defecto: render y Qwen pidiendo la GPU a la vez; el render se queda sin memoria.
        self.album_con_portada("Banda - Disco (2020)")
        with mock.patch.object(pa, "analisis_en_curso", return_value=True), mock.patch.object(
            pa.subprocess, "run", side_effect=AssertionError("lanzó un render")
        ):
            resultado = pa.renderizar()
        self.assertEqual(resultado, {"ok": False, "motivo": "analisis_en_curso"})

    def test_analisis_en_curso_ve_el_subcomando_censurar_de_este_script_y_ningun_otro(self):
        # Defecto: sin verlo, un análisis lanzado por otra sesión no frena el render.
        script = f"pipeline_falso_{self.tmp.name}.py"
        procesos = []
        self.addCleanup(lambda: [(p.kill(), p.wait()) for p in procesos])

        def lanzar(*argumentos):
            procesos.append(subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)", str(self.tmp / script), *argumentos]
            ))

        with mock.patch.object(pa, "SCRIPT_PIPELINE", script):
            self.assertFalse(ANALISIS_EN_CURSO_REAL())
            lanzar("renderizar", "--carpeta", "censurar")
            self.assertFalse(ANALISIS_EN_CURSO_REAL())
            lanzar("censurar", "--carpeta", "x", "--intento", "1")
            self.assertTrue(ANALISIS_EN_CURSO_REAL())

    # -- la línea de comandos ----------------------------------------------------

    def test_los_subcomandos_de_censura_llaman_a_sus_pasos_con_el_intento_como_numero(self):
        # Defecto: el servicio los lanza por línea de comandos; un intento que llega como texto
        # rompe la cuenta de pedazos.
        salida = io.StringIO()
        with mock.patch.object(pa, "a_censura", return_value={"ok": True}) as a_censura, \
                mock.patch.object(pa, "censurar", return_value={"ok": True}) as censurar, \
                mock.patch.object(pa, "aceptar_censura", return_value={"ok": True}) as aceptar, \
                contextlib.redirect_stdout(salida):
            self.assertEqual(pa.main(["a_censura", "--carpeta=/o/Banda"]), 0)
            self.assertEqual(pa.main(["censurar", "--carpeta=/c/Banda", "--intento", "3"]), 0)
            self.assertEqual(pa.main(["aceptar_censura", "--carpeta=/c/Banda"]), 0)
        a_censura.assert_called_once_with("/o/Banda")
        censurar.assert_called_once_with("/c/Banda", 3)
        aceptar.assert_called_once_with("/c/Banda")


class ClienteDeMiniaturas:
    """Cliente de YouTube falso: anota cada thumbnails().set() o lanza el error pedido."""

    def __init__(self, error=None):
        self.puestas = []
        self.error = error

    def thumbnails(self):
        return self

    def set(self, videoId, media_body):
        self.puestas.append((videoId, media_body))
        return self

    def execute(self):
        if self.error:
            raise self.error
        return {}


def media_falsa(ruta, mimetype):
    # Como MediaFileUpload, abre el archivo al crearse: falla si ya se borró.
    return {"bytes": Path(ruta).read_bytes(), "mimetype": mimetype}


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
            mock.patch.object(pa, "REGISTRO_PUBLICADOS", self.tmp / "ya_publicados.tsv"),
            mock.patch.object(pa, "MediaFileUpload", media_falsa),
            mock.patch.object(pa, "sondear", lambda ruta: None),
        ]
        for parche in self.parches:
            parche.start()
        self.subidas = []
        self.youtube = ClienteDeMiniaturas()
        self.captura = self.tmp / "capturas" / f"{pa.slug(self.carpeta.name)}.jpg"

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
            probar_credenciales_disponibles=lambda prefix: (self.youtube, [], [], []),
            encontrar_video=lambda carpeta: next(carpeta.glob("*.mp4"), None),
            cargar_generos=lambda: ([], {}),
            cargar_repertorio=lambda: {},
            procesar_carpeta=lambda *a: {"title": "Banda - Album", "description": "D"},
            subir_video=subir_video,
            authenticate_next=lambda prefix: None,
            is_invalid_grant_error=lambda exc: False,
            is_quota_error=lambda exc: False,
            safe_delete_folder=borrar,
        )

    def subir_bien(self, subir_video=None):
        def subir_video_ok(youtube, video, titulo, descripcion, **kwargs):
            return {"id": "abc123"}

        return pa.subir(
            self.carpeta,
            uploader=self.uploader(subir_video or subir_video_ok),
            sin_deathgrind=True,
            raices=(self.raiz,),
        )

    def test_pone_de_miniatura_el_fotograma_del_medio_que_saco_verificar(self):
        # Defecto: sin esto YouTube elige un fotograma del intro como miniatura.
        self.captura.parent.mkdir(parents=True)
        self.captura.write_bytes(b"fotograma del medio")

        resultado = self.subir_bien()

        self.assertTrue(resultado["ok"], resultado)
        self.assertTrue(resultado["miniatura"], resultado)
        # Se lee antes de borrar el álbum, que también borra la captura.
        self.assertEqual(
            self.youtube.puestas,
            [("abc123", {"bytes": b"fotograma del medio", "mimetype": "image/jpeg"})],
        )
        self.assertFalse(self.captura.exists())

    def test_sin_la_captura_de_verificar_saca_el_fotograma_del_medio(self):
        # Defecto: una subida a mano, sin `verificar` antes, queda con el intro de miniatura.
        generadas = []

        def generar_captura(video, duracion, destino):
            generadas.append((Path(video).name, duracion, destino))
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"nuevo")
            return destino

        with mock.patch.object(pa, "sondear", lambda ruta: {"format": {"duration": "600.0"}}), \
                mock.patch.object(pa, "generar_captura", generar_captura):
            resultado = self.subir_bien()

        self.assertTrue(resultado["miniatura"], resultado)
        self.assertEqual(generadas, [("Banda - Album.mp4", 600.0, self.captura)])
        self.assertEqual(self.youtube.puestas[0][1]["bytes"], b"nuevo")

    def test_si_youtube_rechaza_la_miniatura_el_video_queda_subido_igual(self):
        # Defecto: el error se llevaba el resultado y el álbum, ya subido, se volvía a subir.
        self.captura.parent.mkdir(parents=True)
        self.captura.write_bytes(b"jpg")
        self.youtube.error = HttpError(httplib2.Response({"status": 403}), b"forbidden")

        resultado = self.subir_bien()

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(resultado["video_id"], "abc123")
        self.assertTrue(resultado["carpeta_borrada"])
        self.assertFalse(self.carpeta.exists())
        self.assertFalse(resultado["miniatura"])
        self.assertIn("403", resultado["miniatura_error"])

    def test_sin_fotograma_no_se_pone_miniatura_y_se_avisa(self):
        # Defecto: sin fotograma se daba la miniatura por puesta y no llegaba el aviso.
        resultado = self.subir_bien()

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(self.youtube.puestas, [])
        self.assertFalse(resultado["miniatura"])
        self.assertTrue(resultado["miniatura_error"])

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
        # Defecto: el álbum subido no queda anotado y se vuelve a descargar o renderizar.
        fila = rp.leer(pa.REGISTRO_PUBLICADOS).buscar("Banda - Album (2020) [EP]")
        self.assertEqual((fila["origen"], fila["video_id"]), ("subir", "abc123"))

    def test_limite_del_canal_conserva_la_carpeta(self):
        uploader = None

        def subir_video(*args, **kwargs):
            raise uploader.UploadLimitExceededError("limite")

        uploader = self.uploader(subir_video)
        resultado = pa.subir(self.carpeta, uploader=uploader, sin_deathgrind=True, raices=(self.raiz,))
        self.assertEqual(resultado["motivo"], "limite_canal")
        self.assertTrue(self.carpeta.exists())
        self.assertFalse(pa.PENDIENTES_JSON.exists())
        # Defecto: anotarlo antes de que la subida funcione aparta un álbum que nunca se publicó.
        self.assertFalse(pa.REGISTRO_PUBLICADOS.exists())

    def test_eliminar_aparta_un_album_renderizado_y_borra_su_captura(self):
        captura = pa.DIR_CAPTURAS / f"{pa.slug(self.carpeta.name)}.jpg"
        captura.parent.mkdir(parents=True)
        captura.write_bytes(b"jpg")

        resultado = pa.eliminar(self.carpeta, raices=(self.raiz,))

        self.assertTrue(resultado["ok"], resultado)
        self.assertFalse(self.carpeta.exists())
        self.assertTrue((Path(resultado["papelera"]) / "Banda - Album" / "Banda - Album.mp4").is_file())
        self.assertFalse(captura.exists())
        # Defecto: el álbum eliminado por ya publicado no queda anotado y se vuelve a renderizar.
        fila = rp.leer(pa.REGISTRO_PUBLICADOS).buscar("Banda - Album")
        self.assertEqual((fila["nombre"], fila["origen"]), ("Banda - Album", "eliminar"))

    def test_eliminar_fuera_de_las_raices_no_borra_nada(self):
        afuera = self.tmp / "otra" / "Banda - Album"
        shutil.copytree(self.carpeta, afuera)

        resultado = pa.eliminar(afuera, raices=(self.raiz,))

        self.assertEqual(resultado["motivo"], "carpeta_fuera_de_raiz")
        self.assertTrue(afuera.exists())
        self.assertFalse(pa.REGISTRO_PUBLICADOS.exists())

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


class PaginaFalsa:
    url = "https://studio.youtube.com/"

    def __init__(self, falla_al_abrir=False):
        self.falla_al_abrir = falla_al_abrir

    def goto(self, url, **kwargs):
        if self.falla_al_abrir:
            raise playwright_api.Error("net::ERR_INTERNET_DISCONNECTED")

    def wait_for_load_state(self, *args, **kwargs):
        pass

    def screenshot(self, path):
        pass


class ContextoFalso:
    def __init__(self, cierra_mal=False, pagina=None):
        self.cierra_mal = cierra_mal
        self.pages = [pagina or PaginaFalsa()]

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

    def test_si_studio_no_carga_devuelve_fallo(self):
        contexto = ContextoFalso(pagina=PaginaFalsa(falla_al_abrir=True))
        resultado = self.activar(PlaywrightQueArranca(), contexto)
        self.assertEqual((resultado["estado"], resultado["motivo"]), ("fallo", "error_navegador"))

    def test_sin_ejecutable_del_navegador_se_usa_la_version_por_defecto(self):
        self.assertEqual(pa.version_del_navegador(None), pa.VERSION_STUDIO_POR_DEFECTO)


URL_SELECTOR_DE_CANAL = (
    "https://www.youtube.com/signin_prompt?app=desktop&next=https%3A%2F%2Fstudio.youtube.com%2F"
)


class StudioConVariosCanales:
    """Studio que, como el de verdad, a veces manda a "Selecciona un canal"."""

    def __init__(self, pide_canal, canales=("Daniel Banariba Drummer", "Daniel Banariba", "Alejandro Banariba")):
        self.pide_canal = pide_canal
        self.canales = canales
        self.elegidos = []

    def studio(self, ruta):
        if self.pide_canal:
            # Un 302 no pasa por la ruta del otro dominio: se redirige desde la página.
            cuerpo = f'<script>location.replace("{URL_SELECTOR_DE_CANAL}")</script>'
        else:
            cuerpo = "<h1>Panel del canal</h1>"
        ruta.fulfill(status=200, body=cuerpo, content_type="text/html; charset=utf-8")

    def youtube(self, ruta):
        url = urlparse(ruta.request.url)
        if url.path == "/elegir":
            self.elegidos.append(parse_qs(url.query)["canal"][0])
            self.pide_canal = False
            cuerpo = '<script>location.replace("https://studio.youtube.com/")</script>'
        else:
            filas = "".join(
                f'<div onclick="location.href = \'/elegir?canal={quote(canal)}\'">'
                f"<img alt=''><div><span>{canal}</span><span>8010 suscriptores</span></div></div>"
                for canal in self.canales
            )
            # Detrás del diálogo queda la página de configuración, con el nombre de la cuenta oculto.
            cuerpo = (
                "<nav>Configuración</nav><span hidden>Daniel Banariba</span>"
                f"<div role='dialog'><h2>Selecciona un canal</h2>{filas}"
                "<label><input type='checkbox'> No volver a preguntar</label></div>"
            )
        ruta.fulfill(status=200, body=cuerpo, content_type="text/html; charset=utf-8")


class TestElegirCanalDeStudio(unittest.TestCase):
    """La cuenta tiene varios canales: los pasos de Studio tienen que entrar con "Daniel Banariba"."""

    def setUp(self):
        self.playwright = playwright_api.sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        try:
            self.navegador = self.playwright.chromium.launch(headless=True)
        except playwright_api.Error as exc:
            self.skipTest(f"sin Chromium: {exc}")
        self.addCleanup(self.navegador.close)

    def abrir_studio(self, youtube):
        contexto = self.navegador.new_context()
        contexto.route("https://studio.youtube.com/**", youtube.studio)
        contexto.route("https://www.youtube.com/**", youtube.youtube)
        with mock.patch.object(pa, "preparar_perfil_studio", return_value=("/perfil", None, "Default")), \
                mock.patch.object(playwright_api, "sync_playwright", return_value=PlaywrightQueArranca()), \
                mock.patch.object(pa, "abrir_contexto_studio", return_value=contexto), \
                mock.patch.object(pa, "ESPERA_CANAL_MS", 2000):
            with pa.pagina_de_studio() as pagina:
                return pagina.url

    def test_si_youtube_pide_elegir_canal_elige_daniel_banariba_y_sigue_a_studio(self):
        youtube = StudioConVariosCanales(pide_canal=True)
        url = self.abrir_studio(youtube)
        self.assertEqual(youtube.elegidos, ["Daniel Banariba"])
        self.assertTrue(url.startswith("https://studio.youtube.com/"), url)

    def test_sin_la_pregunta_entra_a_studio_sin_elegir_nada(self):
        youtube = StudioConVariosCanales(pide_canal=False)
        url = self.abrir_studio(youtube)
        self.assertEqual(youtube.elegidos, [])
        self.assertTrue(url.startswith("https://studio.youtube.com/"), url)

    def test_si_el_canal_no_esta_en_la_lista_falla_sin_elegir_otro(self):
        youtube = StudioConVariosCanales(pide_canal=True, canales=("Daniel Banariba Drummer", "Alejandro Banariba"))
        with self.assertRaises(pa.NavegadorNoDisponible) as error:
            self.abrir_studio(youtube)
        self.assertEqual(error.exception.motivo, "canal_no_disponible")
        self.assertEqual(youtube.elegidos, [])


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


class SubidasPaginadas:
    """La lista de subidas del canal en páginas: [(título, video_id), ...] por página."""

    def __init__(self, paginas, falla_en=None):
        self.paginas = paginas
        self.falla_en = falla_en
        self.pedidas = []

    def list(self, part, playlistId, maxResults, pageToken=None):
        numero = int(pageToken or 0)
        self.pedidas.append(numero)
        if numero == self.falla_en:
            raise HttpError(httplib2.Response({"status": "500"}), b"backend error")
        respuesta = {
            "items": [
                {"snippet": {"title": titulo, "resourceId": {"videoId": video_id}}}
                for titulo, video_id in self.paginas[numero]
            ]
        }
        if numero + 1 < len(self.paginas):
            respuesta["nextPageToken"] = str(numero + 1)
        return Llamada(respuesta)


class TestSembrarPublicados(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        parches = [
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
            mock.patch.object(pa, "DIR_AGENTE", self.tmp),
            mock.patch.object(pa, "REGISTRO_PUBLICADOS", self.tmp / "ya_publicados.tsv"),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)
        pa.DIR_LOGS.mkdir()
        resultados = {
            "servicio_eliminar_20261102-101010_aaaaaa.out": {"ok": True, "album": "Banda C - Tres (2019)"},
            "servicio_eliminar_20261102-111111_bbbbbb.out": {
                "ok": False, "motivo": "no_se_pudo_borrar", "album": "Banda D - Cuatro (2018)"},
            "servicio_eliminar_video_20261102-121212_cccccc.out": {
                "ok": True, "video_id": "vidZ", "titulo": "Banda Z - Borrado (2017)"},
        }
        for nombre, resultado in resultados.items():
            (pa.DIR_LOGS / nombre).write_text(json.dumps(resultado) + "\n", encoding="utf-8")
        (self.tmp / "servicio.json").write_text(
            json.dumps({"subidos": ["Banda E - Cinco (2021)"], "rechazados": ["Banda F - Seis (2016)"]}),
            encoding="utf-8",
        )

    def sembrar(self, subidas):
        youtube = SimpleNamespace(playlistItems=lambda: subidas)
        return pa.sembrar_publicados(uploader=uploader_para_programar(youtube))

    def canal(self, falla_en=None):
        return SubidasPaginadas([
            [("🇺🇸 BANDA A - UNO (2020) • [Death Metal] ⟨FULL ALBUM⟩", "vidA"), ("Deleted video", "vidX")],
            [("Banda B - Dos (Full Album)", "vidB")],
        ], falla_en=falla_en)

    def test_carga_todas_las_paginas_del_canal_los_eliminados_y_los_subidos(self):
        # Defecto: leer solo la primera página del canal, o no cargar lo que el servicio ya
        # eliminó o subió; también, anotar un eliminar que falló, un video borrado o un rechazo.
        subidas = self.canal()
        resultado = self.sembrar(subidas)

        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(subidas.pedidas, [0, 1])
        self.assertEqual(resultado["agregados"], 4)
        registro = rp.leer(pa.REGISTRO_PUBLICADOS)
        esperado = {
            "Banda A - Uno (2020)": ("canal", "vidA"),
            "Banda B - Dos (2015) [EP]": ("canal", "vidB"),
            "Banda C - Tres (2019)": ("eliminar", ""),
            "Banda E - Cinco (2021)": ("subir", ""),
        }
        for album, (origen, video_id) in esperado.items():
            with self.subTest(album=album):
                fila = registro.buscar(album)
                self.assertEqual((fila["origen"], fila["video_id"]), (origen, video_id))
        for album in ("Banda D - Cuatro (2018)", "Banda Z - Borrado (2017)", "Banda F - Seis (2016)"):
            with self.subTest(album=album):
                self.assertIsNone(registro.buscar(album))

    def test_sembrar_otra_vez_no_duplica_ni_repite_un_video_ya_anotado(self):
        # Defecto: cada corrida vuelve a agregar las mismas filas, o el video ya anotado por
        # Subir se anota de nuevo con el título del canal.
        rp.anotar(pa.REGISTRO_PUBLICADOS, [("Banda A - Uno (2020)", "subir", "vidA")])

        primera = self.sembrar(self.canal())
        segunda = self.sembrar(self.canal())

        self.assertEqual((primera["agregados"], segunda["agregados"]), (3, 0))
        self.assertEqual(len(rp.leer(pa.REGISTRO_PUBLICADOS).filas), 4)

    def test_no_vuelve_a_anotar_un_album_devuelto_con_deshacer(self):
        # Defecto: el log de `eliminar` seguía ahí y sembrar volvía a anotar el álbum que el
        # usuario devolvió; el próximo `buscar` lo apartaba a Ya publicados/.
        resultados = {
            "servicio_restaurar_20261102-102020_dddddd.out": {"ok": True, "album": "Banda C - Tres (2019)"},
            "servicio_eliminar_20261102-103030_eeeeee.out": {"ok": True, "album": "Banda G - Siete (2015)"},
            "servicio_restaurar_20261102-104040_ffffff.out": {"ok": True, "album": "Banda G - Siete (2015)"},
            # Devuelto y eliminado otra vez: lo último que pasó es el eliminar.
            "servicio_eliminar_20261102-105050_abcdef.out": {"ok": True, "album": "Banda G - Siete (2015)"},
        }
        for nombre, resultado in resultados.items():
            (pa.DIR_LOGS / nombre).write_text(json.dumps(resultado) + "\n", encoding="utf-8")

        self.sembrar(self.canal())

        registro = rp.leer(pa.REGISTRO_PUBLICADOS)
        self.assertIsNone(registro.buscar("Banda C - Tres (2019)"))
        self.assertEqual(registro.buscar("Banda G - Siete (2015)")["origen"], "eliminar")

    def test_un_error_de_la_api_se_informa_y_no_anota_nada(self):
        # Defecto: la corrida termina con un traceback sin JSON, o informa ok con medio canal.
        resultado = self.sembrar(self.canal(falla_en=1))

        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "error_api"))
        self.assertFalse(pa.REGISTRO_PUBLICADOS.exists())


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


    def programar_un_video_que_la_api_no_ve(self, retiro):
        """`programar` de un video que `videos.list` no devuelve; `retiro` es lo que lee Studio."""
        registro = self.tmp / "ya_publicados.tsv"
        pa.registrar_pendiente("V1", "🇩🇪 Banda - Disco (2025)", "Banda - Disco (2025)", datetime.now().astimezone())
        bloqueado_al_leer = []

        def leer_retiro(video_id):
            # El perfil de Studio se comparte: la lectura va dentro del candado de `programar`.
            with open(self.tmp / "programar.lock", "w", encoding="utf-8") as candado:
                try:
                    fcntl.flock(candado, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    bloqueado_al_leer.append(False)
                except BlockingIOError:
                    bloqueado_al_leer.append(True)
            return retiro

        youtube = SimpleNamespace(videos=lambda: SimpleNamespace(list=lambda part, id: Llamada({"items": []})))
        with mock.patch.object(pa, "REGISTRO_PUBLICADOS", registro):
            resultado = pa.programar(
                "V1", esperar=True, leer_retiro=leer_retiro, uploader=uploader_para_programar(youtube)
            )
        return resultado, registro, bloqueado_al_leer

    def test_una_copia_de_un_video_retirado_sale_de_pendientes_y_su_album_queda_registrado(self):
        # Defecto: dejarlo en pendientes (se relanza en cada reinicio y nunca se publica), no
        # anotar el álbum (se vuelve a renderizar y a subir) o perder lo que dijo Studio.
        retiro = {
            "estado": "copia_retirada",
            "detalle": ["De: Naked Whipper Official"],
            "captura_studio": "/tmp/retiro_V1.png",
        }
        resultado, registro, bloqueado = self.programar_un_video_que_la_api_no_ve(retiro)
        self.assertEqual(
            {k: resultado.get(k) for k in ("ok", "motivo", "video_id", "titulo", "detalle", "captura_studio")},
            {
                "ok": False,
                "motivo": "copia_retirada",
                "video_id": "V1",
                "titulo": "🇩🇪 Banda - Disco (2025)",
                "detalle": ["De: Naked Whipper Official"],
                "captura_studio": "/tmp/retiro_V1.png",
            },
        )
        self.assertEqual(bloqueado, [True])
        self.assertIs(resultado["album_registrado"], True)
        self.assertEqual(pa.leer_pendientes(), [])
        filas = pa.registro_publicados.leer(registro).filas
        self.assertEqual([(f["nombre"], f["origen"], f["video_id"]) for f in filas], [("Banda - Disco (2025)", "subir", "V1")])

    def test_una_copia_sin_entrada_en_pendientes_no_dice_que_anoto_el_album(self):
        # Defecto: sin entrada en pendientes (un video liberado de un reclamo se reprograma sin
        # estar ahí) no hay álbum que anotar, pero el aviso aseguraba que quedó en el registro.
        registro = self.tmp / "ya_publicados.tsv"
        retiro = {"estado": "copia_retirada", "detalle": [], "captura_studio": None}
        with mock.patch.object(pa, "REGISTRO_PUBLICADOS", registro):
            resultado = pa.apartar_copia_retirada("V9", "Banda - Disco", retiro, self.tmp / "programar.out")
        self.assertIs(resultado["album_registrado"], False)
        self.assertFalse(registro.exists())

    def test_una_copia_de_un_album_ya_registrado_no_repite_la_fila(self):
        # Defecto: una fila por cada vez que se mira el video (el subir ya lo había anotado).
        registro = self.tmp / "ya_publicados.tsv"
        pa.registro_publicados.anotar(registro, [("Banda - Disco (2025)", "subir", "V1")])
        resultado, registro, _ = self.programar_un_video_que_la_api_no_ve(
            {"estado": "copia_retirada", "detalle": [], "captura_studio": None}
        )
        self.assertEqual(resultado["motivo"], "copia_retirada")
        self.assertEqual(len(pa.registro_publicados.leer(registro).filas), 1)

    def test_un_video_que_no_existe_y_no_es_una_copia_sigue_como_antes(self):
        # Comportamiento previo: `no_existe`, el video sigue en pendientes y no se anota nada.
        resultado, registro, _ = self.programar_un_video_que_la_api_no_ve({"estado": "otro"})
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "no_existe"))
        self.assertEqual([p["video_id"] for p in pa.leer_pendientes()], ["V1"])
        self.assertFalse(registro.exists())

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

    def test_un_video_que_ya_no_esta_en_pendientes_toma_el_titulo_de_su_ultima_programacion(self):
        # Defecto: un video que se vuelve a programar al liberarse su reclamo (ya no está en
        # pendientes) se anuncia con su id en lugar del título.
        pa.registrar_programado("V1", "Banda - Album", momento("2028-11-04 18:00"), True)
        resultado, _ = self.programar_sin_reclamo(YouTubeFalso(["succeeded"]))
        self.assertEqual(resultado["titulo"], "Banda - Album")
        self.assertTrue(resultado["resumen"].startswith("🎉✅ Banda - Album\n"), resultado["resumen"])

    def test_el_resumen_dice_publicado_ahora_o_programado_para(self):
        cuando = momento("2026-11-02 18:00")
        ahora = pa.armar_resumen_programado("T", cuando, False, [], publicado_ahora=True)
        self.assertEqual(ahora, "🎉✅ T\nSin reclamos: publicado ahora.")
        luego = pa.armar_resumen_programado("T", cuando, False, [], publicado_ahora=False)
        self.assertEqual(luego, "✓ T\nSin reclamos: programado para 2026-11-02 18:00.")

    def test_el_resumen_de_un_video_con_reclamo_no_promete_apelar(self):
        # Nada apela un reclamo: apelar una impugnación rechazada arriesga un strike.
        cuando = momento("2028-11-11 07:00")
        texto = pa.armar_resumen_programado("T", cuando, True, ["The Orchard Music"])
        self.assertEqual(
            texto,
            "⚠️ T\nTiene reclamo (The Orchard Music): programado para 2028-11-11 07:00, "
            "hasta que se resuelva el reclamo.",
        )


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

    def playlistItems(self):
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

    def test_red_inalcanzable_se_reintenta_como_un_error_de_conexion(self):
        # Defecto: «Network is unreachable» llega como un OSError simple, no como ConnectionError:
        # ejecutar_api lo lanzaba al primer intento y el paso moría en un corte de red.
        for codigo, texto in (
            (errno.ENETUNREACH, "Network is unreachable"),
            (errno.EHOSTUNREACH, "No route to host"),
            (errno.ENETDOWN, "Network is down"),
        ):
            with self.subTest(errno=errno.errorcode[codigo]):
                self.esperas.clear()
                cliente = ClienteQueFalla([OSError(codigo, texto)], respuesta={"items": ["ok"]})
                respuesta = pa.ejecutar_api(
                    self.uploader(), {"cliente": cliente}, lambda c: c.videos().list().execute()
                )
                self.assertEqual(respuesta, {"items": ["ok"]})
                self.assertEqual((cliente.llamadas, cliente.cierres), (2, 1))
                self.assertEqual(self.esperas, [pa.ESPERA_CONEXION_S])

    def test_los_oserror_que_no_son_de_red_se_lanzan_sin_reintentar(self):
        # Defecto: tratar todo OSError como fallo de conexión reintentaría (y luego daría
        # sin_conexion a) un archivo ausente, un permiso denegado o un error de disco.
        errores = [
            FileNotFoundError(errno.ENOENT, "No such file or directory"),
            PermissionError(errno.EACCES, "Permission denied"),
            OSError(errno.EIO, "Input/output error"),
        ]
        for error in errores:
            with self.subTest(error=repr(error)):
                cliente = ClienteQueFalla([error], respuesta={"items": ["ok"]})
                with self.assertRaises(OSError) as contexto:
                    pa.ejecutar_api(self.uploader(), {"cliente": cliente}, lambda c: c.videos().list().execute())
                self.assertIs(contexto.exception, error)
                self.assertEqual((cliente.llamadas, cliente.cierres, self.esperas), (1, 0, []))

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

    def test_un_video_retirado_se_borra_aunque_la_api_no_lo_liste(self):
        # Defecto: la API no lista un video retirado por copia ni a su dueño; con la comprobación
        # de siempre, "Eliminar" respondería "ya estaba borrado" sin borrar nada.
        canal = CanalParaBorrar({})
        resultado = pa.eliminar_video("V1", retirado=True, uploader=uploader_para_programar(canal))
        self.assertEqual((resultado["ok"], resultado["video_id"]), (True, "V1"), resultado)
        self.assertEqual(canal.videos_falsos.borrados, ["V1"])

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

    def test_un_404_al_borrar_se_distingue_de_otro_error_de_la_api(self):
        # Defecto: todo HttpError salía como `error_api` y el servicio decía «Sigue en YouTube»
        # aunque un 404 puede significar que el video ya no existía (sobre todo con --retirado,
        # que no lo consulta antes).
        no_encontrado = HttpError(httplib2.Response({"status": 404}), b"{}")
        canal = CanalParaBorrar({}, errores_al_borrar=[no_encontrado])
        resultado = pa.eliminar_video("V1", retirado=True, uploader=uploader_para_programar(canal))
        self.assertEqual((resultado["ok"], resultado["motivo"], resultado["video_id"]), (False, "no_encontrado", "V1"))
        self.assertTrue(Path(resultado["log"]).is_file())

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
        for extra, retirado in (([], False), (["--retirado"], True)):
            with self.subTest(extra=extra):
                salida = io.StringIO()
                with (
                    mock.patch.object(pa, "eliminar_video", return_value={"ok": True, "video_id": "V1"}) as borrar,
                    mock.patch.object(pa, "estado", side_effect=AssertionError("corrió estado")),
                    contextlib.redirect_stdout(salida),
                ):
                    codigo = pa.main(["eliminar_video", "--video-id", "V1", *extra])
                borrar.assert_called_once_with("V1", retirado=retirado)
                self.assertEqual(codigo, 0)
                self.assertEqual(json.loads(salida.getvalue()), {"ok": True, "video_id": "V1"})


class TestImpugnarSubcomando(unittest.TestCase):
    """`impugnar` es una declaración legal a nombre del usuario: el CLI y el candado importan."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.eventos = []
        parches = [
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "PROGRAMADOS_JSONL", self.tmp / "programados.jsonl"),
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def test_el_subcomando_corre_impugnar_con_su_opcion_y_no_otro_paso(self):
        # Defecto: main manda cualquier subcomando sin rama a `estado` (el servicio recibiría
        # los pendientes en vez de impugnar), o pierde --sin-enviar y la prueba en seco envía
        # una impugnación real.
        for argumentos, sin_enviar in ((["--video-id", "V1"], False), (["--video-id", "V1", "--sin-enviar"], True)):
            with self.subTest(argumentos=argumentos):
                salida = io.StringIO()
                with (
                    mock.patch.object(pa, "impugnar", return_value={"ok": True, "video_id": "V1"}) as impugnar,
                    mock.patch.object(pa, "estado", side_effect=AssertionError("corrió estado")),
                    contextlib.redirect_stdout(salida),
                ):
                    codigo = pa.main(["impugnar", *argumentos])
                impugnar.assert_called_once_with("V1", sin_enviar=sin_enviar)
                self.assertEqual(codigo, 0)
                self.assertEqual(json.loads(salida.getvalue()), {"ok": True, "video_id": "V1"})

    def test_trabaja_en_studio_con_el_candado_de_programar_tomado(self):
        # Defecto: abrir Brave sobre la copia del perfil mientras `programar` la usa; dos
        # navegadores a la vez sobre el mismo perfil se pisan.
        @contextlib.contextmanager
        def candado(nombre):
            self.eventos.append(f"candado {nombre}")
            yield
            self.eventos.append(f"suelta {nombre}")

        @contextlib.contextmanager
        def abrir_studio():
            self.eventos.append("abre studio")
            yield SimpleNamespace(screenshot=lambda path: None)
            self.eventos.append("cierra studio")

        def impugnar_reclamos(pagina, video_id, sin_enviar, modulo):
            self.eventos.append("impugna")
            return {"reclamos": 1, "impugnados": 1, "fallos": []}

        with (
            mock.patch.object(pa, "bloqueo", candado),
            mock.patch.object(pa, "impugnar_reclamos", impugnar_reclamos),
        ):
            resultado = pa.impugnar("V1", abrir_studio=abrir_studio, modulo=object())
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(
            self.eventos,
            ["candado programar", "abre studio", "impugna", "cierra studio", "suelta programar"],
        )

    def test_sin_navegador_devuelve_el_motivo_en_vez_de_lanzar(self):
        # Defecto: un traceback sin JSON; el servicio vería «sin_respuesta» y no sabría que
        # faltaba la sesión de Studio. Sin título registrado, el título es el id.
        @contextlib.contextmanager
        def abrir_studio():
            raise pa.NavegadorNoDisponible("sin_perfil_de_navegador")
            yield

        resultado = pa.impugnar("V1", abrir_studio=abrir_studio, modulo=object())
        self.assertEqual(
            (resultado["ok"], resultado["motivo"], resultado["video_id"], resultado["titulo"]),
            (False, "sin_perfil_de_navegador", "V1", "V1"),
        )


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

    def test_red_inalcanzable_agota_los_reintentos_y_programar_responde_sin_conexion(self):
        # Caso real (2026-11-10, 11:44): un corte de red lanzó `OSError: [Errno 101] Network is
        # unreachable` desde consultar_video; el paso salió con un traceback, sin su línea JSON.
        cliente = ClienteQueFalla([OSError(errno.ENETUNREACH, "Network is unreachable")] * 10)
        resultado = pa.programar(
            "V1", reclamo=False, activar=monetizacion_activada, uploader=uploader_para_programar(cliente)
        )
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["motivo"], "sin_conexion")
        self.assertEqual(resultado["video_id"], "V1")
        self.assertIn("Network is unreachable", resultado["detalle"])
        self.assertEqual((cliente.llamadas, cliente.cierres), (1 + pa.REINTENTOS_CONEXION, pa.REINTENTOS_CONEXION))

    def test_los_demas_pasos_con_la_api_tambien_responden_sin_conexion_ante_red_inalcanzable(self):
        # Defecto: un paso que atrapa los errores de conexión por su cuenta y se queda sin la red
        # inalcanzable: esa fuga vuelve a ser un traceback sin JSON en ese paso.
        pasos = {
            "eliminar_video": lambda youtube: pa.eliminar_video("V1", uploader=uploader_para_programar(youtube)),
            "mapear_canal": lambda youtube: pa.mapear_canal(uploader=uploader_para_programar(youtube)),
            "sembrar_publicados": lambda youtube: pa.sembrar_publicados(uploader=uploader_para_programar(youtube)),
        }
        for nombre, paso in pasos.items():
            with self.subTest(paso=nombre):
                cliente = ClienteQueFalla([OSError(errno.EHOSTUNREACH, "No route to host")] * 10)
                resultado = paso(cliente)
                self.assertFalse(resultado["ok"])
                self.assertEqual(resultado["motivo"], "sin_conexion")
                self.assertIn("No route to host", resultado["detalle"])
                self.assertEqual(cliente.llamadas, 1 + pa.REINTENTOS_CONEXION)

    def test_un_oserror_que_no_es_de_red_no_se_convierte_en_sin_conexion(self):
        # Defecto: atrapar OSError entero en el paso esconde un archivo ausente o un permiso
        # denegado detrás de «sin conexión», y el servicio reintentaría por una causa que no es la red.
        cliente = ClienteQueFalla([FileNotFoundError(errno.ENOENT, "No such file or directory")] * 10)
        with self.assertRaises(FileNotFoundError):
            pa.programar(
                "V1", reclamo=False, activar=monetizacion_activada, uploader=uploader_para_programar(cliente)
            )
        self.assertEqual((cliente.llamadas, cliente.cierres), (1, 0))

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
