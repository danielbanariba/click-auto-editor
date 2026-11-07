"""`pipeline_agente.mapear_canal`: los videos aparcados por un reclamo, mirados otra vez en Studio.

Los videos salen de la lista de subidas del canal (API falsa). /claims se simula con la misma
página local que el seguimiento de impugnaciones, en Chromium real. Mapear solo lee: cualquier
botón pulsado llega al "servidor" y los tests lo vigilan.
"""
import contextlib
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import pipeline_agente as pa
from tests.test_seguir_impugnaciones import EN_REVISION, FILA, StudioFalso

RECLAMO_ACTIVO = "Reclamo de derechos de autor. Se puede impugnar."
# Un reclamo ya impugnado: Studio deshabilita su "Tomar medidas".
FILA_IMPUGNADA = FILA.replace("<button>Tomar medidas</button>", "<button disabled>Tomar medidas</button>")


class StudioDelCanal(StudioFalso):
    """/claims de cada video; una fila (canción, artista, estado, False) ya no se puede impugnar."""

    def __init__(self, reclamos, sin_sesion=False):
        super().__init__(reclamos)
        self.sin_sesion = sin_sesion

    def atender(self, ruta):
        if self.sin_sesion and "/claims" in ruta.request.url:
            return ruta.fulfill(status=302, headers={"location": "https://accounts.google.com/v3/signin"})
        return super().atender(ruta)

    @staticmethod
    def filas(reclamos):
        return "".join(
            (FILA if (fila[3:] or (True,))[0] else FILA_IMPUGNADA).format(
                cancion=fila[0], artista=fila[1], estado=fila[2]
            )
            for fila in reclamos
        )


class Llamada:
    def __init__(self, respuesta):
        self.respuesta = respuesta

    def execute(self):
        return self.respuesta


class CanalFalso:
    """La lista de subidas en páginas de [(id, privacidad, publishAt o None, título)]."""

    def __init__(self, paginas):
        self.paginas = paginas
        self.catalogo = {
            vid: {"id": vid, "snippet": {"title": titulo},
                  "status": {"privacyStatus": privacidad, **({"publishAt": cuando} if cuando else {})}}
            for pagina in paginas for vid, privacidad, cuando, titulo in pagina
        }

    def playlistItems(self):
        return self

    def videos(self):
        return SimpleNamespace(list=self.listar_videos)

    def list(self, part, playlistId, maxResults, pageToken=None):
        numero = int(pageToken or 0)
        respuesta = {"items": [{"snippet": {"resourceId": {"videoId": v[0]}}} for v in self.paginas[numero]]}
        if numero + 1 < len(self.paginas):
            respuesta["nextPageToken"] = str(numero + 1)
        return Llamada(respuesta)

    def listar_videos(self, part, id):
        # Como la API: el orden de la respuesta no tiene por qué ser el de los ids pedidos.
        return Llamada({"items": [self.catalogo[v] for v in reversed(id.split(","))]})


def uploader_del_canal(canal):
    return SimpleNamespace(
        cargar_env=lambda: None,
        probar_credenciales_disponibles=lambda prefix: (canal, [], [], []),
        get_uploads_playlist_id=lambda cliente: "UU1",
        is_quota_error=lambda exc: False,
        is_invalid_grant_error=lambda exc: False,
        authenticate_next=lambda prefix: None,
    )


def en(dias):
    """Un publishAt como lo da la API: en UTC, con «Z»."""
    return (datetime.now(timezone.utc) + timedelta(days=dias)).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestMapearCanal(unittest.TestCase):
    def setUp(self):
        if sync_playwright is None:
            self.skipTest("sin Playwright")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        try:
            self.navegador = self.playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            self.skipTest(f"sin Chromium: {exc}")
        self.addCleanup(self.navegador.close)
        self.lista_negra = self.tmp / "lista_negra.txt"
        self.lista_negra.write_text("# sellos que no se impugnan\numg\n", encoding="utf-8")
        parches = [
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "estado" / "pendientes.json"),
            mock.patch.object(pa, "MAPEO_JSON", self.tmp / "estado" / "mapeo_canal.json"),
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "LISTA_NEGRA_RECLAMANTES", self.lista_negra),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(pa, "ESPERA_RESULTADO_RECLAMOS_S", 2),
            mock.patch.object(pa, "PAUSA_RELECTURA_RECLAMOS_S", 1),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)
        self.reclamantes = {}
        self.leidos = []  # los videos cuyos reclamantes se leyeron

    def leer_reclamantes(self, pagina, video_id):
        self.leidos.append(video_id)
        return self.reclamantes.get(video_id, [])

    def mapear(self, studio, canal):
        contexto = self.navegador.new_context()
        self.addCleanup(contexto.close)
        contexto.route("https://studio.youtube.com/**", studio.atender)
        contexto.route(
            "https://accounts.google.com/**",
            lambda ruta: ruta.fulfill(status=200, body="<h1>Sign in</h1>", content_type="text/html"),
        )
        pagina = contexto.new_page()

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        return pa.mapear_canal(
            uploader=uploader_del_canal(canal), abrir_studio=abrir_studio, leer_reclamantes=self.leer_reclamantes
        )

    def test_solo_mira_los_aparcados_y_libera_el_que_ya_no_tiene_reclamos(self):
        # Defecto: los videos sin reclamos quedaban aparcados dos años (S2); o se abría un video
        # programado normal, o se perdían los aparcados de la segunda página de subidas.
        canal = CanalFalso([
            [("V1", "private", en(730), "Banda - Uno"), ("V3", "private", en(3), "Banda - Tres"),
             ("V4", "public", None, "Banda - Cuatro")],
            [("V2", "private", en(700), "Banda - Dos")],
        ])
        studio = StudioDelCanal({
            "V1": [],
            "V2": [("Canción A", "Artista", RECLAMO_ACTIVO)],
            "V3": [("Canción B", "Artista", RECLAMO_ACTIVO)],
        })
        self.reclamantes = {"V2": ["Sello Independiente"]}
        resultado = self.mapear(studio, canal)
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual([v["video_id"] for v in resultado["liberados"]], ["V1"])
        self.assertEqual(resultado["liberados"][0]["titulo"], "Banda - Uno")
        self.assertEqual(
            [(v["video_id"], v["reclamantes"], v["impugnables"]) for v in resultado["sin_impugnar"]],
            [("V2", ["Sello Independiente"], 1)],
        )
        self.assertEqual(resultado["aparcados"], 2)
        self.assertNotIn("V3", studio.abiertos)
        self.assertNotIn("V4", studio.abiertos)
        self.assertEqual(studio.pulsados, [])

    def test_el_de_la_lista_negra_va_aparte_y_el_ya_impugnado_queda_en_proceso(self):
        # Defecto: ofrecer impugnar un video de la lista negra (S3: va con su "Eliminar video"),
        # o volver a ofrecer uno cuya impugnación ya está en revisión.
        canal = CanalFalso([[
            ("V1", "private", en(730), "Banda - Uno"),
            ("V2", "private", en(730), "Banda - Dos"),
            ("V3", "private", en(730), "Banda - Tres"),
        ]])
        studio = StudioDelCanal({
            "V1": [("Canción A", "Artista", RECLAMO_ACTIVO)],
            "V2": [("Canción B", "Artista", EN_REVISION.format(dias=20), False)],
            "V3": [("Canción C", "Artista", RECLAMO_ACTIVO), ("Canción D", "Artista", RECLAMO_ACTIVO)],
        })
        self.reclamantes = {"V1": ["Believe Music, UMG"], "V3": ["Sello Independiente"]}
        resultado = self.mapear(studio, canal)
        (negro,) = resultado["lista_negra"]
        self.assertEqual(negro["video_id"], "V1")
        self.assertEqual(negro["reclamantes_en_lista_negra"], ["Believe Music, UMG"])
        self.assertTrue(Path(negro["captura"]).is_file())
        self.assertEqual([(v["video_id"], v["impugnables"]) for v in resultado["sin_impugnar"]], [("V3", 2)])
        self.assertEqual([v["video_id"] for v in resultado["en_proceso"]], ["V2"])
        self.assertEqual(resultado["liberados"], [])
        self.assertNotIn("V2", self.leidos)
        self.assertEqual(studio.pulsados, [])

    def test_un_aviso_de_sin_reclamos_que_despues_dibuja_la_lista_no_libera_el_video(self):
        # Defecto: Studio muestra un momento "No se han encontrado reclamaciones" antes de dibujar
        # la lista; tomar esa primera lectura publicaría un video con reclamo.
        canal = CanalFalso([[("V1", "private", en(730), "Banda - Uno")]])
        studio = StudioDelCanal({"V1": []})
        studio.despues["V1"] = (1200, [("Canción A", "Artista", RECLAMO_ACTIVO)])
        self.reclamantes = {"V1": ["Sello Independiente"]}
        resultado = self.mapear(studio, canal)
        self.assertEqual(resultado["liberados"], [])
        self.assertEqual([v["video_id"] for v in resultado["sin_impugnar"]], ["V1"])

    def test_sin_saber_quien_reclama_no_va_a_impugnar_todos(self):
        # Defecto: sin los reclamantes no se puede mirar la lista negra, y el video entraba al
        # lote: un toque impugnaría a un reclamante de la lista negra.
        canal = CanalFalso([[("V1", "private", en(730), "Banda - Uno")]])
        studio = StudioDelCanal({"V1": [("Canción A", "Artista", RECLAMO_ACTIVO)]})
        resultado = self.mapear(studio, canal)
        self.assertEqual(resultado["sin_impugnar"], [])
        self.assertEqual(
            [(f["video_id"], f["motivo"]) for f in resultado["fallos"]], [("V1", "reclamantes_no_disponibles")]
        )

    def test_un_video_que_rompe_la_corrida_va_al_final_de_la_proxima(self):
        # Defecto: un video que siempre hace fallar la lectura quedaba primero en cada corrida
        # y ninguna llegaba a los demás.
        canal = CanalFalso([[("V1", "private", en(730), "Banda - Uno"), ("V2", "private", en(730), "Banda - Dos")]])
        studio = StudioDelCanal({v: [("Canción", "Artista", RECLAMO_ACTIVO)] for v in ("V1", "V2")})
        self.reclamantes = {"V2": ["Sello Independiente"]}

        def leer_reclamantes(pagina, video_id):
            if video_id == "V1":
                raise PlaywrightError("Timeout 30000ms exceeded")
            return self.reclamantes[video_id]

        self.leer_reclamantes = leer_reclamantes
        primera = self.mapear(studio, canal)
        segunda = self.mapear(studio, canal)
        self.assertEqual(primera["motivo"], "error_navegador")
        self.assertEqual([v["video_id"] for v in segunda["sin_impugnar"]], ["V2"])

    def test_una_pagina_que_no_se_lee_no_libera_el_video(self):
        # Defecto: una página de error tomada por "sin reclamos" publicaría un video con reclamo.
        canal = CanalFalso([[("V1", "private", en(730), "Banda - Uno")]])
        studio = StudioDelCanal({"V1": [("Canción A", "Artista", RECLAMO_ACTIVO)]})
        studio.con_error.add("V1")
        resultado = self.mapear(studio, canal)
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["liberados"], [])
        self.assertEqual(
            [(f["video_id"], f["motivo"]) for f in resultado["fallos"]], [("V1", "lista_de_reclamos_no_disponible")]
        )

    def test_lo_que_no_alcanza_a_mirarse_sale_sin_revisar_y_va_primero_la_proxima_vez(self):
        # Defecto: con el tope de tiempo, el resto se perdía sin aviso, o la corrida siguiente
        # volvía a empezar por los mismos y nunca llegaba a los últimos.
        canal = CanalFalso([[
            ("V1", "private", en(730), "Banda - Uno"),
            ("V2", "private", en(730), "Banda - Dos"),
            ("V3", "private", en(730), "Banda - Tres"),
        ]])
        studio = StudioDelCanal({v: [("Canción", "Artista", EN_REVISION.format(dias=9), False)] for v in ("V1", "V2", "V3")})
        with mock.patch.object(pa, "TIEMPO_MAXIMO_MAPEO_S", 0):
            primera = self.mapear(studio, canal)
            abiertos_antes = len(studio.abiertos)
            segunda = self.mapear(studio, canal)
        self.assertEqual([v["video_id"] for v in primera["en_proceso"]], ["V1"])
        self.assertEqual([v["video_id"] for v in primera["sin_revisar"]], ["V2", "V3"])
        self.assertEqual(studio.abiertos[abiertos_antes], "V2")
        self.assertEqual([v["video_id"] for v in segunda["sin_revisar"]], ["V3", "V1"])

    def test_sin_sesion_en_studio_no_libera_nada_y_lo_dice(self):
        # Defecto: tomar la página de inicio de sesión por "sin reclamos" y publicar el video.
        canal = CanalFalso([[("V1", "private", en(730), "Banda - Uno")]])
        resultado = self.mapear(StudioDelCanal({"V1": []}, sin_sesion=True), canal)
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["motivo"], "sin_sesion_en_studio")
        self.assertEqual(resultado["liberados"], [])
        self.assertEqual([v["video_id"] for v in resultado["sin_revisar"]], ["V1"])


class TestSubcomandoMapearCanal(unittest.TestCase):
    def test_el_subcomando_corre_el_mapeo(self):
        # Defecto: el servicio lanza `mapear_canal` y el CLI no lo conoce.
        with mock.patch.object(pa, "mapear_canal", return_value={"ok": True}) as mapear, \
                mock.patch.object(pa.os, "chdir"), contextlib.redirect_stdout(None):
            codigo = pa.main(["mapear_canal"])
        self.assertEqual(codigo, 0)
        mapear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
