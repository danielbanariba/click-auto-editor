"""`pipeline_agente.seguir_impugnaciones`, contra una página local que imita /claims de Studio.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. La forma de cada
fila es la vista en vivo el 2026-11-04: un `ytcr-video-content-list-row` con canción, artista,
"Derechos de autor", "Audio" y, en `#impact-container`, la línea de estado más "Ver detalles"
y "Tomar medidas". El seguimiento solo lee: cualquier botón pulsado llega al "servidor".
"""
import contextlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import pipeline_agente as pa

PAGINA = """<!doctype html><html><head><meta charset="utf-8"><title>Reclamaciones</title>
<style>ytcr-video-content-list, ytcr-video-content-list-row, ytcp-button { display: block; }</style>
</head><body>
<h1>Contenido protegido</h1>
<ytcr-video-content-list>__FILAS__</ytcr-video-content-list>
<div>__VACIO__</div>
<script>
document.querySelectorAll("button").forEach(b => b.onclick = () =>
  fetch("/_pulsado", {method: "POST", body: b.innerText}));
</script></body></html>"""

FILA = """<ytcr-video-content-list-row>
<div>{cancion}</div><div>{artista}</div><div>Derechos de autor</div><div>Audio</div>
<div id="impact-container" class="table-cell content-impact"><div>{estado}</div>
<div class="action-buttons"><ytcp-button class="details-button"><button>Ver detalles</button></ytcp-button>
<ytcp-button><button>Tomar medidas</button></ytcp-button></div></div>
</ytcr-video-content-list-row>"""

PAGINA_DE_ERROR = "<!doctype html><html><body><h1>Algo salió mal</h1></body></html>"

EN_REVISION = "Impugnación en proceso de revisión. Caduca dentro de {dias} días."
RECHAZADA = "El reclamante ha rechazado tu impugnación."


class StudioFalso:
    """Los reclamos de cada video, las páginas /claims que se abrieron y los botones pulsados."""

    def __init__(self, reclamos):
        self.reclamos = reclamos  # {video_id: [(canción, artista, estado)]}
        self.con_error = set()  # videos cuya /claims responde una página de error
        self.abiertos = []
        self.pulsados = []

    def atender(self, ruta):
        url = ruta.request.url
        if url.endswith("/_pulsado"):
            self.pulsados.append(ruta.request.post_data)
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        encontrado = re.search(r"/video/([^/]+)/claims", url)
        if not encontrado:
            return ruta.fulfill(status=200, body="<h1>Studio</h1>", content_type="text/html")
        video_id = encontrado.group(1)
        self.abiertos.append(video_id)
        if video_id in self.con_error:
            return ruta.fulfill(status=200, body=PAGINA_DE_ERROR, content_type="text/html; charset=utf-8")
        filas = self.reclamos.get(video_id, [])
        html = PAGINA.replace(
            "__FILAS__", "".join(FILA.format(cancion=c, artista=a, estado=e) for c, a, e in filas)
        ).replace("__VACIO__", "" if filas else "No se han encontrado reclamaciones en este vídeo")
        return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")


class TestSeguirImpugnaciones(unittest.TestCase):
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
        self.logs = self.tmp / "logs"
        self.logs.mkdir()
        self.registro = self.tmp / "estado" / "impugnaciones.json"
        parches = [
            mock.patch.object(pa, "DIR_LOGS", self.logs),
            mock.patch.object(pa, "IMPUGNACIONES_JSON", self.registro),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "estado" / "pendientes.json"),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(pa, "ESPERA_RESULTADO_RECLAMOS_S", 2),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def impugnado(self, video_id, impugnados, titulo):
        """La salida de un `impugnar` lanzado por el servicio, como la deja en data/agente/logs."""
        salida = self.logs / f"servicio_impugnar_20261104-0{len(list(self.logs.iterdir()))}0000_abc123.out"
        resultado = {"ok": bool(impugnados), "video_id": video_id, "titulo": titulo, "reclamos": 2,
                     "impugnados": impugnados, "fallos": []}
        salida.write_text("Lanzo impugnar\n" + json.dumps(resultado, ensure_ascii=False) + "\n", encoding="utf-8")

    def seguir(self, studio):
        contexto = self.navegador.new_context()
        self.addCleanup(contexto.close)
        contexto.route("https://studio.youtube.com/**", studio.atender)
        pagina = contexto.new_page()

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        return pa.seguir_impugnaciones(abrir_studio=abrir_studio)

    def test_avisa_un_cambio_de_estado_una_vez_y_no_la_cuenta_de_los_dias(self):
        # Defecto: un cambio de estado que no se avisa, o un aviso diario porque "Caduca dentro
        # de N días" cambia todos los días.
        self.impugnado("V1", 2, "Banda - Uno")
        self.impugnado("V2", 0, "Banda - Dos")  # la impugnación no salió: no hay nada que seguir
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29)),
                                     ("Canción B", "Artista", EN_REVISION.format(dias=29))]})
        primero = self.seguir(studio)
        studio.reclamos["V1"] = [("Canción A", "Artista", EN_REVISION.format(dias=28)),
                                 ("Canción B", "Artista", RECHAZADA)]
        segundo = self.seguir(studio)
        tercero = self.seguir(studio)
        self.assertEqual(primero["cambios"], [])
        self.assertEqual(segundo["cambios"], [{
            "video_id": "V1", "titulo": "Banda - Uno", "reclamo": "Canción B — Artista",
            "antes": EN_REVISION.format(dias=29), "ahora": RECHAZADA,
        }])
        self.assertEqual(tercero["cambios"], [])
        self.assertTrue(tercero["ok"], tercero)
        self.assertNotIn("V2", studio.abiertos)
        self.assertEqual(studio.pulsados, [])

    def test_un_reclamo_que_desaparece_se_avisa_y_el_video_sin_pendientes_deja_de_seguirse(self):
        # Defecto: un reclamo retirado (ya no está en la lista) no se avisaba, o el video se
        # seguía abriendo todos los días aunque no le quedara nada pendiente.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        studio.reclamos["V1"] = []
        segundo = self.seguir(studio)
        self.seguir(studio)
        self.assertEqual(
            [(c["reclamo"], c["ahora"]) for c in segundo["cambios"]], [("Canción A — Artista", None)]
        )
        self.assertEqual(studio.abiertos, ["V1", "V1"])

    def test_una_pagina_que_no_carga_es_un_fallo_y_no_da_nada_por_retirado(self):
        # Defecto: una /claims que no cargó se leía como "sin reclamos" y daba todo por retirado.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        studio.con_error.add("V1")
        segundo = self.seguir(studio)
        studio.con_error.clear()
        tercero = self.seguir(studio)
        self.assertFalse(segundo["ok"])
        self.assertEqual(segundo["cambios"], [])
        self.assertEqual([(f["video_id"], f["motivo"]) for f in segundo["fallos"]],
                         [("V1", "lista_de_reclamos_no_disponible")])
        self.assertEqual(tercero["cambios"], [])


if __name__ == "__main__":
    unittest.main()
