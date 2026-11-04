"""`pipeline_agente.leer_retiro_en_studio`, contra una página local que imita /video/<id>/edit.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. La forma es la
vista en vivo el 2026-11-04 con Naked Whipper (lv2-IuhwF4c): en "Avisos", "Retirado" abre un
globo `ytcp-video-notices-tooltip-body`, y su botón "Revisar problemas" abre el diálogo
"Términos y políticas" con quién pidió la retirada. La lectura solo pulsa esas dos cosas: cada
clic llega al "servidor".
"""
import contextlib
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

PAGINA = """<!doctype html><html><head><meta charset="utf-8"><title>Detalles del vídeo</title>
<style>ytcp-video-notices-tooltip-body { display: none; } .visible { display: block !important; }</style>
</head><body>
<h1>Detalles del vídeo</h1>
<div>Avisos</div>__AVISO__
<ytcp-video-notices-tooltip-body id="globo">__GLOBO__
<button id="revisar">Revisar problemas</button></ytcp-video-notices-tooltip-body>
<div id="dialogo" role="dialog" style="display: none">__DIALOGO__ <a href="#">apelarla</a>
<button>Cerrar</button></div>
<script>
document.addEventListener("click", e => {
  fetch("/_pulsado", {method: "POST", body: (e.target.innerText || "").trim()});
}, true);
const aviso = document.getElementById("aviso");
if (aviso) aviso.addEventListener("click", () => document.getElementById("globo").classList.add("visible"));
document.getElementById("revisar").addEventListener("click", () => {
  if (__ABRE_DIALOGO__) document.getElementById("dialogo").style.display = "block";
});
</script></body></html>"""

AVISO = '<span id="aviso">Retirado</span>'
GLOBO_COPIA = (
    "Este vídeo se ha retirado de YouTube.<br><br>El motivo es que el vídeo es un duplicado de otro "
    "que se ha retirado, lo que infringe nuestros términos y políticas. Por lo tanto, solo tú puedes "
    "ver este vídeo.<br><br>"
)
DIALOGO_COPIA = (
    "<h2>Términos y políticas</h2><p>No se puede subir este vídeo. Contiene una copia de otro vídeo "
    "que se retiró anteriormente de la plataforma debido a una solicitud de retirada por infracción "
    "de derechos de autor. Más información</p>"
    "<div><b>Contenido cuyos derechos de autor se infringen:</b> Chapel Defilement</div>"
    "<div><b>Solicitud recibida:</b> 2 feb 2025</div>"
    "<div><b>De:</b> Naked Whipper Official</div>"
    "<p>Si no estás de acuerdo con esta decisión, puedes</p>"
)
GLOBO_NORMAS = (
    "Este vídeo se ha retirado de YouTube.<br><br>El motivo es que infringe las Normas de la "
    "comunidad de YouTube.<br><br>"
)


class StudioFalso:
    def __init__(self, aviso=AVISO, globo=GLOBO_COPIA, dialogo=DIALOGO_COPIA, abre_dialogo=True):
        self.html = (
            PAGINA.replace("__AVISO__", aviso)
            .replace("__GLOBO__", globo)
            .replace("__DIALOGO__", dialogo)
            .replace("__ABRE_DIALOGO__", "true" if abre_dialogo else "false")
        )
        self.abiertas = []
        self.pulsados = []

    def atender(self, ruta):
        url = ruta.request.url
        if url.endswith("/_pulsado"):
            self.pulsados.append(ruta.request.post_data)
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        self.abiertas.append(url)
        return ruta.fulfill(status=200, body=self.html, content_type="text/html; charset=utf-8")


class TestLeerRetiroEnStudio(unittest.TestCase):
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
        parches = [
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(pa, "ESPERA_AVISO_RETIRADO_MS", 2000),
            mock.patch.object(pa, "ESPERA_DIALOGO_RETIRADO_MS", 2000),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def leer(self, studio):
        contexto = self.navegador.new_context()
        self.addCleanup(contexto.close)
        contexto.route("https://studio.youtube.com/**", studio.atender)
        pagina = contexto.new_page()

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        resultado = pa.leer_retiro_en_studio("V1", abrir_studio=abrir_studio)
        pagina.wait_for_timeout(300)  # que lleguen los avisos de clic que quedaron en vuelo
        return resultado

    def test_una_copia_de_un_video_retirado_trae_quien_pidio_la_retirada(self):
        # Defecto: no reconocer el retiro por copia (el servicio diría "rechazó o perdió"), o
        # perder quién pidió la retirada, que es lo que explica el aviso.
        studio = StudioFalso()
        resultado = self.leer(studio)
        self.assertEqual(resultado["estado"], "copia_retirada", resultado)
        self.assertEqual(
            resultado["detalle"],
            [
                "Contenido cuyos derechos de autor se infringen: Chapel Defilement",
                "Solicitud recibida: 2 feb 2025",
                "De: Naked Whipper Official",
            ],
        )
        self.assertTrue(Path(resultado["captura_studio"]).is_file())
        self.assertEqual(studio.abiertas, ["https://studio.youtube.com/video/V1/edit"])
        # Solo abre información: nunca "apelarla" ni nada dentro del diálogo.
        self.assertEqual(studio.pulsados, ["Retirado", "Revisar problemas"])

    def test_un_retiro_por_otro_motivo_no_es_una_copia(self):
        # Defecto: tomar cualquier retiro por una copia y ofrecer solo borrar el video.
        studio = StudioFalso(globo=GLOBO_NORMAS)
        resultado = self.leer(studio)
        self.assertEqual(resultado["estado"], "otro", resultado)
        self.assertEqual(studio.pulsados, ["Retirado"])

    def test_sin_el_aviso_de_retirado_no_es_una_copia(self):
        # Defecto: un video borrado (Studio no muestra "Retirado") dado por copia retirada.
        studio = StudioFalso(aviso="")
        resultado = self.leer(studio)
        self.assertEqual(resultado["estado"], "otro", resultado)
        self.assertEqual(studio.pulsados, [])

    def test_si_el_dialogo_no_abre_el_globo_alcanza_para_saber_que_es_una_copia(self):
        # Defecto: perder el aviso por no poder leer quién pidió la retirada.
        studio = StudioFalso(abre_dialogo=False)
        resultado = self.leer(studio)
        self.assertEqual((resultado["estado"], resultado["detalle"]), ("copia_retirada", []))


if __name__ == "__main__":
    unittest.main()
