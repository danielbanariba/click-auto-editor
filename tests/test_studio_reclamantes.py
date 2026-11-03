"""Lectura de los reclamantes en /video/<id>/claims, contra una página local que imita a Studio.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. La
estructura es la que se vio en vivo con un video con 7 reclamos: las filas solo traen
la canción y los botones "Ver detalles" / "Tomar medidas", la lista llega después de
cargar la página, y cada "Ver detalles" abre un diálogo "Detalles de la reclamación"
con un bloque "Reclamantes" (por ejemplo "Believe Music, UMG"). Como en Studio, un diálogo
cerrado sigue en el DOM, oculto, y cada "Ver detalles" agrega un diálogo nuevo al final.
"""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import pipeline_agente as pa

PAGINA = """<!doctype html><html><body>
<h1>Resumen de reclamaciones</h1>
<div id="lista"></div>
<div id="vacio"></div>
<script>
const RECLAMOS = __RECLAMOS__;
const CARGA_MS = __CARGA_MS__;
const DETALLE_MS = __DETALLE_MS__;
const $ = (id) => document.getElementById(id);
const linea = (texto) => { const d = document.createElement("div"); d.textContent = texto; return d; };
function abrir(reclamo) {
  // Studio no reutiliza el diálogo: agrega uno nuevo y deja los anteriores ocultos.
  const dialogo = document.createElement("div");
  dialogo.setAttribute("role", "dialog");
  const titulo = document.createElement("h2");
  titulo.textContent = "Detalles de la reclamación";
  const bloque = document.createElement("div");
  const cerrar = document.createElement("button");
  cerrar.textContent = "Cerrar";
  cerrar.onclick = () => { dialogo.hidden = true; };
  dialogo.append(titulo, bloque, cerrar);
  document.body.append(dialogo);
  setTimeout(() => {
    if (dialogo.hidden) return;
    bloque.replaceChildren(
      linea("Reclamantes"), ...reclamo.reclamantes.map(linea),
      linea("Impacto"), linea("Los anuncios se bloquean"));
  }, DETALLE_MS);
}
setTimeout(() => {
  if (!RECLAMOS.length) { $("vacio").textContent = "No se han encontrado reclamaciones en este vídeo"; return; }
  for (const reclamo of RECLAMOS) {
    const fila = document.createElement("div");
    const ver = document.createElement("button"); ver.textContent = "Ver detalles";
    ver.onclick = () => abrir(reclamo);
    const tomar = document.createElement("button"); tomar.textContent = "Tomar medidas";
    tomar.onclick = () => fetch("/_tomar_medidas", {method: "POST"});
    fila.append(linea(reclamo.cancion), linea("Derechos de autor"), linea("Audio"), ver, tomar);
    $("lista").append(fila);
  }
}, CARGA_MS);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  const abiertos = [...document.querySelectorAll("[role=dialog]")].filter((d) => !d.hidden);
  if (abiertos.length) abiertos[abiertos.length - 1].hidden = true;
});
</script></body></html>"""


class StudioFalso:
    """La página de reclamos de un video y lo que la automatización hizo en ella."""

    def __init__(self, reclamos, carga_ms=0, detalle_ms=0, redirige=False):
        self.reclamos = reclamos  # [{"cancion": ..., "reclamantes": [...]}]
        self.carga_ms = carga_ms  # cuánto tarda en llegar la lista de reclamos
        self.detalle_ms = detalle_ms  # cuánto tarda el bloque "Reclamantes" tras abrir el diálogo
        self.redirige = redirige  # /copyright redirige a /claims, como Studio
        self.tomar_medidas = 0

    def atender(self, ruta):
        if ruta.request.url.endswith("/_tomar_medidas"):
            self.tomar_medidas += 1
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if self.redirige and ruta.request.url.endswith("/copyright"):
            redireccion = '<meta http-equiv="refresh" content="0;url=/video/V1/claims">'
            return ruta.fulfill(status=200, body=redireccion, content_type="text/html; charset=utf-8")
        html = (
            PAGINA.replace("__RECLAMOS__", json.dumps(self.reclamos))
            .replace("__CARGA_MS__", str(self.carga_ms))
            .replace("__DETALLE_MS__", str(self.detalle_ms))
        )
        return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")


def reclamo(cancion, *reclamantes):
    return {"cancion": cancion, "reclamantes": list(reclamantes)}


class PaginaQueNavega:
    """Una página real cuyos primeros `evaluate` caen en plena navegación.

    En Studio, /copyright redirige a /claims; un `evaluate` que justo cruza esa navegación
    falla con «Execution context was destroyed». El resto se delega en la página real."""

    def __init__(self, pagina, fallos):
        self.pagina = pagina
        self.fallos = fallos

    def evaluate(self, *args, **kwargs):
        if self.fallos:
            self.fallos -= 1
            raise PlaywrightError(
                "Page.evaluate: Execution context was destroyed, most likely because of a navigation"
            )
        return self.pagina.evaluate(*args, **kwargs)

    def __getattr__(self, nombre):
        return getattr(self.pagina, nombre)


class TestLeerReclamantes(unittest.TestCase):
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
        self.apelacion = pa.apelacion_agente()

    def leer(self, studio, evaluate_que_falla_por_navegacion=0):
        pagina = self.navegador.new_page()
        pagina.route("https://studio.youtube.com/**", studio.atender)
        if evaluate_que_falla_por_navegacion:
            pagina = PaginaQueNavega(pagina, evaluate_que_falla_por_navegacion)
        return self.apelacion.extract_claimants_from_video(pagina, "V1", delay_s=0.2)

    def test_espera_a_que_llegue_la_lista_de_reclamos_y_no_toca_tomar_medidas(self):
        # Defecto: contar los botones al cargar la página, antes de que lleguen las filas,
        # devolvía [] aunque el video tuviera reclamos.
        studio = StudioFalso(
            [
                reclamo("Canción A", "Believe Music", "en nombre de Church Road Records"),
                reclamo("Canción B", "Believe Music, UMG", "en nombre de Church Road Records"),
            ],
            carga_ms=3500,
        )
        reclamantes = self.leer(studio)
        self.assertEqual(
            reclamantes,
            ["Believe Music", "en nombre de Church Road Records", "Believe Music, UMG"],
        )
        # Leer es solo mirar: "Tomar medidas" abriría la disputa del reclamo.
        self.assertEqual(studio.tomar_medidas, 0)

    def test_lee_el_reclamante_que_solo_aparece_en_un_reclamo_posterior(self):
        # Defecto: esperar el encabezado «Reclamantes» del primer diálogo (que sigue en el DOM,
        # oculto) dejaba en [] a todos los reclamos desde el segundo, y con ellos al reclamante
        # de la lista negra que solo figura en uno de los últimos.
        studio = StudioFalso(
            [
                reclamo("Canción A", "Sello A"),
                reclamo("Canción B", "Sello B"),
                reclamo("Canción C", "Sello C, UMG"),
            ]
        )
        self.assertEqual(self.leer(studio), ["Sello A", "Sello B", "Sello C, UMG"])

    def test_lee_los_siete_reclamos_de_un_video(self):
        # Defecto: el tope de 6 reclamos dejaba fuera al reclamante del séptimo.
        studio = StudioFalso([reclamo(f"Canción {i}", f"Sello {i}") for i in range(1, 8)])
        self.assertEqual(self.leer(studio), [f"Sello {i}" for i in range(1, 8)])

    def test_espera_a_que_el_dialogo_cargue_su_bloque_de_reclamantes(self):
        # Defecto: leer el diálogo tras una pausa fija, antes de que cargue el bloque
        # "Reclamantes", perdía los nombres de ese reclamo.
        studio = StudioFalso([reclamo("Canción A", "UMG | en nombre de Artists & Acts")], detalle_ms=2500)
        self.assertEqual(self.leer(studio), ["UMG | en nombre de Artists & Acts"])

    def test_sigue_esperando_la_lista_si_un_evaluate_cae_en_la_navegacion_a_claims(self):
        # Defecto: un «Execution context was destroyed» al esperar la lista, porque
        # /copyright redirige a /claims, hacía devolver [] para un video con reclamos.
        studio = StudioFalso([reclamo("Canción A", "Sello A, UMG")], carga_ms=2500, redirige=True)
        reclamantes = self.leer(studio, evaluate_que_falla_por_navegacion=2)
        self.assertEqual(reclamantes, ["Sello A, UMG"])

    def test_un_video_sin_reclamos_devuelve_lista_vacia_sin_agotar_la_espera(self):
        # Defecto: esperar los botones los 30 s completos en cada video limpio.
        inicio = time.monotonic()
        with mock.patch.object(self.apelacion, "ESPERA_BOTONES_S", 20):
            self.assertEqual(self.leer(StudioFalso([])), [])
        self.assertLess(time.monotonic() - inicio, 12)


if __name__ == "__main__":
    unittest.main()
