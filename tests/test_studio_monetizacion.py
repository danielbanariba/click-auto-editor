"""La automatización de Studio, contra una página local que imita la de monetización.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. Los
textos y roles (radios Activada/Desactivada, Siguiente/Hecho, Guardar, el diálogo
"Cuéntanos qué aparece en tu vídeo") son los que tiene Studio, vistos en vivo.
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import studio_monetizacion as sm

URL = "https://studio.youtube.com/video/V1/monetization"

PAGINA = """<!doctype html><html><body>
<h1>Formas de obtener ingresos</h1>
<button id="guardar" disabled>Guardar</button>
<div id="estado">__ESTADO__</div>
<button id="editar" aria-label="Editar el estado de monetización del vídeo">v</button>
<div id="popup" hidden>
  <label><input type="radio" name="m" value="on"> Activada</label>
  <label><input type="radio" name="m" value="off" checked> Desactivada</label>
  <button id="continuar">Hecho</button>
</div>
<div role="dialog" id="dialogo" hidden>
  <h2>Cuéntanos qué aparece en tu vídeo</h2>
  <button id="cancelar">Cancelar</button><button disabled>Enviar</button>
</div>
<script>
const pedirCuestionario = __CUESTIONARIO__;
const $ = (id) => document.getElementById(id);
$("editar").onclick = () => { $("popup").hidden = false; };
document.querySelectorAll("input[name=m]").forEach((r) => r.onchange = () => {
  $("continuar").textContent = (r.value === "on" && pedirCuestionario) ? "Siguiente" : "Hecho";
});
$("continuar").onclick = () => {
  const activar = document.querySelector("input[name=m]:checked").value === "on";
  if (activar && pedirCuestionario) { $("dialogo").hidden = false; return; }
  $("popup").hidden = true;
  $("guardar").disabled = !activar;
};
$("cancelar").onclick = () => { $("dialogo").hidden = true; $("popup").hidden = true; };
$("guardar").onclick = async () => {
  await fetch("/_guardar", {method: "POST"});
  $("guardar").disabled = true;
};
</script></body></html>"""


class StudioFalso:
    """Estado del video en el 'servidor' y todo lo que la automatización le mandó."""

    def __init__(self, estado="Desactivada", cuestionario=False, guarda=True, pagina=None):
        self.estado = estado
        self.cuestionario = cuestionario
        self.guarda = guarda
        self.pagina = pagina
        self.guardados = 0

    def atender(self, ruta):
        peticion = ruta.request
        if peticion.url.endswith("/_guardar"):
            self.guardados += 1
            if self.guarda:
                self.estado = "Activada"
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if self.pagina is not None:
            return ruta.fulfill(status=200, body=self.pagina, content_type="text/html; charset=utf-8")
        html = PAGINA.replace("__ESTADO__", self.estado).replace(
            "__CUESTIONARIO__", json.dumps(self.cuestionario)
        )
        return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")


class TestActivarMonetizacion(unittest.TestCase):
    def setUp(self):
        if sync_playwright is None:
            self.skipTest("sin Playwright")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        try:
            self.navegador = self.playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            self.skipTest(f"sin Chromium: {exc}")
        self.addCleanup(self.navegador.close)

    def activar(self, studio, espera_estado_s=5):
        pagina = self.navegador.new_page()
        pagina.route("https://studio.youtube.com/**", studio.atender)
        return sm.activar_monetizacion(
            pagina, "V1", self.tmp / "captura.png", espera_estado_s=espera_estado_s
        )

    def test_activa_y_comprueba_recargando_que_quedo_guardado(self):
        studio = StudioFalso()
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "activada", resultado)
        self.assertEqual((studio.guardados, studio.estado), (1, "Activada"))
        self.assertTrue((self.tmp / "captura.png").exists())

    def test_un_video_ya_activado_no_se_vuelve_a_guardar(self):
        studio = StudioFalso(estado="Activada")
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "ya_activa", resultado)
        self.assertEqual(studio.guardados, 0)

    def test_no_responde_el_cuestionario_de_idoneidad_y_deja_el_video_como_estaba(self):
        studio = StudioFalso(cuestionario=True)
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "fallo", resultado)
        self.assertEqual(resultado["motivo"], "cuestionario_de_idoneidad")
        self.assertEqual((studio.guardados, studio.estado), (0, "Desactivada"))

    def test_guardar_sin_que_studio_lo_guarde_se_reporta_como_fallo(self):
        # El botón se apaga igual: solo recargar muestra si quedó guardado.
        studio = StudioFalso(guarda=False)
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "fallo", resultado)
        self.assertEqual(resultado["motivo"], "no_se_guardo")

    def test_sin_sesion_el_motivo_es_sin_sesion_en_studio(self):
        def a_login(ruta):
            ruta.fulfill(status=302, headers={"location": "https://accounts.google.com/v3/signin"})

        pagina = self.navegador.new_page()
        pagina.route("https://studio.youtube.com/**", a_login)
        pagina.route(
            "https://accounts.google.com/**",
            lambda ruta: ruta.fulfill(status=200, body="<h1>Sign in</h1>", content_type="text/html"),
        )
        resultado = sm.activar_monetizacion(pagina, "V1", self.tmp / "c.png", espera_estado_s=1)
        self.assertEqual(resultado["motivo"], "sin_sesion_en_studio", resultado)

    def test_pagina_sin_el_estado_conocido_no_toca_nada_y_explica(self):
        studio = StudioFalso(pagina="<h1>Formas de obtener ingresos</h1><p>Vídeo no apto</p>")
        resultado = self.activar(studio, espera_estado_s=1)
        self.assertEqual(resultado["motivo"], "estado_desconocido", resultado)
        self.assertIn("Vídeo no apto", resultado["detalle"])
        self.assertEqual(studio.guardados, 0)


if __name__ == "__main__":
    unittest.main()
