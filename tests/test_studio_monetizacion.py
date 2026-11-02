"""La automatización de Studio, contra una página local que imita la de monetización.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. Los
textos y roles (radios Activada/Desactivada, Siguiente/Hecho, Guardar, el diálogo
"Cuéntanos qué aparece en tu vídeo") son los que tiene Studio, vistos en vivo.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import studio_monetizacion as sm

URL = "https://studio.youtube.com/video/V1/monetization"
DEMORA_GUARDADO_MS = 1500  # lo que tarda el "servidor" en guardar tras pulsar Guardar

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
  <label><input type="radio" name="cat" value="violencia-leve"> Se incluye violencia leve</label>
  __CHECKBOX__
  <div id="panel"><div>Cómo has calificado tu vídeo en general:</div><div id="calificacion"></div></div>
  <button id="cancelar">Cancelar</button><button id="enviar" disabled>Enviar</button>
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
const nada = $("nada");
if (nada) nada.onclick = () => {
  const marcado = nada.getAttribute("aria-checked") !== "true";
  nada.setAttribute("aria-checked", String(marcado));
  $("calificacion").textContent = marcado ? "__CALIFICACION__" : "";
  $("enviar").disabled = !marcado;
};
$("enviar").onclick = async () => {
  const radios = [...document.querySelectorAll("input[name=cat]:checked")].map((r) => r.value);
  await fetch("/_enviar", {method: "POST", body: JSON.stringify({nada: nada.getAttribute("aria-checked") === "true", radios})});
  $("dialogo").hidden = true; $("popup").hidden = true; $("guardar").disabled = false;
};
$("cancelar").onclick = async () => {
  await fetch("/_cancelar", {method: "POST"});
  $("dialogo").hidden = true; $("popup").hidden = true;
};
// Como en Studio: Guardar se apaga al instante, el estado pasa por "Comprobando" y el guardado
// real llega después. Recargar la página antes de que llegue cancela el guardado.
$("guardar").onclick = () => {
  $("guardar").disabled = true;
  $("estado").textContent = "Comprobando";
  setTimeout(async () => {
    const respuesta = await fetch("/_guardar", {method: "POST"});
    $("estado").textContent = (await respuesta.json()).estado;
  }, __DEMORA_GUARDADO_MS__);
};
</script></body></html>"""


class StudioFalso:
    """Estado del video en el 'servidor' y todo lo que la automatización le mandó."""

    def __init__(
        self,
        estado="Desactivada",
        cuestionario=False,
        guarda=True,
        pagina=None,
        calificacion="Adecuado para anuncios",
        con_checkbox=True,
    ):
        self.estado = estado
        self.cuestionario = cuestionario
        self.guarda = guarda
        self.pagina = pagina
        self.calificacion = calificacion
        self.con_checkbox = con_checkbox
        self.guardados = 0
        self.enviados = []  # lo que se mandó con "Enviar" (irreversible en Studio)
        self.cancelados = 0

    def atender(self, ruta):
        peticion = ruta.request
        if peticion.url.endswith("/_enviar"):
            self.enviados.append(json.loads(peticion.post_data))
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if peticion.url.endswith("/_cancelar"):
            self.cancelados += 1
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if peticion.url.endswith("/_guardar"):
            self.guardados += 1
            if self.guarda:
                self.estado = "Activada"
            return ruta.fulfill(
                status=200, body=json.dumps({"estado": self.estado}), content_type="application/json"
            )
        if self.pagina is not None:
            return ruta.fulfill(status=200, body=self.pagina, content_type="text/html; charset=utf-8")
        checkbox = (
            '<div role="checkbox" id="nada" tabindex="0" aria-checked="false" '
            'aria-label="Nada de lo anterior">Nada de lo anterior</div>'
            if self.con_checkbox
            else ""
        )
        html = (
            PAGINA.replace("__ESTADO__", self.estado)
            .replace("__CUESTIONARIO__", json.dumps(self.cuestionario))
            .replace("__CHECKBOX__", checkbox)
            .replace("__CALIFICACION__", self.calificacion)
            .replace("__DEMORA_GUARDADO_MS__", str(DEMORA_GUARDADO_MS))
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

    def test_cuestionario_se_responde_nada_de_lo_anterior_sin_elegir_radios_y_queda_activada(self):
        studio = StudioFalso(cuestionario=True)
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "activada", resultado)
        self.assertEqual(studio.enviados, [{"nada": True, "radios": []}])
        self.assertEqual((studio.guardados, studio.estado), (1, "Activada"))

    def test_si_el_panel_no_dice_adecuado_para_anuncios_cancela_y_no_envia(self):
        studio = StudioFalso(cuestionario=True, calificacion="Anuncios limitados")
        with mock.patch.object(sm, "ESPERA_PASO_S", 1):
            resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "fallo", resultado)
        self.assertEqual(resultado["motivo"], "calificacion_no_adecuada")
        self.assertEqual((studio.enviados, studio.cancelados, studio.guardados), ([], 1, 0))
        self.assertEqual(studio.estado, "Desactivada")

    def test_sin_el_checkbox_nada_de_lo_anterior_cancela_y_no_envia(self):
        studio = StudioFalso(cuestionario=True, con_checkbox=False)
        with mock.patch.object(sm, "ESPERA_PASO_S", 1):
            resultado = self.activar(studio)
        self.assertEqual(resultado["motivo"], "no_se_encontro_nada_de_lo_anterior", resultado)
        self.assertEqual((studio.enviados, studio.cancelados, studio.guardados), ([], 1, 0))

    def test_si_falla_despues_de_enviar_avisa_que_el_cuestionario_ya_se_envio(self):
        # Las respuestas son irreversibles: el motivo tiene que decir que ya salieron.
        studio = StudioFalso(cuestionario=True, guarda=False)
        resultado = self.activar(studio)
        self.assertEqual(resultado["estado"], "fallo", resultado)
        self.assertEqual(resultado["motivo"], "cuestionario_ya_enviado")
        self.assertEqual(len(studio.enviados), 1)

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
