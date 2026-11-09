"""`pipeline_agente.apelar`, contra una página local que imita la lista de contenido de Studio.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. Los textos y
la forma de la página son los vistos en vivo el 2026-11-04 con la prueba en seco: el globo
de "Avisos" con sus botones, el diálogo "Revisar un problema" (que no se cierra con Escape y
pregunta "¿Seguro que quieres salir?"), la tarjeta "Apelar la decisión", y el panel
"Enviar comentarios a Google", que es un iframe con la caja "Explica tus comentarios
(obligatorio)" y dos "Enviar" (un `div` envoltorio y el botón). Apelar es una declaración a
nombre del usuario: lo que el "servidor" recibe es lo que estos tests vigilan.
"""
import contextlib
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

import pipeline_agente as pa
import studio_apelaciones as sa

PANEL = """<!doctype html><meta charset="utf-8"><body>
<button aria-label="Cerrar" style="width:24px;height:24px" onclick="parent.cerrarPanel()"></button>
<p>Explica tus comentarios (obligatorio)</p>
<textarea aria-label="Explica tus comentarios (obligatorio)"
  oninput="document.getElementById('e').disabled = !this.value;
           document.getElementById('envoltorio').setAttribute('aria-disabled', String(!this.value))"></textarea>
<div role="button" id="envoltorio" aria-disabled="true"><button id="e" disabled
  onclick="parent.enviarComentarios(document.querySelector('textarea').value)">Enviar</button></div>
</body>"""

PAGINA = """<!doctype html><html><head><meta charset="utf-8"><title>Contenido del canal</title>
<style>
  ytcp-video-row, ytcp-video-notices-tooltip-body, yttou-dialog, tp-yt-paper-dialog { display: block; }
  ytcp-video-row { padding: 6px; border-bottom: 1px solid #ccc; }
  .tablecell-restrictions { display: inline-block; padding: 4px; width: 300px; }
  ytcp-video-notices-tooltip-body { position: fixed; right: 20px; top: 20px; width: 320px; background: #eee; }
  yttou-dialog, tp-yt-paper-dialog { position: fixed; left: 20px; top: 20px; width: 600px; min-height: 300px;
                                     background: #fff; border: 1px solid #000; padding: 12px; }
  [hidden] { display: none !important; }
  button { margin: 4px; min-width: 24px; min-height: 24px; }
  [role=checkbox] { display: inline-block; width: 20px; height: 20px; border: 1px solid #000; }
  [aria-checked=true] { background: #000; }
  #panel { position: fixed; right: 0; top: 0; width: 400px; height: 100%; background: #fff; }
</style></head><body>
<div id="filas"></div>
<button id="sig" aria-label="Página siguiente">›</button>
<ytcp-video-notices-tooltip-body id="globo" hidden></ytcp-video-notices-tooltip-body>
<yttou-dialog id="dlg" hidden></yttou-dialog>
<tp-yt-paper-dialog id="fb" hidden></tp-yt-paper-dialog>
<script>
const VIDEOS = __VIDEOS__;
const POR_PAGINA = __POR_PAGINA__;
const NO_CONFIRMA = __NO_CONFIRMA__;
const PANEL = __PANEL__;
const CAUSA_EDAD = "YouTube ha limitado la audiencia de este vídeo a usuarios de al menos 18 años";
const $ = (id) => document.getElementById(id);
const post = (ruta, datos) => fetch(ruta, {method: "POST", body: JSON.stringify(datos)});
let pagina = 0;

function pintarFilas() {
  const desde = pagina * POR_PAGINA;
  $("filas").innerHTML = VIDEOS.slice(desde, desde + POR_PAGINA).map(v =>
    `<ytcp-video-row><a href="/video/${v.id}/edit"><span id="video-title">${v.titulo}</span></a>`
    + `<div class="tablecell-restrictions" data-id="${v.id}">${v.avisos}</div></ytcp-video-row>`).join("");
  $("sig").setAttribute("aria-disabled", String(desde + POR_PAGINA >= VIDEOS.length));
  document.querySelectorAll(".tablecell-restrictions").forEach(c => c.onmouseenter = () => globo(c.dataset.id));
}
$("sig").onclick = () => {
  if ($("sig").getAttribute("aria-disabled") === "true") return;
  pagina++;
  setTimeout(pintarFilas, 300);  // Studio muestra un rato la página anterior
};

function globo(id) {
  const v = VIDEOS.find(x => x.id === id);
  const g = $("globo");
  g.innerHTML = v.botones.map(b => `<button>${b}</button>`).join("");
  g.hidden = false;
  g.querySelectorAll("button").forEach(b => b.onclick = () => {
    g.hidden = true;
    if (b.innerText === CAUSA_EDAD || b.innerText === "Revisar y tomar medidas") abrirEdad(v);
    if (b.innerText === "Ver comentarios") abrirComentarios(v);
  });
}

function engancharX() {
  $("x").onclick = () => {
    if ($("texto")) { $("confirma").hidden = false; return; }
    $("dlg").hidden = true;
  };
  if ($("salir")) $("salir").onclick = () => { $("dlg").hidden = true; };
  if ($("entendido")) $("entendido").onclick = () => post("/_entendido", {id: $("dlg").dataset.id});
}

function abrirEdad(v) {
  const d = $("dlg");
  d.hidden = false;
  d.dataset.id = v.id;
  if (v.edad !== "apelable") {
    const titulo = v.edad === "rechazada" ? "Apelación rechazada" : "Apelación en revisión";
    d.innerHTML = `<h2>${titulo}</h2><p>Tras revisar tu contenido minuciosamente...</p><button id="cerrar">Cerrar</button>`;
    $("cerrar").onclick = () => { d.hidden = true; };
    return;
  }
  d.innerHTML = `<h2>Revisar un problema: ${v.titulo}</h2><button id="x" aria-label="Cerrar"></button>`
    + `<p>Repasa tu contenido y nuestras políticas</p>`
    + `<button id="ver">Ver todas las opciones</button><button id="entendido">Entendido</button>`;
  engancharX();
  $("ver").onclick = () => formulario(v);
}

function formulario(v) {
  const d = $("dlg");
  d.innerHTML = `<h2>Revisar un problema: ${v.titulo}</h2><button id="x" aria-label="Cerrar"></button>`
    + `<h3>Confirmar problema</h3><button id="entendido">Entendido</button>`
    + `<h3>Apelar la decisión</h3><div role="checkbox" aria-checked="false" id="casilla"></div>`
    + `<span>Tengo la certeza de que mi contenido no infringe las políticas de Normas de la comunidad</span>`
    + `<p>Tu explicación</p><textarea id="texto" maxlength="300"></textarea>`
    + `<button id="enviar" disabled>Enviar apelación</button>`
    + `<div id="confirma" hidden><p>¿Seguro que quieres salir?</p><button>Seguir aquí</button><button id="salir">Salir</button></div>`;
  const listo = () => {
    $("enviar").disabled = !($("casilla").getAttribute("aria-checked") === "true" && $("texto").value.length >= 50);
  };
  $("casilla").onclick = () => {
    $("casilla").setAttribute("aria-checked", String($("casilla").getAttribute("aria-checked") !== "true"));
    listo();
  };
  $("texto").oninput = listo;
  $("enviar").onclick = async () => {
    await post("/_apelacion", {id: v.id, texto: $("texto").value,
                               casilla: $("casilla").getAttribute("aria-checked") === "true"});
    if (NO_CONFIRMA) return;  // el formulario queda como estaba
    d.innerHTML = `<h2>Apelación enviada</h2><button id="x" aria-label="Cerrar"></button>`;
    engancharX();
  };
  engancharX();
}

function abrirComentarios(v) {
  const d = $("fb");
  d.hidden = false;
  d.innerHTML = `<h2>Comentarios de la revisión manual</h2><p>Una revisión manual ha confirmado...</p>`
    + `<button id="cerrarfb">Cerrar</button><button id="ec">Enviar comentarios</button>`;
  $("cerrarfb").onclick = () => { d.hidden = true; };
  $("ec").onclick = () => {
    const panel = document.createElement("iframe");
    panel.id = "panel";
    panel.srcdoc = PANEL;
    document.body.appendChild(panel);
    window.cerrarPanel = () => panel.remove();
    window.enviarComentarios = async (texto) => {
      await post("/_comentarios", {id: v.id, texto});
      if (!NO_CONFIRMA) panel.remove();
    };
  };
}
pintarFilas();
</script></body></html>"""

CAUSA_ANUNCIOS = "Los anunciantes no muestran anuncios en este tipo de contenido"


def video(video_id, edad=None, anuncios=None, revisar=False):
    """Un video de la lista: `edad` apelable/rechazada, `anuncios` comentarios/otro.

    `revisar`: el globo trae solo "Revisar y tomar medidas" en lugar de las causas (visto en
    vivo en videos con la apelación ya rechazada).
    """
    if edad:
        avisos = "Visible para mayores de 18 años Sin ingresos publicitarios +1"
        botones = ["Revisar y tomar medidas"] if revisar else [sa.CAUSA_EDAD, CAUSA_ANUNCIOS]
    elif anuncios == "comentarios":
        avisos, botones = "Ingresos publicitarios limitados", ["Ver comentarios"]
    else:
        avisos, botones = "Ingresos publicitarios limitados", ["Solicitar revisión"]
    return {"id": video_id, "titulo": f"Banda {video_id} - Álbum (2026)", "avisos": avisos,
            "botones": botones, "edad": edad, "anuncios": anuncios}


class StudioFalso:
    """La lista de contenido del canal y lo que la automatización mandó."""

    def __init__(self, videos, por_pagina=30, no_confirma=False, sin_sesion=False):
        self.videos = videos
        self.por_pagina = por_pagina
        self.no_confirma = no_confirma  # Studio deja el formulario abierto tras enviar
        self.sin_sesion = sin_sesion
        self.apelaciones = []  # lo que llegó con "Enviar apelación"
        self.comentarios = []  # lo que llegó con "Enviar" del panel de Google
        self.entendidos = []  # "Entendido" confirma el problema: nunca debe llegar

    def atender(self, ruta):
        url = unquote(ruta.request.url)
        for camino, destino in (
            ("/_apelacion", self.apelaciones),
            ("/_comentarios", self.comentarios),
            ("/_entendido", self.entendidos),
        ):
            if url.endswith(camino):
                destino.append(json.loads(ruta.request.post_data))
                if camino == "/_apelacion" and not self.no_confirma:
                    next(v for v in self.videos if v["id"] == destino[-1]["id"])["edad"] = "enviada"
                return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if self.sin_sesion:
            return ruta.fulfill(status=302, headers={"location": "https://accounts.google.com/v3/signin"})
        if "/videos/upload" in url:
            lista = "edad" if "AGE_RESTRICTION" in url else "anuncios"
            videos = [v for v in self.videos if v[lista]]
            html = (
                PAGINA.replace("__VIDEOS__", json.dumps(videos, ensure_ascii=False))
                .replace("__POR_PAGINA__", str(self.por_pagina))
                .replace("__NO_CONFIRMA__", json.dumps(self.no_confirma))
                .replace("__PANEL__", json.dumps(PANEL))
            )
            return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")
        if "/channel/" in url:
            return ruta.fulfill(status=200, body="<h1>Panel</h1>", content_type="text/html")
        # Al entrar, Studio lleva a la página del canal.
        redireccion = '<script>location.replace("/channel/UCcanalFalso/videos")</script>'
        return ruta.fulfill(status=200, body=redireccion, content_type="text/html")


class RelojFalso:
    """El `time` de studio_apelaciones con un desfase que el test sube a mano.

    Simula los minutos que Studio tarda sin esperarlos, y el resto (esperas, pausas) sigue siendo real.
    """

    def __init__(self):
        self.desfase = 0.0

    def monotonic(self):
        return time.monotonic() + self.desfase

    def __getattr__(self, nombre):
        return getattr(time, nombre)


class TestApelar(unittest.TestCase):
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
        self.registro = self.tmp / "estado" / "apelaciones.json"
        parches = [
            mock.patch.object(pa, "APELACIONES_JSON", self.registro),
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "estado" / "pendientes.json"),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(sa, "ESPERA_LISTA_S", 5),
            mock.patch.object(sa, "ESPERA_AVISO_S", 3),
            mock.patch.object(sa, "ESPERA_DIALOGO_S", 5),
            mock.patch.object(sa, "ESPERA_FORMULARIO_S", 3),
            mock.patch.object(sa, "ESPERA_CONFIRMACION_S", 2),
            mock.patch.object(sa, "PAUSA_S", 0.1),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def apelar(self, studio, **opciones):
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

        return pa.apelar(abrir_studio=abrir_studio, **opciones)

    def anotados(self):
        return json.loads(self.registro.read_text(encoding="utf-8"))

    def registrar(self, registro):
        self.registro.parent.mkdir(parents=True, exist_ok=True)
        self.registro.write_text(json.dumps(registro), encoding="utf-8")

    def test_apela_la_restriccion_de_edad_con_el_texto_en_hebreo_y_la_anota(self):
        # Defecto: otro texto, la casilla sin marcar, "Entendido" pulsado o el video sin anotar.
        studio = StudioFalso([video("videoID0001", edad="apelable")])
        resultado = self.apelar(studio)
        self.assertEqual(studio.apelaciones, [{"id": "videoID0001", "texto": sa.TEXTO_APELACION_EDAD, "casilla": True}])
        self.assertEqual(studio.entendidos, [])
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual([v["estado"] for v in resultado["videos"]], ["enviado"])
        self.assertEqual(self.anotados()["edad:videoID0001"]["estado"], "enviado")

    def test_salta_lo_registrado_y_sigue_en_la_pagina_siguiente(self):
        # Defecto: reenviar lo ya apelado, o no pasar de la primera página de la lista.
        self.registro.parent.mkdir(parents=True)
        self.registro.write_text(
            json.dumps({"edad:videoID0001": {"estado": "enviado"}, "edad:videoID0002": {"estado": "ya_apelado"}}),
            encoding="utf-8",
        )
        studio = StudioFalso([video(i, edad="apelable") for i in ("videoID0001", "videoID0002", "videoID0003")], por_pagina=2)
        self.apelar(studio)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0003"])

    def test_un_envio_que_studio_no_confirma_se_anota_y_no_se_repite(self):
        # Defecto: dar por no enviado lo que quizás salió, y mandarlo de nuevo al día siguiente.
        studio = StudioFalso([video("videoID0001", edad="apelable")], no_confirma=True)
        primero = self.apelar(studio)
        self.apelar(studio)
        self.assertEqual(len(studio.apelaciones), 1)
        self.assertEqual([v["estado"] for v in primero["videos"]], ["no_confirmado"])
        self.assertEqual(self.anotados()["edad:videoID0001"]["estado"], "no_confirmado")

    def test_una_apelacion_rechazada_se_anota_sin_mandar_nada(self):
        # Defecto: tomar el diálogo de una apelación ya resuelta como un formulario nuevo.
        studio = StudioFalso([video("videoID0001", edad="rechazada")])
        resultado = self.apelar(studio)
        self.assertEqual(studio.apelaciones, [])
        self.assertEqual(resultado["videos"][0]["estado"], "ya_apelado")
        self.assertEqual(self.anotados()["edad:videoID0001"]["detalle"], "Apelación rechazada")

    def test_un_aviso_con_revisar_y_tomar_medidas_abre_el_problema_y_anota_la_rechazada(self):
        # Defecto: sin la causa en el globo no se abría nada y tres de esos frenaban la corrida.
        studio = StudioFalso([video("videoID0001", edad="rechazada", revisar=True)])
        resultado = self.apelar(studio)
        self.assertEqual(studio.apelaciones, [])
        self.assertEqual(resultado["videos"][0]["estado"], "ya_apelado")
        self.assertEqual(self.anotados()["edad:videoID0001"]["detalle"], "Apelación rechazada")

    def test_por_revisar_y_tomar_medidas_un_formulario_apelable_no_se_manda_y_queda_a_mano(self):
        # Defecto: ese botón es genérico y el formulario no dice de qué problema es; mandar ahí
        # el texto de edad sería una declaración a ciegas.
        studio = StudioFalso([video("videoID0001", edad="apelable", revisar=True)])
        resultado = self.apelar(studio)
        self.assertEqual(studio.apelaciones, [])
        self.assertEqual(resultado["videos"][0]["estado"], "revisar_a_mano")
        self.assertEqual(self.anotados()["edad:videoID0001"]["estado"], "revisar_a_mano")

    def test_anuncios_limitados_manda_el_texto_por_el_panel_de_comentarios(self):
        # Defecto: no llegar a la caja del iframe, otro texto, o mandar comentarios de anuncios a
        # un video con restricción de edad (su apelación es la de edad).
        studio = StudioFalso([
            video("videoID0001", edad="rechazada", anuncios="comentarios"),
            video("videoID0002", anuncios="comentarios"),
        ])
        resultado = self.apelar(studio)
        self.assertEqual(studio.comentarios, [{"id": "videoID0002", "texto": sa.TEXTO_COMENTARIOS_ANUNCIOS}])
        self.assertTrue(resultado["ok"], resultado)
        self.assertEqual(self.anotados()["anuncios:videoID0002"]["estado"], "enviado")
        self.assertNotIn("anuncios:videoID0001", self.anotados())

    def test_sin_revision_manual_no_manda_nada_y_lo_anota(self):
        # Defecto: pulsar un botón desconocido del globo o volver a abrirlo todos los días.
        studio = StudioFalso([video("videoID0001", anuncios="otro")])
        resultado = self.apelar(studio)
        self.assertEqual(studio.comentarios, [])
        self.assertEqual(resultado["videos"][0]["estado"], "sin_opcion")
        self.assertEqual(self.anotados()["anuncios:videoID0001"]["estado"], "sin_opcion")

    def test_respeta_el_limite_de_envios_por_corrida(self):
        # Defecto: mandar todo el atraso de una vez (cientos de apelaciones parecen spam).
        studio = StudioFalso([video(i, edad="apelable") for i in ("videoID0001", "videoID0002", "videoID0003")])
        self.apelar(studio, limite=2)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0001", "videoID0002"])

    def test_sin_enviar_llena_los_formularios_y_no_manda_ni_anota_nada(self):
        # Defecto: la prueba en seco manda la apelación o deja videos anotados como hechos.
        studio = StudioFalso([video("videoID0001", edad="apelable"), video("videoID0002", anuncios="comentarios")])
        resultado = self.apelar(studio, sin_enviar=True)
        self.assertEqual((studio.apelaciones, studio.comentarios, studio.entendidos), ([], [], []))
        self.assertEqual([v["estado"] for v in resultado["videos"]], ["listo", "listo"])
        self.assertFalse(self.registro.exists())

    def test_una_apelacion_enviada_que_youtube_rechazo_se_avisa_una_vez_y_se_anota(self):
        # Defecto: una apelación mandada no se volvía a mirar y su rechazo nunca se avisaba.
        self.registrar({"edad:videoID0001": {"estado": "enviado", "tipo": "edad", "video_id": "videoID0001"}})
        studio = StudioFalso([video("videoID0001", edad="rechazada")])
        primero = self.apelar(studio)
        segundo = self.apelar(studio)
        self.assertEqual((studio.apelaciones, studio.entendidos), ([], []))
        self.assertEqual(
            [(r["video_id"], r["resultado"]) for r in primero["resultados"]], [("videoID0001", "rechazada")]
        )
        self.assertEqual(segundo["resultados"], [])
        anotado = self.anotados()["edad:videoID0001"]
        self.assertEqual((anotado["estado"], anotado["resultado"]), ("enviado", "rechazada"))

    def test_una_apelacion_en_revision_no_es_un_resultado(self):
        # Defecto: tomar cualquier diálogo que nombra la apelación como un rechazo.
        self.registrar({"edad:videoID0001": {"estado": "enviado"}})
        resultado = self.apelar(StudioFalso([video("videoID0001", edad="enviada")]))
        self.assertEqual(resultado["resultados"], [])
        self.assertNotIn("resultado", self.anotados()["edad:videoID0001"])

    def test_al_llegar_al_limite_sigue_mirando_las_apelaciones_ya_enviadas(self):
        # Defecto: al llegar al límite la corrida terminaba y no miraba las que estaban más abajo.
        self.registrar({"edad:videoID0003": {"estado": "enviado"}})
        studio = StudioFalso([
            video("videoID0001", edad="apelable"),
            video("videoID0002", edad="apelable"),
            video("videoID0003", edad="rechazada"),
        ])
        resultado = self.apelar(studio, limite=1)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0001"])
        self.assertEqual([r["video_id"] for r in resultado["resultados"]], ["videoID0003"])

    def test_una_apelacion_cuyo_video_sigue_fuera_de_la_lista_en_la_corrida_siguiente_se_avisa_como_aceptada(self):
        # Defecto: YouTube acepta una apelación sacando el video de la lista, y eso nunca se avisaba;
        # y darla por aceptada a la primera ausencia (video borrado, privado o lista que cambió)
        # dejaba un falso "aceptada" para siempre, porque el resultado se anota una sola vez.
        self.registrar({
            "edad:videoID0001": {"estado": "enviado", "tipo": "edad", "video_id": "videoID0001", "titulo": "Uno"},
            "anuncios:videoID0002": {"estado": "enviado", "tipo": "anuncios", "video_id": "videoID0002", "titulo": "Dos"},
        })
        studio = StudioFalso([video("videoID0003", edad="rechazada"), video("videoID0004", anuncios="otro")])
        primera = self.apelar(studio)
        self.assertEqual(primera["resultados"], [])
        for clave in ("edad:videoID0001", "anuncios:videoID0002"):
            anotado = self.anotados()[clave]
            self.assertIn("ausente", anotado)
            self.assertEqual(anotado["estado"], "enviado")
            self.assertNotIn("resultado", anotado)
        segunda = self.apelar(studio)
        self.assertEqual(
            [(r["tipo"], r["video_id"], r["titulo"], r["resultado"]) for r in segunda["resultados"]],
            [("edad", "videoID0001", "Uno", "aceptada"), ("anuncios", "videoID0002", "Dos", "aceptada")],
        )
        self.assertEqual(self.anotados()["anuncios:videoID0002"]["resultado"], "aceptada")

    def test_un_video_que_vuelve_a_la_lista_pierde_la_marca_de_ausente(self):
        # Defecto: un video que falta una vez y vuelve conservaba la marca, y su próxima ausencia
        # ya lo daba por aceptado sin que fueran dos corridas seguidas.
        self.registrar({
            "edad:videoID0001": {"estado": "enviado", "tipo": "edad", "video_id": "videoID0001", "titulo": "Uno"},
        })
        studio = StudioFalso([video("videoID0003", edad="rechazada")])
        self.apelar(studio)
        self.assertIn("ausente", self.anotados()["edad:videoID0001"])
        studio.videos.append(video("videoID0001", edad="enviada"))
        self.apelar(studio)
        self.assertNotIn("ausente", self.anotados()["edad:videoID0001"])
        studio.videos.pop()
        resultado = self.apelar(studio)
        self.assertEqual(resultado["resultados"], [])
        self.assertNotIn("resultado", self.anotados()["edad:videoID0001"])

    def test_en_la_prueba_en_seco_la_ausencia_de_un_video_no_se_anota(self):
        # Defecto: la prueba en seco escribía en el registro real y contaba como una de las dos
        # corridas seguidas.
        antes = {"edad:videoID0001": {"estado": "enviado", "tipo": "edad", "video_id": "videoID0001"}}
        self.registrar(antes)
        self.apelar(StudioFalso([video("videoID0003", edad="rechazada")]), sin_enviar=True)
        self.assertEqual(self.anotados(), antes)

    def test_una_lista_sin_recorrer_hasta_el_final_no_da_nada_por_aceptado(self):
        # Defecto: dar por aceptada una apelación porque su video no estaba en las páginas vistas.
        self.registrar({"edad:videoID0003": {"estado": "enviado"}})
        studio = StudioFalso([video(i, edad="rechazada") for i in ("videoID0001", "videoID0002")], por_pagina=1)
        with mock.patch.object(sa, "MAX_PAGINAS", 1):
            resultado = self.apelar(studio)
        self.assertEqual(resultado["resultados"], [])
        self.assertNotIn("resultado", self.anotados()["edad:videoID0003"])
        self.assertNotIn("ausente", self.anotados()["edad:videoID0003"])

    def test_una_lista_vacia_no_da_nada_por_aceptado(self):
        # Defecto: una lista que no llegó a cargar (sin filas) daba por aceptado todo lo pendiente.
        self.registrar({"edad:videoID0001": {"estado": "enviado"}})
        resultado = self.apelar(StudioFalso([]))
        self.assertEqual(resultado["resultados"], [])
        self.assertNotIn("resultado", self.anotados()["edad:videoID0001"])
        self.assertNotIn("ausente", self.anotados()["edad:videoID0001"])

    def test_un_resultado_que_no_se_puede_leer_es_un_fallo_de_revision_sin_tocar_el_registro(self):
        # Defecto: un seguimiento que falla se perdía en silencio o se avisaba como "no pude apelar".
        self.registrar({"edad:videoID0001": {"estado": "enviado"}})
        sin_causa = {**video("videoID0001", edad="rechazada"), "botones": ["Otra cosa"]}
        resultado = self.apelar(StudioFalso([sin_causa]))
        self.assertEqual([(f["video_id"], f.get("revision")) for f in resultado["fallos"]], [("videoID0001", True)])
        self.assertEqual(self.anotados(), {"edad:videoID0001": {"estado": "enviado"}})

    def test_seguimientos_que_no_se_pueden_leer_no_frenan_las_apelaciones_nuevas(self):
        # Defecto: tres apelaciones ya mandadas con un aviso ilegible cortaban la corrida
        # (demasiados_fallos) antes de mandar las nuevas, y eso se repetía todos los días.
        pendientes = ("videoID0001", "videoID0002", "videoID0003")
        self.registrar({f"edad:{v}": {"estado": "enviado"} for v in pendientes})
        ilegibles = [{**video(v, edad="rechazada"), "botones": ["Otra cosa"]} for v in pendientes]
        studio = StudioFalso(ilegibles + [video("videoID0004", edad="apelable"),
                                          video("videoID0005", anuncios="comentarios")])
        resultado = self.apelar(studio)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0004"])
        self.assertEqual([c["id"] for c in studio.comentarios], ["videoID0005"])
        self.assertNotIn("motivo", resultado)
        self.assertEqual([f.get("revision") for f in resultado["fallos"]], [True, True, True])

    def test_los_seguimientos_tienen_su_propio_tiempo_y_no_le_quitan_el_suyo_a_los_envios(self):
        # Defecto: seguimientos y envíos gastaban los mismos 45 min; con varias apelaciones
        # esperando resultado se acababa el tiempo antes de llegar a mandar las nuevas. El
        # seguimiento que no se puede leer también gasta tiempo (espera el diálogo).
        pendientes = [f"videoID000{i}" for i in range(1, 6)]
        for legible in (True, False):
            with self.subTest(legible=legible):
                self.registrar({f"edad:{v}": {"estado": "enviado"} for v in pendientes})
                botones = {} if legible else {"botones": ["Otra cosa"]}
                studio = StudioFalso(
                    [{**video(v, edad="enviada"), **botones} for v in pendientes] + [video("videoID0006", edad="apelable")]
                )
                reloj = RelojFalso()
                seguidos = []

                def seguimiento_de_diez_minutos(pagina, fila):
                    seguidos.append(fila["video_id"])
                    reloj.desfase += 10 * 60
                    return sa._revisar_edad(pagina, fila)

                with (
                    mock.patch.object(sa, "time", reloj),
                    mock.patch.object(sa, "ESPERA_DIALOGO_S", 1),
                    mock.patch.dict(sa.REVISAR, {"edad": seguimiento_de_diez_minutos}),
                ):
                    resultado = self.apelar(studio)
                self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0006"])
                self.assertNotIn("motivo", resultado)
                # Los seguimientos que quedaban se saltaron: los 45 min de la corrida siguen siendo el techo.
                self.assertLess(len(seguidos), len(pendientes))

    def test_un_resultado_leido_entre_envios_fallidos_no_reinicia_la_cuenta(self):
        # Defecto: un resultado leído entre dos envíos fallidos ponía la cuenta en cero, y un
        # formulario roto se volvía a probar con cada video del atraso.
        self.registrar({"edad:videoID0002": {"estado": "enviado"}, "edad:videoID0004": {"estado": "enviado"}})

        def roto(video_id):
            return {**video(video_id, edad="apelable"), "botones": ["Otra cosa"]}

        studio = StudioFalso([
            roto("videoID0001"), video("videoID0002", edad="rechazada"), roto("videoID0003"),
            video("videoID0004", edad="rechazada"), roto("videoID0005"), video("videoID0006", edad="apelable"),
        ])
        with mock.patch.object(sa, "MAX_SIN_ABRIR", 3):  # estos fallos son diálogos que no se abren
            resultado = self.apelar(studio)
        self.assertEqual(resultado["motivo"], "demasiados_fallos")
        self.assertEqual(studio.apelaciones, [])

    def test_tres_dialogos_que_no_se_abren_no_frenan_el_resto_de_la_lista(self):
        # Defecto: un fallo pasajero de Studio (el diálogo no se abre) contaba como tres fallos
        # seguidos y cortaba la corrida: los videos de más abajo no se apelaban nunca. Esos
        # videos no se anotan, así que mañana se reintentan solos.
        sin_abrir = [{**video(v, edad="apelable"), "botones": ["Otra cosa"]}
                     for v in ("videoID0001", "videoID0002", "videoID0003")]
        studio = StudioFalso(sin_abrir + [video("videoID0004", edad="apelable")])
        with mock.patch.object(sa, "ESPERA_DIALOGO_S", 1):
            resultado = self.apelar(studio)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0004"])
        self.assertNotIn("motivo", resultado)
        self.assertEqual([f["motivo"] for f in resultado["fallos"]], ["dialogo_desconocido"] * 3)
        self.assertEqual(list(self.anotados()), ["edad:videoID0004"])

    def test_los_dialogos_sin_abrir_se_cuentan_de_nuevo_despues_de_apelar_bien_un_video(self):
        # Defecto: la cuenta de diálogos sin abrir no se reiniciaba al apelar bien un video, y
        # una corrida con fallos pasajeros salteados se cortaba aunque Studio siguiera funcionando.
        def roto(video_id):
            return {**video(video_id, edad="apelable"), "botones": ["Otra cosa"]}

        studio = StudioFalso([
            roto("videoID0001"), roto("videoID0002"), video("videoID0003", edad="apelable"),
            roto("videoID0004"), roto("videoID0005"), video("videoID0006", edad="apelable"),
        ])
        with mock.patch.object(sa, "MAX_SIN_ABRIR", 3), mock.patch.object(sa, "ESPERA_DIALOGO_S", 1):
            resultado = self.apelar(studio)
        self.assertEqual([a["id"] for a in studio.apelaciones], ["videoID0003", "videoID0006"])
        self.assertNotIn("motivo", resultado)

    def test_otros_fallos_siguen_cortando_la_corrida_a_los_tres_seguidos(self):
        # Defecto: aplicar el tope holgado de los diálogos sin abrir a todo fallo, y probar un
        # formulario roto con cada video del atraso.
        rotos = [{**video(v, edad="apelable"), "botones": []} for v in ("videoID0001", "videoID0002", "videoID0003")]
        studio = StudioFalso(rotos + [video("videoID0004", edad="apelable")])
        with mock.patch.object(sa, "INTENTOS_AVISO", 1), mock.patch.object(sa, "ESPERA_AVISO_S", 1):
            resultado = self.apelar(studio)
        self.assertEqual(resultado["motivo"], "demasiados_fallos")
        self.assertEqual(studio.apelaciones, [])
        self.assertEqual([f["motivo"] for f in resultado["fallos"]], ["aviso_no_disponible"] * 3)

    def test_sin_sesion_en_studio_no_manda_nada_y_lo_dice(self):
        studio = StudioFalso([video("videoID0001", edad="apelable")], sin_sesion=True)
        resultado = self.apelar(studio)
        self.assertEqual(studio.apelaciones, [])
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["motivo"], "sin_sesion_en_studio")


if __name__ == "__main__":
    unittest.main()
