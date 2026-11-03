"""`pipeline_agente.impugnar`, contra una página local que imita el formulario de disputa de Studio.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. Los textos
y roles son los que tiene Studio, vistos en vivo con un video con 5 reclamos: la lista
de /claims con "Tomar medidas", el diálogo "Seleccionar acción" con sus tarjetas, y el
formulario "Impugnar reclamación: <canción>" de cuatro pasos (Descripción general, Motivo,
Detalles, Argumentos) que termina en "Enviar". Impugnar es una declaración legal a nombre
del usuario: lo que el "servidor" recibe con "Enviar" es lo que estos tests vigilan.
"""
import contextlib
import json
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
<style>
  .fila { display: flex; gap: 12px; padding: 8px; align-items: center; }
  button[disabled] { opacity: .4; }
  #dlg { position: fixed; left: 20px; top: 20px; right: 20px; bottom: 20px; overflow: auto;
         background: #fff; border: 1px solid #000; padding: 16px; }
  .card { display: block; width: 100%; text-align: left; min-height: 72px; margin: 6px 0; }
  [role=radio], [role=checkbox] { display: block; padding: 6px; border: 1px solid #ccc; margin: 6px 0; }
  [aria-checked=true] { background: #cde; }
</style></head><body>
<h1>Contenido protegido</h1>
<div id="lista"></div>
<div id="vacio"></div>
<div role="dialog" id="dlg" hidden></div>
<script>
const RECLAMOS = __RECLAMOS__;
const TITULO = __TITULO__;
const CARGA_MS = __CARGA_MS__;
const PIDE_VERIFICACION = __VERIFICACION__;
const $ = (id) => document.getElementById(id);
const dlg = $("dlg");
const paso = (nombre) => fetch("/_paso", {method: "POST", body: nombre});
const RAZONES = [
  ["Contenido original", "El contenido reclamado de mi vídeo lo creé yo y tengo todos los derechos sobre él"],
  ["Licencia", "Tengo permiso o dispongo de una licencia para usar el contenido reclamado de mi vídeo"],
  ["Excepción de derechos de autor como el uso legítimo",
   "He modificado considerablemente el contenido reclamado de mi vídeo, el vídeo está amparado por el principio de uso legítimo"],
  ["Dominio público", "El contenido reclamado de mi vídeo es de dominio público y ya no está protegido"],
];
const OTRAS_ACCIONES = [
  ["Borrar canción", "Quita la canción reclamada, pero manteniendo otros sonidos como los diálogos"],
  ["Reemplazar canción", "Elige un audio de la Biblioteca de audio de YouTube para sustituir el audio reclamado"],
  ["Recortar segmento", "Recorta el vídeo y el audio de los segmentos reclamados para resolver la reclamación"],
];
const DECLARACIONES = [
  "Mi vídeo no infringe los derechos de autor que se indican en la reclamación",
  "Entiendo que el reclamante podrá revisar mi vídeo y los argumentos de mi impugnación",
  "Soy consciente de que presentar impugnaciones fraudulentas puede dar lugar a la cancelación de mi cuenta de YouTube",
];
let est = null;  // {indice, paso, motivo}

function alternar(caja, despues) {
  caja.onclick = () => {
    caja.setAttribute("aria-checked", String(caja.getAttribute("aria-checked") !== "true"));
    if (despues) despues();
  };
}

function pintar() {
  const reclamo = RECLAMOS[est.indice];
  const p = est.paso;
  let html = "";
  if (p === 0) {
    html = `<div>Contenido usado: ${reclamo.cancion}</div><h2>Seleccionar acción</h2>`
      + OTRAS_ACCIONES.map(([t, d]) => `<button class="card"><div>${t}</div><div>${d}</div></button>`).join("")
      + `<button class="card" id="c-impugnar"><div>Impugnar</div>`
      + `<div>Impugna una reclamación si tienes los derechos necesarios para usar el contenido.</div></button>`;
  } else if (p === "verif") {
    html = `<h2>Verifica que eres tú</h2><p>Para continuar, necesitamos confirmar que eres tú.</p>`;
  } else {
    html = `<div>Vídeo: ${TITULO}</div><h2>Impugnar reclamación: ${reclamo.cancion}</h2>`
      + `<ol><li>Descripción general</li><li>Motivo</li><li>Detalles</li><li>Argumentos</li></ol>`;
    if (p === 1) {
      html += `<p>Tiempo de procesamiento: hasta 30 días</p><p>¿Cuándo puedo impugnar una reclamación?</p>`;
    } else if (p === 2) {
      html += `<h3>Motivo</h3><p>Selecciona el motivo principal por el que quieres impugnar esta reclamación</p>`
        + RAZONES.map(([t, d]) => `<div role="radio" aria-checked="false" tabindex="0"><div>${t}</div><div>${d}</div></div>`).join("")
        + `<p>Recuerda que estos motivos no son válidos para impugnar una reclamación de derechos de autor.</p>`
        + `<p>Tengo una copia de la canción o del vídeo</p><p>No estoy obteniendo ingresos con el vídeo</p>`;
    } else if (p === 3) {
      html += `<h3>No todas las licencias son iguales</h3><p>Si tienes una licencia, cerciórate de entender las condiciones del contrato.</p>`
        + `<div role="checkbox" aria-checked="false" tabindex="0">El titular de derechos de autor me ha dado permiso para usar el contenido.</div>`;
    } else if (p === 4) {
      html += `<div><textarea id="texto" aria-label="Incluye la información de tu licencia a continuación (obligatorio)" maxlength="2000"></textarea>`
        + `<span id="cont">0/2000</span></div><p>Haremos llegar este mensaje al reclamante para que lo revise.</p>`
        + `<p>Revisa las siguientes declaraciones y marca las casillas para confirmar que estás de acuerdo.</p>`
        + DECLARACIONES.map((t) => `<div role="checkbox" aria-checked="false" tabindex="0">${t}</div>`).join("")
        + `<div><label for="firma">Firma (obligatorio)</label><input id="firma" type="text"></div>`
        + `<small>Introduce tu nombre y apellidos, no el de tu empresa ni el de tu canal.</small><div id="aviso"></div>`;
    }
  }
  const pie = p === "verif" ? `<button id="cerrar">Cerrar</button>`
    : (p >= 2 ? `<button id="atras">Atrás</button>` : "")
      + (p === 0 ? `<button id="cancelar">Cancelar</button>` : "")
      + (p === 4 ? `<button id="enviar">Enviar</button>` : `<button id="continuar" aria-label="Continuar" ${p === 1 ? "" : "disabled"}>Continuar</button>`);
  dlg.innerHTML = html + `<div class="pie">${pie}</div>`;
  paso(p === 0 ? "seleccion" : p === "verif" ? "verificacion"
       : ["", "descripcion", "motivo", "detalles", "argumentos"][p]);
  enganchar(p);
}

function enganchar(p) {
  const continuar = $("continuar");
  if (continuar) continuar.onclick = () => {
    est.paso = (p === 0 && PIDE_VERIFICACION) ? "verif" : p + 1;
    pintar();
  };
  if ($("cancelar")) $("cancelar").onclick = () => { dlg.hidden = true; };
  if ($("cerrar")) $("cerrar").onclick = () => { dlg.hidden = true; };
  if (p === 0) $("c-impugnar").onclick = () => { continuar.disabled = false; };
  if (p === 2) dlg.querySelectorAll("[role=radio]").forEach((radio) => {
    radio.onclick = () => {
      dlg.querySelectorAll("[role=radio]").forEach((r) => r.setAttribute("aria-checked", "false"));
      radio.setAttribute("aria-checked", "true");
      est.motivo = radio.firstChild.textContent;
      continuar.disabled = false;
    };
  });
  if (p === 3) alternar(dlg.querySelector("[role=checkbox]"), () => {
    continuar.disabled = dlg.querySelector("[role=checkbox]").getAttribute("aria-checked") !== "true";
    est.permiso = !continuar.disabled;
  });
  if (p === 4) {
    dlg.querySelectorAll("[role=checkbox]").forEach((caja) => alternar(caja));
    $("texto").oninput = () => { $("cont").textContent = `${$("texto").value.length}/2000`; };
    $("enviar").onclick = enviar;
  }
}

async function enviar() {
  const marcadas = dlg.querySelectorAll("[role=checkbox][aria-checked=true]").length;
  const texto = $("texto").value, firma = $("firma").value;
  if (marcadas < 3) { $("aviso").textContent = "Marca todas las casillas anteriores para enviar"; return; }
  if (!texto.trim() || !firma.trim()) { $("aviso").textContent = "Faltan campos obligatorios"; return; }
  const respuesta = await fetch("/_enviar", {method: "POST", body: JSON.stringify({
    indice: est.indice, motivo: est.motivo, permiso: !!est.permiso, texto, firma, declaraciones: marcadas})});
  // Como en vivo: a veces Studio deja el formulario abierto tras "Enviar", sin ningún error.
  if ((await respuesta.json()).queda_abierto) return;
  dlg.hidden = true;
  dlg.innerHTML = "";
}

setTimeout(() => {
  if (!RECLAMOS.length) { $("vacio").textContent = "No se han encontrado reclamaciones en este vídeo"; return; }
  RECLAMOS.forEach((reclamo, indice) => {
    const fila = document.createElement("div");
    fila.className = "fila";
    fila.innerHTML = `<div>${reclamo.cancion}</div><div>Derechos de autor</div><div>Audio</div>`
      // Como en Studio: cada "Tomar medidas" es un <button> dentro de un <ytcp-button-shape>.
      + `<button>Ver detalles</button>`
      + `<ytcp-button-shape><button class="tomar" ${reclamo.impugnado ? "disabled" : ""}>Tomar medidas</button></ytcp-button-shape>`;
    // El clic lo atiende el envoltorio (también el que llega desde el <button> de adentro),
    // y un reclamo ya impugnado no abre nada.
    fila.querySelector("ytcp-button-shape").onclick = () => {
      if (fila.querySelector(".tomar").disabled) return;
      est = {indice, paso: 0, motivo: null, permiso: false};
      dlg.hidden = false;
      pintar();
    };
    $("lista").append(fila);
  });
}, CARGA_MS);
</script></body></html>"""


PAGINA_DE_ERROR = "<!doctype html><html><body><h1>Algo salió mal</h1></body></html>"


class StudioFalso:
    """La página de reclamos de un video y todo lo que la automatización mandó con "Enviar"."""

    def __init__(
        self,
        reclamos,
        titulo="Banda - Álbum (2026)",
        carga_ms=0,
        acepta_envios=True,
        verificacion_tras=None,
        sin_sesion=False,
        pagina_de_error=None,
        envios=None,
    ):
        self.reclamos = reclamos  # [{"cancion": ..., "impugnado": bool}]
        self.titulo = titulo
        self.carga_ms = carga_ms
        self.acepta_envios = acepta_envios  # False: Studio cierra el formulario pero no lo registra
        # Lo que hace Studio con el envío n (contado desde 1), si no es lo de siempre (cerrar el
        # formulario y registrarlo): "queda_abierto", "registra", "cargas_sin_mostrar" (cuántas
        # cargas de /claims siguen sin mostrarlo) y "reclamo_nuevo" (uno que llega con el envío).
        self.envios = envios or {}
        self.por_mostrar = []  # [índice, cargas de /claims que faltan para mostrarlo impugnado]
        # Con N impugnaciones ya aceptadas, el siguiente "Continuar" pide "Verifica que eres tú".
        self.verificacion_tras = verificacion_tras
        self.sin_sesion = sin_sesion
        # "siempre" o "tras_enviar": /claims responde una página de error, sin la lista ni el
        # aviso de que no hay reclamaciones.
        self.pagina_de_error = pagina_de_error
        self.enviados = []  # lo que llegó con "Enviar" (irreversible en Studio)
        self.pasos = []  # los pasos del formulario que se mostraron
        self.urls = []  # las páginas que se abrieron

    def impugnados(self):
        return sum(1 for reclamo in self.reclamos if reclamo["impugnado"])

    def mostrar_lo_registrado(self):
        """Una carga de /claims: muestra lo registrado cuya demora pasó y descuenta la del resto."""
        pendientes = []
        for indice, cargas in self.por_mostrar:
            if cargas:
                pendientes.append([indice, cargas - 1])
            else:
                self.reclamos[indice]["impugnado"] = True
        self.por_mostrar = pendientes

    def atender(self, ruta):
        url = ruta.request.url
        if url.endswith("/_paso"):
            self.pasos.append(ruta.request.post_data)
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        if url.endswith("/_enviar"):
            datos = json.loads(ruta.request.post_data)
            self.enviados.append(datos)
            envio = {"registra": self.acepta_envios, **self.envios.get(len(self.enviados), {})}
            if envio["registra"]:
                if envio.get("cargas_sin_mostrar"):
                    self.por_mostrar.append([datos["indice"], envio["cargas_sin_mostrar"]])
                else:
                    self.reclamos[datos["indice"]]["impugnado"] = True
            if "reclamo_nuevo" in envio:
                self.reclamos.append(envio["reclamo_nuevo"])
            respuesta = json.dumps({"queda_abierto": envio.get("queda_abierto", False)})
            return ruta.fulfill(status=200, body=respuesta, content_type="application/json")
        if self.sin_sesion:
            return ruta.fulfill(status=302, headers={"location": "https://accounts.google.com/v3/signin"})
        self.urls.append(url)
        if url.endswith("/copyright"):
            # Studio lleva /copyright a /claims desde el navegador, una vez cargada la página: un
            # `evaluate` que corre en ese momento falla con "Execution context was destroyed".
            redireccion = '<script>setTimeout(() => location.replace("/video/V1/claims"), 1500)</script>'
            return ruta.fulfill(status=200, body=redireccion, content_type="text/html; charset=utf-8")
        if self.pagina_de_error == "siempre" or (self.pagina_de_error == "tras_enviar" and self.enviados):
            return ruta.fulfill(status=200, body=PAGINA_DE_ERROR, content_type="text/html; charset=utf-8")
        if url.endswith("/claims"):
            self.mostrar_lo_registrado()
        verificacion = self.verificacion_tras is not None and self.impugnados() >= self.verificacion_tras
        html = (
            PAGINA.replace("__RECLAMOS__", json.dumps(self.reclamos))
            .replace("__TITULO__", json.dumps(self.titulo))
            .replace("__CARGA_MS__", str(self.carga_ms))
            .replace("__VERIFICACION__", json.dumps(verificacion))
        )
        return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")


class PaginaQueNavega:
    """Una página real cuyos primeros `evaluate` caen en plena navegación (como en Studio)."""

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


def reclamo(cancion, impugnado=False):
    return {"cancion": cancion, "impugnado": impugnado}


# Lo visto en vivo: tras "Enviar" el formulario queda abierto, sin error, y Studio no registra nada.
FORMULARIO_ABIERTO = {"queda_abierto": True, "registra": False}


class TestImpugnar(unittest.TestCase):
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
        # 10. inpunar_video.py guarda su evidencia junto a su propio archivo: se le da una
        # carpeta de prueba para que ningún test escriba dentro del repositorio.
        self.modulo = pa.cargar_script("10. inpunar_video.py", "inpunar_video_prueba")
        self.modulo.__file__ = str(self.tmp / "10. inpunar_video.py")
        self.evidencia = self.tmp / "evidencia"
        parches = [
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs"),
            mock.patch.object(pa, "DIR_EVIDENCIA_IMPUGNAR", self.evidencia),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "estado" / "pendientes.json"),
            mock.patch.object(pa, "PROGRAMADOS_JSONL", self.tmp / "estado" / "programados.jsonl"),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(pa, "PAUSA_IMPUGNAR_S", 0.1),
            mock.patch.object(pa, "ESPERA_ENVIO_IMPUGNAR_S", 0.5),
            mock.patch.object(pa, "INTENTOS_CONFIRMAR_IMPUGNACION", 2),
            mock.patch.object(pa, "ESPERA_CONFIRMAR_IMPUGNACION_S", 0.2),
            mock.patch.object(pa, "PAUSA_REINTENTO_NAVEGACION_S", 0.1),
            mock.patch.object(pa, "PAUSA_REINTENTO_IMPUGNAR_S", 0.1),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)
        (self.tmp / "estado").mkdir()
        registro = {"video_id": "V1", "titulo": "Banda - Álbum (2026)", "publica_en": "2028-11-02T10:00:00-06:00"}
        (self.tmp / "estado" / "programados.jsonl").write_text(
            json.dumps(registro, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def impugnar(self, studio, sin_enviar=False, evaluate_que_falla=0):
        self.contexto = self.navegador.new_context()
        self.addCleanup(self.contexto.close)
        self.contexto.route("https://studio.youtube.com/**", studio.atender)
        self.contexto.route(
            "https://accounts.google.com/**",
            lambda ruta: ruta.fulfill(status=200, body="<h1>Sign in</h1>", content_type="text/html"),
        )
        pagina = self.contexto.new_page()
        if evaluate_que_falla:
            pagina = PaginaQueNavega(pagina, evaluate_que_falla)

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        return pa.impugnar("V1", sin_enviar=sin_enviar, abrir_studio=abrir_studio, modulo=self.modulo)

    def test_impugna_cada_reclamo_habilitado_y_no_toca_el_que_ya_estaba_impugnado(self):
        # Defecto: impugnar siempre el primer botón de la lista (también el ya impugnado), o
        # parar tras el primero, dejaba reclamos sin disputar o repetía una impugnación.
        studio = StudioFalso([reclamo("Canción A", impugnado=True), reclamo("Canción B"), reclamo("Canción C")])
        resultado = self.impugnar(studio)
        self.assertEqual(
            (resultado["ok"], resultado["reclamos"], resultado["impugnados"], resultado["fallos"]),
            (True, 2, 2, []),
        )
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [1, 2])
        self.assertTrue(all(reclamo["impugnado"] for reclamo in studio.reclamos))
        self.assertEqual(resultado["titulo"], "Banda - Álbum (2026)")
        self.assertTrue(Path(resultado["captura"]).exists())

    def test_envia_motivo_licencia_el_texto_y_la_firma_configurados(self):
        # Defecto: enviar la impugnación con otro motivo, con el texto o la firma vacíos o con
        # una casilla sin marcar: es una declaración legal a nombre del usuario.
        studio = StudioFalso([reclamo("Canción A")])
        resultado = self.impugnar(studio)
        self.assertTrue(resultado["ok"], resultado)
        (enviado,) = studio.enviados
        self.assertEqual(enviado["motivo"], "Licencia")
        self.assertTrue(enviado["permiso"])
        self.assertEqual(enviado["texto"], self.modulo.mensaje)
        self.assertEqual(enviado["firma"], self.modulo.firma)
        self.assertEqual(enviado["declaraciones"], 3)

    def test_sin_enviar_recorre_el_formulario_y_no_envia_nada(self):
        # Defecto: que la prueba en seco dispare una impugnación real, o que se quede en un
        # paso intermedio y no demuestre que el formulario se completa.
        studio = StudioFalso([reclamo("Canción A"), reclamo("Canción B")])
        resultado = self.impugnar(studio, sin_enviar=True)
        self.assertEqual((resultado["ok"], resultado["sin_enviar"], resultado["listos"]), (True, True, 1))
        self.assertEqual(resultado["impugnados"], 0)
        self.assertEqual(studio.enviados, [])
        self.assertEqual(studio.impugnados(), 0)
        self.assertEqual(
            [paso for paso in studio.pasos if paso != "seleccion"],
            ["descripcion", "motivo", "detalles", "argumentos"],
        )
        # Sin pestañas olvidadas con un formulario a medio llenar.
        self.assertEqual(len(self.contexto.pages), 1)

    def test_si_studio_cierra_el_formulario_pero_no_lo_muestra_no_se_reintenta(self):
        # Defecto: dar la impugnación por hecha sin mirar Studio, o reintentar un envío que
        # Studio tomó (cerró el formulario) y quizá solo tarda en mostrar: la declaración legal
        # saldría dos veces.
        studio = StudioFalso([reclamo("Canción A"), reclamo("Canción B")], acepta_envios=False)
        resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual((resultado["reclamos"], resultado["impugnados"]), (2, 0))
        (fallo,) = resultado["fallos"]
        self.assertEqual(fallo["motivo"], "no_confirmado")
        self.assertEqual(len(studio.enviados), 1)  # ni reintento ni segundo reclamo

    def test_si_el_formulario_queda_abierto_se_reintenta_y_sigue_con_los_demas(self):
        # Defecto (visto en vivo): Studio a veces deja el formulario abierto tras "Enviar", sin
        # error y sin registrar nada; el paso frenaba ahí y el video quedaba a medio impugnar
        # (Mormon Mincers, 2 de 22). Reintentar ese reclamo tras una pausa lo registra.
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B"), reclamo("Canción C")],
            envios={2: FORMULARIO_ABIERTO},
        )
        resultado = self.impugnar(studio)
        self.assertEqual(
            (resultado["ok"], resultado["reclamos"], resultado["impugnados"], resultado["fallos"]),
            (True, 3, 3, []),
        )
        # Solo B sale dos veces: A, ya registrado, no se vuelve a impugnar.
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0, 1, 1, 2])
        self.assertEqual(studio.impugnados(), 3)

    def test_si_el_reintento_tampoco_sale_frena_sin_otro_intento(self):
        # Defecto: reintentar sin tope, o pasar al reclamo siguiente con uno sin impugnar.
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B")],
            envios={1: FORMULARIO_ABIERTO, 2: FORMULARIO_ABIERTO},
        )
        resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual((resultado["reclamos"], resultado["impugnados"]), (2, 0))
        self.assertEqual([fallo["motivo"] for fallo in resultado["fallos"]], ["no_confirmado"])
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0, 0])

    def test_si_studio_lo_registra_durante_la_pausa_no_lo_reenvia(self):
        # Defecto: reenviar sin volver a mirar /claims tras la pausa. Si Studio mostró tarde el
        # primer envío, el reintento salía para el reclamo siguiente y desde ahí cada envío
        # repetía un reclamo ya impugnado.
        demorado = {"queda_abierto": True, "registra": True, "cargas_sin_mostrar": 2}
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B"), reclamo("Canción C")], envios={1: demorado}
        )
        resultado = self.impugnar(studio)
        self.assertEqual(
            (resultado["ok"], resultado["reclamos"], resultado["impugnados"], resultado["fallos"]),
            (True, 3, 3, []),
        )
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0, 1, 2])

    def test_si_studio_lo_muestra_mientras_se_abre_el_reintento_no_lo_reenvia(self):
        # Defecto: mirar /claims solo tras la pausa. Si Studio mostraba el primer envío justo
        # después, la pestaña del reintento abría el reclamo siguiente y lo enviaba como si
        # fuera el mismo; con ese también demorado, salía dos veces.
        demorado = {"queda_abierto": True, "registra": True, "cargas_sin_mostrar": 3}
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B"), reclamo("Canción C")], envios={1: demorado}
        )
        resultado = self.impugnar(studio)
        self.assertEqual(
            (resultado["ok"], resultado["reclamos"], resultado["impugnados"], resultado["fallos"]),
            (True, 3, 3, []),
        )
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0, 1, 2])

    def test_si_llega_un_reclamo_nuevo_con_el_envio_no_se_reintenta(self):
        # Defecto: con el envío registrado y un reclamo nuevo, /claims muestra la misma cantidad
        # de reclamos impugnables. Un reintento impugnaba otro reclamo y, al final, también el
        # nuevo, que no estaba al empezar.
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B"), reclamo("Canción C")],
            envios={1: {"queda_abierto": True, "registra": True, "reclamo_nuevo": reclamo("Canción D")}},
        )
        resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual([fallo["motivo"] for fallo in resultado["fallos"]], ["no_confirmado"])
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0])

    def test_si_tras_enviar_la_lista_no_carga_no_se_da_por_impugnado(self):
        # Defecto: con un solo reclamo, una página de error sin filas se leía como "0 reclamos
        # impugnables" y la impugnación se daba por hecha aunque Studio no la registró.
        # El formulario queda abierto, como cuando sí se reintenta: sin la lista no hay prueba
        # de que el envío no salió, así que tampoco se reintenta.
        studio = StudioFalso(
            [reclamo("Canción A")], envios={1: FORMULARIO_ABIERTO}, pagina_de_error="tras_enviar"
        )
        with mock.patch.object(pa, "ESPERA_RESULTADO_RECLAMOS_S", 1):
            resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual((resultado["reclamos"], resultado["impugnados"]), (1, 0))
        self.assertEqual([fallo["motivo"] for fallo in resultado["fallos"]], ["no_confirmado"])
        self.assertEqual(len(studio.enviados), 1)

    def test_si_la_lista_no_carga_al_empezar_no_dice_que_no_hay_nada_que_impugnar(self):
        # Defecto: una página de error al empezar se informaba como "no tiene reclamos que se
        # puedan impugnar", y el usuario daba por cerrado un video que sigue con reclamos.
        studio = StudioFalso([reclamo("Canción A")], pagina_de_error="siempre")
        with mock.patch.object(pa, "ESPERA_RESULTADO_RECLAMOS_S", 1):
            resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertNotEqual(resultado.get("motivo"), "sin_reclamos_impugnables")
        self.assertEqual([fallo["motivo"] for fallo in resultado["fallos"]], ["lista_de_reclamos_no_disponible"])
        self.assertEqual((studio.pasos, studio.enviados), ([], []))

    def test_una_verificacion_de_identidad_falla_sin_enviar_y_deja_evidencia(self):
        # Defecto: esperar 10 minutos un 2FA que el servicio no puede pasar, o seguir al
        # formulario "a ciegas" y enviar algo.
        studio = StudioFalso([reclamo("Canción A"), reclamo("Canción B")], verificacion_tras=0)
        resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["impugnados"], 0)
        (fallo,) = resultado["fallos"]
        self.assertEqual(fallo["motivo"], "verificacion_de_identidad")
        self.assertEqual(studio.enviados, [])
        self.assertTrue(list(self.evidencia.glob("*.json")), "sin evidencia del fallo")

    def test_si_un_reclamo_falla_informa_los_anteriores_y_no_sigue_con_los_demas(self):
        # Defecto: perder la cuenta de lo ya impugnado cuando falla el reclamo siguiente, o
        # seguir con los demás sin entender por qué falló uno.
        studio = StudioFalso(
            [reclamo("Canción A"), reclamo("Canción B"), reclamo("Canción C")], verificacion_tras=1
        )
        resultado = self.impugnar(studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual((resultado["reclamos"], resultado["impugnados"]), (3, 1))
        self.assertEqual([fallo["motivo"] for fallo in resultado["fallos"]], ["verificacion_de_identidad"])
        self.assertEqual([enviado["indice"] for enviado in studio.enviados], [0])

    def test_un_video_sin_reclamos_habilitados_no_abre_ningun_formulario(self):
        # Defecto: abrir "Tomar medidas" de un reclamo ya impugnado, o confundir "nada que
        # impugnar" con un fallo.
        studio = StudioFalso([reclamo("Canción A", impugnado=True)])
        resultado = self.impugnar(studio)
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "sin_reclamos_impugnables"))
        self.assertEqual((resultado["reclamos"], resultado["impugnados"]), (0, 0))
        self.assertEqual(studio.pasos, [])

    def test_un_video_sin_ningun_reclamo_tampoco_tiene_nada_que_impugnar(self):
        # Defecto: esperar los 20 s completos o lanzar un error cuando la página dice que no
        # hay reclamaciones.
        resultado = self.impugnar(StudioFalso([]))
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "sin_reclamos_impugnables"))

    def test_espera_a_que_llegue_la_lista_de_reclamos(self):
        # Defecto: contar los botones al cargar la página, antes de que lleguen las filas,
        # daba "sin reclamos" para un video que sí los tiene.
        # La verificación de identidad corta el recorrido al instante: aquí solo importa el conteo.
        studio = StudioFalso([reclamo("Canción A")], carga_ms=6500, verificacion_tras=0)
        resultado = self.impugnar(studio, sin_enviar=True)
        self.assertEqual(resultado["reclamos"], 1, resultado)
        self.assertNotIn("motivo", resultado)

    def test_abre_claims_directo_y_no_pasa_por_la_redireccion_de_copyright(self):
        # Defecto (visto en vivo): abrir /copyright, que Studio lleva a /claims desde el
        # navegador; un `evaluate` que cae en esa navegación abortaba con "Execution context
        # was destroyed" antes de contar un solo reclamo. Vale para la pestaña que cuenta y
        # para la de cada reclamo.
        studio = StudioFalso([reclamo("Canción A")], verificacion_tras=0)
        resultado = self.impugnar(studio, sin_enviar=True)
        self.assertEqual(resultado["reclamos"], 1, resultado)
        self.assertEqual([url for url in studio.urls if url.endswith("/copyright")], [])
        self.assertGreaterEqual(len([url for url in studio.urls if url.endswith("/claims")]), 2)

    def test_un_evaluate_que_cae_en_una_navegacion_se_reintenta_y_cuenta_igual(self):
        # Defecto (visto en vivo): el error de navegación se volvía "error_navegador" y el
        # video quedaba sin impugnar aunque la página estuviera bien un segundo después.
        studio = StudioFalso([reclamo("Canción A")], verificacion_tras=0)
        resultado = self.impugnar(studio, sin_enviar=True, evaluate_que_falla=2)
        self.assertEqual(resultado["reclamos"], 1, resultado)
        self.assertNotIn("motivo", resultado)

    def test_sin_sesion_en_studio_no_impugna_nada_y_lo_dice(self):
        # Defecto: tomar la página de inicio de sesión por "sin reclamos" y avisar que no
        # había nada que impugnar.
        studio = StudioFalso([reclamo("Canción A")], sin_sesion=True)
        resultado = self.impugnar(studio)
        self.assertEqual((resultado["ok"], resultado["motivo"]), (False, "sin_sesion_en_studio"))
        # Un fallo en el primer paso igual informa el título que el servicio registró.
        self.assertEqual(resultado["titulo"], "Banda - Álbum (2026)")
        self.assertEqual(studio.enviados, [])


if __name__ == "__main__":
    unittest.main()
