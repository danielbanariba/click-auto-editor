"""Activa la monetización (anuncios) de un video en YouTube Studio.

La API de datos de YouTube no tiene ese campo: se hace en
studio.youtube.com/video/<id>/monetization con Playwright. Nunca lanza
errores hacia el servicio: devuelve un diccionario

    {"estado": "activada" | "ya_activa" | "fallo", "motivo": ..., "captura": ...}

Studio exige calificar el video (cuestionario de idoneidad para anuncios) al
activar. Esas respuestas son una declaración del canal que no se puede cambiar
después, así que este módulo NO las responde: cancela el cuestionario y deja el
video como estaba (motivo `cuestionario_de_idoneidad`).
"""
import re
import time

from playwright.sync_api import Error as PlaywrightError

from studio_claims import dismiss_blocking_overlays

URL_MONETIZACION = "https://studio.youtube.com/video/{video_id}/monetization"
ESPERA_ESTADO_S = 45  # la página tarda en cargar: se espera por texto, no por un sleep fijo
ESPERA_PASO_S = 10

P_ESTADO_ACTIVADA = re.compile(r"^\s*(activada|on)\s*$", re.IGNORECASE | re.MULTILINE)
P_ESTADO_DESACTIVADA = re.compile(r"^\s*(desactivada|off)\s*$", re.IGNORECASE | re.MULTILINE)
P_EDITAR_ESTADO = re.compile(
    r"(editar el estado de monetizaci[oó]n|edit.*monetization status)", re.IGNORECASE
)
P_RADIO_ACTIVADA = re.compile(r"^\s*(activada|on)\s*$", re.IGNORECASE)
P_CONTINUAR = re.compile(r"^\s*(siguiente|hecho|next|done)\s*$", re.IGNORECASE)
P_GUARDAR = re.compile(r"^\s*(guardar|save)\s*$", re.IGNORECASE)
P_CANCELAR = re.compile(r"^\s*(cancelar|cancel)\s*$", re.IGNORECASE)
P_CUESTIONARIO = re.compile(
    r"(cu[eé]ntanos qu[eé] aparece en tu v[ií]deo|tell us what.s in your video)", re.IGNORECASE
)

JS_TEXTO = "() => (document.body && document.body.innerText) || ''"


def _fallo(motivo, captura, detalle=None):
    resultado = {"estado": "fallo", "motivo": motivo, "captura": str(captura)}
    if detalle:
        resultado["detalle"] = detalle
    return resultado


def _esperar(condicion, timeout_s, intervalo_s=0.5):
    """Devuelve el primer valor verdadero de `condicion()`, o None al agotar el tiempo."""
    limite = time.monotonic() + timeout_s
    while True:
        valor = condicion()
        if valor:
            return valor
        if time.monotonic() >= limite:
            return None
        time.sleep(intervalo_s)


def _estado_en_pantalla(pagina):
    texto = pagina.evaluate(JS_TEXTO)
    if P_ESTADO_ACTIVADA.search(texto):
        return "activada"
    if P_ESTADO_DESACTIVADA.search(texto):
        return "desactivada"
    return None


def _abrir_y_leer_estado(pagina, video_id, espera_estado_s):
    pagina.goto(
        URL_MONETIZACION.format(video_id=video_id), wait_until="domcontentloaded", timeout=60000
    )
    return _esperar(lambda: _estado_en_pantalla(pagina), espera_estado_s)


def _boton_habilitado(pagina, nombre):
    boton = pagina.get_by_role("button", name=nombre)
    return boton.count() > 0 and boton.first.is_enabled()


def _cuestionario_abierto(pagina):
    return P_CUESTIONARIO.search(pagina.evaluate(JS_TEXTO)) is not None


def _activar(pagina, video_id, captura, espera_estado_s):
    estado = _abrir_y_leer_estado(pagina, video_id, espera_estado_s)
    if "accounts.google.com" in pagina.url:
        return _fallo("sin_sesion_en_studio", captura)
    if estado is None:
        texto = pagina.evaluate(JS_TEXTO)
        return _fallo("estado_desconocido", captura, detalle=texto[-300:].strip())
    if estado == "activada":
        return {"estado": "ya_activa", "captura": str(captura)}

    dismiss_blocking_overlays(pagina)
    pagina.get_by_role("button", name=P_EDITAR_ESTADO).first.click(timeout=ESPERA_PASO_S * 1000)
    pagina.get_by_role("radio", name=P_RADIO_ACTIVADA).first.check(timeout=ESPERA_PASO_S * 1000)
    pagina.get_by_role("button", name=P_CONTINUAR).first.click(timeout=ESPERA_PASO_S * 1000)

    # Tras elegir "Activada" Studio puede pedir el cuestionario o dejar guardar directo.
    siguiente = _esperar(
        lambda: "cuestionario"
        if _cuestionario_abierto(pagina)
        else ("guardar" if _boton_habilitado(pagina, P_GUARDAR) else None),
        ESPERA_PASO_S,
    )
    if siguiente == "cuestionario":
        pagina.get_by_role("dialog").get_by_role("button", name=P_CANCELAR).first.click(
            timeout=ESPERA_PASO_S * 1000
        )
        return _fallo("cuestionario_de_idoneidad", captura)
    if siguiente is None:
        return _fallo("no_se_pudo_guardar", captura)

    pagina.get_by_role("button", name=P_GUARDAR).first.click(timeout=ESPERA_PASO_S * 1000)
    _esperar(lambda: not _boton_habilitado(pagina, P_GUARDAR), ESPERA_PASO_S)
    # Se comprueba recargando: que el botón se apague no prueba que Studio lo guardó.
    if _abrir_y_leer_estado(pagina, video_id, espera_estado_s) != "activada":
        return _fallo("no_se_guardo", captura)
    return {"estado": "activada", "captura": str(captura)}


def activar_monetizacion(pagina, video_id, captura, espera_estado_s=ESPERA_ESTADO_S):
    try:
        resultado = _activar(pagina, video_id, captura, espera_estado_s)
    except PlaywrightError as exc:
        resultado = _fallo("error_navegador", captura, detalle=str(exc).splitlines()[0][:200])
    try:
        pagina.screenshot(path=str(captura))
    except PlaywrightError:
        resultado["captura"] = None
    return resultado
