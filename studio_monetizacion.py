"""Activa la monetización (anuncios) de un video en YouTube Studio.

La API de datos de YouTube no tiene ese campo: se hace en
studio.youtube.com/video/<id>/monetization con Playwright. Nunca lanza
errores hacia el servicio: devuelve un diccionario

    {"estado": "activada" | "ya_activa" | "fallo", "motivo": ..., "captura": ...}

Al activar, Studio pide el cuestionario de idoneidad para anuncios. Sus
respuestas son una declaración del canal y no se pueden cambiar después, así
que solo se envía si el panel lateral confirma "Adecuado para anuncios"; ante
cualquier otra cosa se cancela y el video queda como estaba. Si algo falla
después de enviarlo, el motivo es `cuestionario_ya_enviado`.
"""
import re
import time

from playwright.sync_api import Error as PlaywrightError

from studio_claims import dismiss_blocking_overlays

URL_MONETIZACION = "https://studio.youtube.com/video/{video_id}/monetization"
ESPERA_ESTADO_S = 45  # la página tarda en cargar: se espera por texto, no por un sleep fijo
ESPERA_PASO_S = 10
ESPERA_GUARDADO_S = 60  # Studio pasa por "Comprobando" un buen rato tras pulsar Guardar

P_ESTADO_ACTIVADA = re.compile(r"^\s*(activada|on)\s*$", re.IGNORECASE | re.MULTILINE)
P_COMPROBANDO = re.compile(r"^\s*(comprobando|checking)\s*$", re.IGNORECASE | re.MULTILINE)
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
# Decisión del usuario: sus videos son arte, así que el cuestionario siempre se responde
# "Nada de lo anterior" (sin elegir ningún radio de las categorías).
RESPUESTA_CUESTIONARIO = "Nada de lo anterior"
P_RESPUESTA_CUESTIONARIO = re.compile(rf"^\s*{re.escape(RESPUESTA_CUESTIONARIO)}\s*$", re.IGNORECASE)
P_ENVIAR = re.compile(r"^\s*(enviar|submit)\s*$", re.IGNORECASE)
P_CALIFICACION = re.compile(
    r"c[oó]mo has calificado tu v[ií]deo en general:\s*\n\s*(?P<calificacion>[^\n]+)", re.IGNORECASE
)
P_ADECUADO_PARA_ANUNCIOS = re.compile(r"^\s*adecuado para anuncios\s*$", re.IGNORECASE)

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


def _esperar_a_que_studio_guarde(pagina):
    """Espera a que Studio termine de guardar tras pulsar Guardar.

    Guardar se apaga al instante, pero el guardado llega después y la página pasa por
    "Comprobando". Recargar antes de eso lo cancela y el video queda como estaba.
    """
    _esperar(lambda: not _boton_habilitado(pagina, P_GUARDAR), ESPERA_PASO_S)
    if _esperar(lambda: P_COMPROBANDO.search(pagina.evaluate(JS_TEXTO)), ESPERA_PASO_S):
        _esperar(lambda: not P_COMPROBANDO.search(pagina.evaluate(JS_TEXTO)), ESPERA_GUARDADO_S)


def _cuestionario_abierto(pagina):
    return P_CUESTIONARIO.search(pagina.evaluate(JS_TEXTO)) is not None


def _cancelar_cuestionario(pagina, captura, motivo):
    pagina.get_by_role("dialog").get_by_role("button", name=P_CANCELAR).first.click(
        timeout=ESPERA_PASO_S * 1000
    )
    return _fallo(motivo, captura)


def _calificacion_del_panel(pagina):
    coincidencia = P_CALIFICACION.search(pagina.evaluate(JS_TEXTO))
    return coincidencia.group("calificacion") if coincidencia else ""


def _responder_cuestionario(pagina, captura, progreso):
    """Marca "Nada de lo anterior" y, solo si el panel dice "Adecuado para anuncios", envía."""
    dialogo = pagina.get_by_role("dialog")
    casilla = dialogo.get_by_role("checkbox", name=P_RESPUESTA_CUESTIONARIO)
    if not _esperar(lambda: casilla.count() > 0, ESPERA_PASO_S):
        return _cancelar_cuestionario(pagina, captura, "no_se_encontro_nada_de_lo_anterior")
    # El div del checkbox queda tapado por su contenedor, que es el que recibe el clic.
    casilla.first.click(force=True, timeout=ESPERA_PASO_S * 1000)
    if not _esperar(
        lambda: P_ADECUADO_PARA_ANUNCIOS.match(_calificacion_del_panel(pagina)), ESPERA_PASO_S
    ):
        return _cancelar_cuestionario(pagina, captura, "calificacion_no_adecuada")
    # Desde aquí no hay vuelta atrás: Studio no deja cambiar las respuestas enviadas.
    progreso["enviado"] = True
    dialogo.get_by_role("button", name=P_ENVIAR).first.click(timeout=ESPERA_PASO_S * 1000)
    if not _esperar(lambda: not _cuestionario_abierto(pagina), ESPERA_PASO_S):
        return _fallo("cuestionario_ya_enviado", captura, detalle="el cuestionario no se cerró")
    return None


def _activar(pagina, video_id, captura, espera_estado_s, progreso):
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
        fallo = _responder_cuestionario(pagina, captura, progreso)
        if fallo:
            return fallo
        siguiente = _esperar(lambda: _boton_habilitado(pagina, P_GUARDAR), ESPERA_PASO_S)
    if siguiente is None:
        return _fallo("no_se_pudo_guardar", captura)

    pagina.get_by_role("button", name=P_GUARDAR).first.click(timeout=ESPERA_PASO_S * 1000)
    _esperar_a_que_studio_guarde(pagina)
    # Se comprueba recargando: que el botón se apague no prueba que Studio lo guardó.
    if _abrir_y_leer_estado(pagina, video_id, espera_estado_s) != "activada":
        return _fallo("no_se_guardo", captura)
    return {"estado": "activada", "captura": str(captura)}


def activar_monetizacion(pagina, video_id, captura, espera_estado_s=ESPERA_ESTADO_S):
    progreso = {"enviado": False}
    try:
        resultado = _activar(pagina, video_id, captura, espera_estado_s, progreso)
    except PlaywrightError as exc:
        resultado = _fallo("error_navegador", captura, detalle=str(exc).splitlines()[0][:200])
    if progreso["enviado"] and resultado["estado"] == "fallo":
        # Lo que haya fallado, las respuestas ya salieron y no se pueden cambiar.
        causa = resultado["motivo"]
        resultado["motivo"] = "cuestionario_ya_enviado"
        resultado["detalle"] = causa if not resultado.get("detalle") else f"{causa}: {resultado['detalle']}"
    try:
        pagina.screenshot(path=str(captura))
    except PlaywrightError:
        resultado["captura"] = None
    return resultado
