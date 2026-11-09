"""Apela en YouTube Studio la restricción de edad y los anuncios limitados, con un texto en hebreo.

Lo usa `pipeline_agente.py apelar`, que el servicio lanza una vez por día. Recorre la lista
de contenido de Studio filtrada por cada problema y, por cada video que todavía no está en
el registro, abre su aviso (lo ya mandado sin resultado se mira en el mismo recorrido; ver
`apelar_videos`):

- Restricción de edad: la causa "YouTube ha limitado la audiencia...", "Ver todas las
  opciones" y la tarjeta "Apelar la decisión" (casilla, "Tu explicación" y "Enviar
  apelación"). "Entendido" confirma el problema y nunca se pulsa. Si el aviso solo trae
  "Revisar y tomar medidas", se abre por ahí, pero solo se anota una apelación ya resuelta:
  un formulario abierto así queda para revisar a mano, porque no dice de qué problema es.
- Anuncios limitados con revisión manual: "Ver comentarios", "Enviar comentarios" y el panel
  de comentarios de Google (texto obligatorio y "Enviar"). No es una apelación formal.

Apelar es una declaración a nombre del usuario: cada video se anota en el registro antes de
pulsar el botón de envío y nunca se le vuelve a mandar nada; después solo se lee qué
resolvió YouTube. Nunca lanza errores hacia el paso: devuelve un diccionario con lo que hizo.
"""
import re
import time
from urllib.parse import quote

from playwright.sync_api import Error as PlaywrightError

# Textos aprobados por el usuario (2026-11-03), en hebreo moderno, con la voz del canal.
TEXTO_APELACION_EDAD = (
    "הסרטון הזה הוא אלבום מלא של להקת מטאל קיצוני. מוצגות בו רק עטיפת האלבום הרשמית ורשימת "
    "השירים, לצד המוזיקה. העטיפה היא יצירת אמנות של הלהקה וחלק מהמסורת החזותית של הז'אנר, "
    "והערוץ מתעד ומשמר את סצנת המוזיקה הזו. בשל ההקשר האמנותי והתיעודי, אנו מבקשים להסיר את "
    "הגבלת הגיל. תודה."
)
TEXTO_COMENTARIOS_ANUNCIOS = (
    "אנו מבקשים בדיקה חוזרת של התאמת הסרטון הזה לפרסומות. זהו אלבום מלא של להקת מטאל: מוצגות "
    "בו רק עטיפת האלבום הרשמית ורשימת השירים, והמילים הפוגעניות בכותרת מצונזרות. העטיפה היא "
    "יצירת אמנות, האופיינית לאסתטיקה של הז'אנר. זהו תוכן מוזיקלי ואמנותי, המתאים למפרסמים "
    "שפונים לקהל של מוזיקת מטאל. תודה."
)

URL_STUDIO = "https://studio.youtube.com"
URL_LISTA = URL_STUDIO + "/channel/{canal}/videos/upload?filter={filtro}"
FILTROS = {
    "edad": '[{"name":"AGE_RESTRICTION","value":"AGE_RESTRICTED"}]',
    "anuncios": '[{"name":"MONETIZATION","value":["LIMITED"]}]',
}
# Lo que dice la columna "Avisos" de una fila de cada lista. Con el filtro recién puesto,
# Studio sigue mostrando un rato las filas sin filtrar: la lista está lista cuando todas
# las filas lo cumplen.
P_FILA_DE_LA_LISTA = {
    "edad": re.compile(r"mayores de 18", re.IGNORECASE),
    "anuncios": re.compile(r"ingresos publicitarios limitados|sin ingresos publicitarios", re.IGNORECASE),
}
P_MAYORES_DE_18 = re.compile(r"mayores de 18", re.IGNORECASE)
P_ANUNCIOS_LIMITADOS = re.compile(r"ingresos publicitarios limitados", re.IGNORECASE)
P_SIGUIENTE = re.compile(r"(página siguiente|next page)", re.IGNORECASE)
P_APELACION = re.compile(r"apelaci[oó]n", re.IGNORECASE)
P_RECHAZADA = re.compile(r"apelaci[oó]n rechazada", re.IGNORECASE)
P_ENVIAR = re.compile(r"^\s*enviar\s*$", re.IGNORECASE)

CAUSA_EDAD = "YouTube ha limitado la audiencia de este vídeo a usuarios de al menos 18 años"
REVISAR_Y_TOMAR_MEDIDAS = "Revisar y tomar medidas"
VER_TODAS_LAS_OPCIONES = "Ver todas las opciones"
ENVIAR_APELACION = "Enviar apelación"
VER_COMENTARIOS = "Ver comentarios"
ENVIAR_COMENTARIOS = "Enviar comentarios"
CAJA_DE_COMENTARIOS = "Explica tus comentarios (obligatorio)"
TIPOS = ("edad", "anuncios")
# Lo que se mandó (o pudo mandarse) y todavía no tiene resultado: se mira en cada corrida.
ESPERAN_RESULTADO = ("enviando", "enviado", "no_confirmado")

ESPERA_LISTA_S = 30
ESPERA_AVISO_S = 8
INTENTOS_AVISO = 4
ESPERA_DIALOGO_S = 45  # el diálogo "Revisar un problema" tardó más de 10 s en vivo
ESPERA_FORMULARIO_S = 20
ESPERA_CONFIRMACION_S = 20
PAUSA_S = 1.5
MAX_PAGINAS = 50
MAX_FALLOS = 3  # fallos seguidos (sin enviar nada) antes de dejar la corrida para mañana
TIEMPO_MAXIMO_S = 45 * 60
# Lo que pueden gastar los seguimientos dentro de TIEMPO_MAXIMO_S: el resto es de los envíos.
TIEMPO_MAXIMO_SEGUIMIENTOS_S = 15 * 60

JS_FILAS = """() => Array.from(document.querySelectorAll('ytcp-video-row')).map(fila => {
    const enlace = Array.from(fila.querySelectorAll('a[href]'))
        .map(a => a.href.match(/\\/video\\/([A-Za-z0-9_-]{11})/)).find(Boolean);
    const titulo = fila.querySelector('#video-title');
    const avisos = fila.querySelector('[class*="tablecell-restrictions"]');
    return {video_id: enlace ? enlace[1] : '', titulo: titulo ? titulo.innerText.trim() : '',
            avisos: avisos ? avisos.innerText.trim().replace(/\\s+/g, ' ') : ''};
})"""
# Studio deja en el DOM, ocultos, los diálogos y avisos ya cerrados, y `:visible` no sirve
# con los anfitriones de sus diálogos: lo visible se decide por el tamaño en pantalla.
JS_DIALOGO = """() => {
    const d = Array.from(document.querySelectorAll('yttou-dialog, ytcp-dialog, tp-yt-paper-dialog'))
        .find(e => { const r = e.getBoundingClientRect(); return r.width > 50 && r.height > 50; });
    return d ? (d.innerText || '').trim() : null;
}"""
# El elemento visible cuyo texto es exactamente `texto` (el más interno), y si su botón está
# habilitado. Los botones de Studio se pulsan por coordenadas: el clic de Playwright sobre
# el anfitrión no siempre llega.
JS_BOTON = """(texto) => {
    const e = Array.from(document.querySelectorAll('body *'))
        .filter(e => (e.innerText || '').trim() === texto)
        .map(e => [e, e.getBoundingClientRect()])
        .filter(([e, r]) => r.width > 4 && r.height > 4).pop();
    if (!e) return null;
    const boton = e[0].closest('ytcp-button, button, [role="button"]') || e[0];
    const deshabilitado = boton.hasAttribute('disabled') || boton.getAttribute('aria-disabled') === 'true';
    return {x: e[1].x + e[1].width / 2, y: e[1].y + e[1].height / 2, habilitado: !deshabilitado};
}"""
# La X del diálogo visible (un botón sin texto, con la etiqueta "Cerrar").
JS_CERRAR = """() => {
    const d = Array.from(document.querySelectorAll('yttou-dialog, ytcp-dialog, tp-yt-paper-dialog'))
        .find(e => { const r = e.getBoundingClientRect(); return r.width > 50 && r.height > 50; });
    if (!d) return null;
    const x = Array.from(d.querySelectorAll('[aria-label="Cerrar"], [aria-label="Close"]'))
        .map(e => e.getBoundingClientRect()).find(r => r.width > 4 && r.height > 4);
    return x ? {x: x.x + x.width / 2, y: x.y + x.height / 2} : null;
}"""
# La casilla de la tarjeta "Apelar la decisión": la única del diálogo visible.
JS_CASILLAS = """() => {
    const d = Array.from(document.querySelectorAll('yttou-dialog, ytcp-dialog, tp-yt-paper-dialog'))
        .find(e => { const r = e.getBoundingClientRect(); return r.width > 50 && r.height > 50; });
    if (!d) return [];
    return Array.from(d.querySelectorAll('[role="checkbox"]'))
        .map(c => [c, c.getBoundingClientRect()]).filter(([c, r]) => r.width > 4 && r.height > 4)
        .map(([c, r]) => ({x: r.x + r.width / 2, y: r.y + r.height / 2,
                           marcada: c.getAttribute('aria-checked') === 'true'}));
}"""


class ErrorDeApelacion(Exception):
    def __init__(self, motivo, detalle=None):
        super().__init__(motivo if detalle is None else f"{motivo}: {detalle}")
        self.motivo = motivo
        self.detalle = detalle


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


def _boton(pagina, texto):
    return pagina.evaluate(JS_BOTON, texto)


def _pulsar(pagina, texto):
    boton = _boton(pagina, texto)
    if not boton:
        raise ErrorDeApelacion("formulario_no_disponible", f"no aparece «{texto}»")
    pagina.mouse.click(boton["x"], boton["y"])


def _dialogo(pagina):
    return pagina.evaluate(JS_DIALOGO)


def _comprobar_sesion(pagina):
    if "accounts.google.com" in pagina.url:
        raise ErrorDeApelacion("sin_sesion_en_studio")


def canal_de_studio(pagina):
    """El id del canal, de la URL a la que Studio lleva al entrar."""
    if "/channel/" not in pagina.url:
        pagina.goto(URL_STUDIO, wait_until="domcontentloaded")
        _comprobar_sesion(pagina)
        pagina.wait_for_url(re.compile(r"/channel/UC"), timeout=60_000)
    return re.search(r"/channel/(UC[\w-]+)", pagina.url).group(1)


def _filas_de_la_lista(pagina, tipo, distinta_de=None):
    """Las filas, cuando todas son de la lista filtrada; [] si la lista no tiene ninguna.

    `distinta_de` es la primera fila de la página anterior: tras "Página siguiente", la
    tabla sigue mostrándola hasta que llegan las nuevas.
    """
    patron = P_FILA_DE_LA_LISTA[tipo]

    def listas():
        filas = pagina.evaluate(JS_FILAS)
        if not filas or not all(f["video_id"] and patron.search(f["avisos"]) for f in filas):
            return None
        if distinta_de and filas[0]["video_id"] == distinta_de:
            return None
        return filas

    filas = _esperar(listas, ESPERA_LISTA_S)
    if filas:
        return filas
    if not pagina.evaluate(JS_FILAS):
        return []
    raise ErrorDeApelacion("lista_no_disponible", f"la lista de «{tipo}» no terminó de cargar")


def espera_resultado(datos):
    return datos.get("estado") in ESPERAN_RESULTADO and not datos.get("resultado")


def candidatos(pagina, canal, tipo, registro, recorrido):
    """Los videos de la lista de `tipo` sin registro o con la apelación esperando resultado.

    Va página por página. `recorrido["vistos"]` junta todos los videos de la lista, y
    `recorrido["completo"]` queda en True solo si se llegó a la última página.
    """
    url = URL_LISTA.format(canal=canal, filtro=quote(FILTROS[tipo], safe=""))
    pagina.goto(url, wait_until="domcontentloaded")
    _comprobar_sesion(pagina)
    filas = _filas_de_la_lista(pagina, tipo)
    for _ in range(MAX_PAGINAS):
        recorrido["vistos"].update(fila["video_id"] for fila in filas)
        for fila in filas:
            datos = registro.get(f"{tipo}:{fila['video_id']}")
            if datos is not None and not espera_resultado(datos):
                continue
            # Un video con restricción de edad también sale en la lista de anuncios: su
            # apelación es la de edad.
            if tipo == "anuncios" and (
                P_MAYORES_DE_18.search(fila["avisos"]) or not P_ANUNCIOS_LIMITADOS.search(fila["avisos"])
            ):
                continue
            yield fila
        siguiente = pagina.get_by_role("button", name=P_SIGUIENTE).first
        if not siguiente.count() or siguiente.get_attribute("aria-disabled") == "true":
            recorrido["completo"] = True
            return
        siguiente.click()
        filas = _filas_de_la_lista(pagina, tipo, distinta_de=filas[0]["video_id"])
        if not filas:
            return


def abrir_aviso(pagina, video_id, *opciones):
    """Pasa el mouse por los avisos del video y pulsa en el globo la primera de `opciones` que aparece.

    Devuelve los botones del globo; si ninguna de `opciones` está entre ellos, no pulsa nada.
    """
    fila = pagina.locator("ytcp-video-row").filter(
        has=pagina.locator(f'a[href*="/video/{video_id}/"]')
    ).first
    celda = fila.locator('[class*="tablecell-restrictions"]').first
    ultimo = None
    for _ in range(INTENTOS_AVISO):
        try:
            celda.hover(timeout=10_000)
            globo = pagina.locator("ytcp-video-notices-tooltip-body:visible").first
            globo.wait_for(timeout=ESPERA_AVISO_S * 1000)
            globo.hover(timeout=5_000)
            botones = [t.strip() for t in globo.locator('button, [role="button"]').all_inner_texts() if t.strip()]
            boton = next((o for o in opciones if o in botones), None)
            if boton is None:
                return botones
            globo.locator('button, [role="button"]').filter(
                has_text=re.compile(rf"^\s*{re.escape(boton)}\s*$")
            ).first.click(timeout=5_000)
            return botones
        except PlaywrightError as exc:
            ultimo = exc
            pagina.mouse.move(5, 5)
            time.sleep(PAUSA_S)
    raise ErrorDeApelacion("aviso_no_disponible", str(ultimo).splitlines()[0][:150])


def cerrar_dialogos(pagina):
    """Cierra lo que quedó abierto; si no se cierra, la corrida no sigue sobre esa pantalla.

    "Revisar un problema" no se cierra con Escape (visto en vivo): se pulsa su X, y Studio
    pregunta "¿Seguro que quieres salir?", que se contesta con "Salir".
    """
    for _ in range(4):
        if not _dialogo(pagina):
            return
        cerrar = _boton(pagina, "Salir") or pagina.evaluate(JS_CERRAR) or _boton(pagina, "Cerrar")
        if cerrar:
            pagina.mouse.click(cerrar["x"], cerrar["y"])
        else:
            pagina.keyboard.press("Escape")
        time.sleep(PAUSA_S)
    if _dialogo(pagina):
        raise ErrorDeApelacion("dialogo_abierto", "Studio dejó un diálogo abierto")


def _leer_problema(pagina):
    """"apelable" si el diálogo del problema deja apelar; si no, su primer renglón ("Apelación …")."""

    def estado():
        if _boton(pagina, VER_TODAS_LAS_OPCIONES):
            return "apelable"
        texto = _dialogo(pagina)
        if texto and P_APELACION.search(texto.splitlines()[0]):
            return texto.splitlines()[0]
        return None

    visto = _esperar(estado, ESPERA_DIALOGO_S)
    if visto is None:
        raise ErrorDeApelacion("dialogo_desconocido", (_dialogo(pagina) or "no se abrió")[:150])
    return visto


def _apelar_edad(pagina, fila, sin_enviar, anotar):
    # Algunos globos traen "Revisar y tomar medidas" en lugar de la causa (visto en vivo en
    # apelaciones ya rechazadas): abre el diálogo del problema, que se clasifica igual.
    botones = abrir_aviso(pagina, fila["video_id"], CAUSA_EDAD, REVISAR_Y_TOMAR_MEDIDAS)
    visto = _leer_problema(pagina)
    if visto != "apelable":
        # "Apelación rechazada" o una ya enviada: Studio no deja apelar otra vez.
        return {"estado": "ya_apelado", "detalle": visto}
    if CAUSA_EDAD not in botones:
        # Ese botón es genérico y el formulario no dice de qué problema es: el texto de edad
        # no se manda a ciegas.
        return {"estado": "revisar_a_mano", "detalle": REVISAR_Y_TOMAR_MEDIDAS}
    _pulsar(pagina, VER_TODAS_LAS_OPCIONES)
    if not _esperar(lambda: _boton(pagina, ENVIAR_APELACION), ESPERA_FORMULARIO_S):
        raise ErrorDeApelacion("formulario_no_disponible", f"no aparece «{ENVIAR_APELACION}»")
    casillas = pagina.evaluate(JS_CASILLAS)
    if len(casillas) != 1:
        raise ErrorDeApelacion("formulario_no_disponible", f"{len(casillas)} casillas en el diálogo")
    if not casillas[0]["marcada"]:
        pagina.mouse.click(casillas[0]["x"], casillas[0]["y"])
    if not _esperar(lambda: [c for c in pagina.evaluate(JS_CASILLAS) if c["marcada"]], ESPERA_FORMULARIO_S):
        raise ErrorDeApelacion("formulario_no_disponible", "la casilla no quedó marcada")
    explicacion = pagina.locator("textarea:visible")
    if explicacion.count() != 1:
        raise ErrorDeApelacion("formulario_no_disponible", f"{explicacion.count()} cajas de texto")
    explicacion.fill(TEXTO_APELACION_EDAD)
    if not _esperar(lambda: (_boton(pagina, ENVIAR_APELACION) or {}).get("habilitado"), ESPERA_FORMULARIO_S):
        raise ErrorDeApelacion("formulario_no_disponible", f"«{ENVIAR_APELACION}» sigue deshabilitado")
    if sin_enviar:
        return {"estado": "listo"}
    anotar({"estado": "enviando"})
    _pulsar(pagina, ENVIAR_APELACION)
    cerrado = _esperar(lambda: pagina.locator("textarea:visible").count() == 0, ESPERA_CONFIRMACION_S)
    return {"estado": "enviado" if cerrado else "no_confirmado"}


def _caja_de_comentarios(pagina):
    """La caja de texto del panel de comentarios de Google y el iframe que la tiene.

    El panel es un iframe `about:blank` dentro del de google.com/tools/feedback (visto en
    vivo): se busca la caja por su etiqueta en todos los iframes.
    """
    for marco in pagina.frames:
        caja = marco.locator(f'textarea[aria-label="{CAJA_DE_COMENTARIOS}"]:visible')
        try:
            if caja.count() == 1:
                return marco, caja
        except PlaywrightError:  # un iframe que se recarga mientras se lo mira
            continue
    return None


def _cerrar_panel(marco):
    """Cierra el panel de Google con su X: tapa el diálogo de Studio que queda detrás."""
    try:
        if not marco.is_detached():
            marco.locator('button[aria-label="Cerrar"]:visible').first.click(timeout=5_000)
            time.sleep(PAUSA_S)
    except PlaywrightError as exc:
        print(f"No se pudo cerrar el panel de comentarios: {str(exc).splitlines()[0][:120]}")


def _apelar_anuncios(pagina, fila, sin_enviar, anotar):
    botones = abrir_aviso(pagina, fila["video_id"], VER_COMENTARIOS)
    if VER_COMENTARIOS not in botones:
        # Sin revisión manual (u otra cosa): ese formulario no se conoce y no se manda nada.
        return {"estado": "sin_opcion", "detalle": " · ".join(botones)[:150]}
    if not _esperar(lambda: _boton(pagina, ENVIAR_COMENTARIOS), ESPERA_DIALOGO_S):
        raise ErrorDeApelacion("dialogo_desconocido", (_dialogo(pagina) or "no se abrió")[:150])
    _pulsar(pagina, ENVIAR_COMENTARIOS)
    encontrado = _esperar(lambda: _caja_de_comentarios(pagina), ESPERA_FORMULARIO_S)
    if not encontrado:
        raise ErrorDeApelacion("formulario_no_disponible", "no apareció el panel de comentarios de Google")
    marco, caja = encontrado
    caja.fill(TEXTO_COMENTARIOS_ANUNCIOS)
    # El panel tiene un `div` envoltorio y el `button` con el mismo texto: se pulsa el botón.
    enviar = marco.locator("button:visible").filter(has_text=P_ENVIAR)
    if not _esperar(lambda: enviar.count() == 1 and enviar.is_enabled(), ESPERA_FORMULARIO_S):
        raise ErrorDeApelacion("formulario_no_disponible", "«Enviar» del panel no se habilitó")
    if sin_enviar:
        _cerrar_panel(marco)
        return {"estado": "listo"}
    anotar({"estado": "enviando"})
    enviar.click(timeout=10_000)

    def cerrado():
        try:
            return marco.is_detached() or caja.count() == 0
        except PlaywrightError:
            return True

    if _esperar(cerrado, ESPERA_CONFIRMACION_S):
        return {"estado": "enviado"}
    _cerrar_panel(marco)
    return {"estado": "no_confirmado"}


def _revisar_edad(pagina, fila):
    """Lo que dice Studio de una apelación de edad ya mandada. Solo lee el diálogo: no pulsa nada en él."""
    abrir_aviso(pagina, fila["video_id"], CAUSA_EDAD, REVISAR_Y_TOMAR_MEDIDAS)
    visto = _leer_problema(pagina)
    if P_RECHAZADA.search(visto):
        return {"resultado": "rechazada", "detalle": visto}
    return None  # en revisión, o algo que no es un resultado


APELAR = {"edad": _apelar_edad, "anuncios": _apelar_anuncios}
# Los anuncios no tienen rechazo formal: su único resultado es salir de la lista.
REVISAR = {"edad": _revisar_edad}


def apelar_videos(pagina, registro, anotar, limite, sin_enviar=False, capturar=None):
    """Apela los videos de las dos listas que no están en `registro`; devuelve el resumen.

    `anotar(clave, datos)` guarda un video en el registro. Se llama antes de pulsar el
    botón de envío, así un corte a mitad nunca hace que se mande dos veces. `limite` es
    cuántos envíos hace como máximo; `sin_enviar` llena el primer formulario de cada tipo
    y lo cierra sin enviarlo. `capturar(nombre)` guarda una captura y devuelve su ruta.

    Las apelaciones ya mandadas se miran en el mismo recorrido, también después del límite:
    "Apelación rechazada" es un rechazo, y un video que falta en una lista recorrida hasta
    el final dos corridas seguidas es una aceptada (la primera ausencia solo anota `ausente`
    en su entrada, y reaparecer en la lista la quita). Cada resultado se anota y sale una
    sola vez en `resumen["resultados"]`. Mirarlas tiene su propio tope de tiempo
    (`TIEMPO_MAXIMO_SEGUIMIENTOS_S`) dentro de los `TIEMPO_MAXIMO_S` de la corrida: gastado,
    se saltan y el resto del tiempo es de los envíos.
    """
    resumen = {"videos": [], "fallos": [], "resultados": []}
    inicio = time.monotonic()
    enviados = 0
    fallos_seguidos = 0
    en_seguimientos = 0.0  # segundos gastados leyendo resultados de apelaciones ya mandadas

    def anotar_resultado(clave, base, resultado, detalle=None):
        resumen["resultados"].append({**base, "resultado": resultado, "detalle": detalle})
        if not sin_enviar:
            fecha = time.strftime("%Y-%m-%dT%H:%M:%S")
            anotar(clave, {**registro[clave], "resultado": resultado, "detalle_resultado": detalle,
                           "fecha_resultado": fecha})
        print(f"{clave}: {resultado}")

    try:
        canal = canal_de_studio(pagina)
        for tipo in TIPOS:
            recorrido = {"vistos": set(), "completo": False}
            for fila in candidatos(pagina, canal, tipo, registro, recorrido):
                ahora = time.monotonic()
                if ahora - inicio > TIEMPO_MAXIMO_S:
                    return resumen
                clave = f"{tipo}:{fila['video_id']}"
                base = {"tipo": tipo, "video_id": fila["video_id"], "titulo": fila["titulo"]}
                seguimiento = clave in registro
                if seguimiento and tipo not in REVISAR:
                    continue
                if seguimiento and en_seguimientos > TIEMPO_MAXIMO_SEGUIMIENTOS_S:
                    print(f"{clave}: sin revisar, se acabó el tiempo de los seguimientos")
                    continue  # la lista se sigue recorriendo: los envíos conservan el resto de la corrida
                if not seguimiento and not sin_enviar and enviados >= limite:
                    continue  # hoy no se manda nada más, pero la lista se sigue mirando por los resultados
                fecha = time.strftime("%Y-%m-%dT%H:%M:%S")
                try:
                    if seguimiento:
                        hecho = REVISAR[tipo](pagina, fila)
                    else:
                        hecho = APELAR[tipo](
                            pagina, fila, sin_enviar, lambda datos: anotar(clave, {**base, **datos, "fecha": fecha})
                        )
                except ErrorDeApelacion as exc:
                    if exc.motivo in ("sin_sesion_en_studio", "lista_no_disponible", "dialogo_abierto"):
                        raise
                    captura = capturar(f"apelar_{fila['video_id']}") if capturar else None
                    fallo = {**base, "motivo": exc.motivo, "detalle": exc.detalle, "captura": captura}
                    if seguimiento:
                        fallo["revision"] = True  # no se pudo leer el resultado; no se mandó nada
                    resumen["fallos"].append(fallo)
                    print(f"No se pudo {'revisar' if seguimiento else 'apelar'} {clave}: {exc}")
                    cerrar_dialogos(pagina)
                    if seguimiento:
                        en_seguimientos += time.monotonic() - ahora
                        continue  # no poder leer un resultado no frena las apelaciones nuevas
                    fallos_seguidos += 1
                    if fallos_seguidos >= MAX_FALLOS:
                        resumen["motivo"] = "demasiados_fallos"
                        return resumen
                    continue
                if seguimiento:
                    if hecho:
                        anotar_resultado(clave, base, hecho["resultado"], hecho["detalle"])
                    cerrar_dialogos(pagina)
                    en_seguimientos += time.monotonic() - ahora
                    continue
                fallos_seguidos = 0
                if hecho["estado"] in ("enviado", "no_confirmado"):
                    enviados += 1
                if hecho["estado"] in ("enviado", "no_confirmado", "listo") and capturar:
                    hecho["captura"] = capturar(f"apelar_{fila['video_id']}")
                if not sin_enviar:
                    anotar(clave, {**base, **hecho, "fecha": fecha})
                resumen["videos"].append({**base, **hecho})
                print(f"{clave}: {hecho['estado']}")
                cerrar_dialogos(pagina)
                if hecho["estado"] == "listo":
                    break  # la prueba en seco llena un solo formulario de cada tipo
            # Sin filas no hay prueba de que la lista cargó: nada se da por aceptado.
            if recorrido["vistos"]:
                for clave, datos in list(registro.items()):
                    tipo_de_clave, video_id = clave.split(":", 1)
                    if tipo_de_clave != tipo or not espera_resultado(datos):
                        continue
                    if video_id in recorrido["vistos"]:
                        if "ausente" in datos and not sin_enviar:  # volvió a la lista
                            anotar(clave, {k: v for k, v in datos.items() if k != "ausente"})
                    elif recorrido["completo"]:
                        # Un video también sale de la lista si se borra, se vuelve privado o la
                        # lista cambia: la primera ausencia solo se marca, la segunda seguida es
                        # la aceptada.
                        if "ausente" in datos:
                            base = {"tipo": tipo, "video_id": video_id, "titulo": datos.get("titulo")}
                            anotar_resultado(clave, base, "aceptada")
                        elif not sin_enviar:
                            anotar(clave, {**datos, "ausente": time.strftime("%Y-%m-%dT%H:%M:%S")})
    except ErrorDeApelacion as exc:
        resumen.update(motivo=exc.motivo, detalle=exc.detalle)
    except PlaywrightError as exc:
        resumen.update(motivo="error_navegador", detalle=str(exc).splitlines()[0][:200])
    return resumen
