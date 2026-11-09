#!/usr/bin/env python3
"""Servicio que publica los álbumes de a uno, con aprobación por Telegram.

Corre siempre en segundo plano (systemd/publicar-albumes.service):

1. Renderiza el próximo álbum, o toma uno que ya esté en Renderizando/.
2. Lo verifica y manda UN mensaje con la captura y los botones Subir / Rechazar.
3. No sube nada ni renderiza otro hasta que el usuario aprieta un botón.
4. Con Subir: lo sube como privado, borra la carpeta y sigue con el próximo.
   En paralelo espera el procesado, revisa reclamos, activa la monetización y
   lo publica en el acto o lo programa (siempre a una hora de otro video).
5. Si un reclamante está en la lista negra, pregunta si elimina el video de YouTube.
   Si el video tiene reclamo y ninguno es de la lista negra, pregunta si lo impugna.
6. Una vez por día apela en hebreo la restricción de edad y los anuncios limitados de los
   videos del canal, sin preguntar, y después avisa qué mandó.

Cada paso lo hace pipeline_agente.py en un proceso aparte. Las preguntas
abiertas y los álbumes rechazados viven en data/agente/servicio.json, así un
reinicio no pierde nada.

Necesita en .env:
    PUBLICAR_TELEGRAM_TOKEN     token del bot (de @BotFather)
    PUBLICAR_TELEGRAM_CHAT_ID   chat del usuario: solo él puede apretar los botones
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

import pipeline_agente as pa
from config import PROJECT_ROOT

ESTADO_JSON = pa.DIR_AGENTE / "servicio.json"
ESPERA_TELEGRAM_S = 20
# Tope de la espera creciente con Telegram caído: la espera frena todo el ciclo,
# y las tareas terminadas se atienden recién al volver.
PAUSA_TELEGRAM_MAX_S = 60
REINTENTO_S = 30 * 60
INTENTOS_PROGRAMAR = 3  # fallos seguidos de `programar` antes de preguntar
# Veces que Telegram puede rechazar un aviso de resultado, con Telegram respondiendo, antes de
# descartarlo: un texto que nunca acepta no puede frenar los avisos que vienen detrás.
INTENTOS_AVISO = 5
INTERVALO_APELACIONES_S = 24 * 3600
# Cuántos álbumes apartados por ya publicados lista el aviso; del resto solo dice cuántos son.
MAX_APARTADOS_EN_AVISO = 10
ORIGEN_PUBLICADO = {"canal": "en el canal", "eliminar": "lo eliminaste", "subir": "lo subiste"}
LIMITE_LEYENDA = 1024
LIMITE_TEXTO = 4096
SIN_CONFIGURACION = 78  # EX_CONFIG: el unit de systemd no reinicia con este código

# Filas de botones. Eliminar va en su propia fila, lejos de Subir, para no apretarlo sin querer.
BOTONES = {
    "renderizar": [
        [("✅ Renderizar y subir", "renderizar")],
        [("🗑 Eliminar (ya está publicado)", "eliminar")],
    ],
    # La del modo lista: ➕ guarda el álbum sin renderizarlo y ▶️ renderiza la lista entera.
    # {n} es cuántos álbumes hay en la lista.
    "lista": [
        [("➕ A la lista", "a_la_lista")],
        [("🗑 Eliminar (ya está publicado)", "eliminar")],
        [("▶️ Renderizar la lista ({n})", "renderizar_lista")],
    ],
    # No quedan álbumes para sumar: solo queda renderizar la lista.
    "fin_lista": [[("▶️ Renderizar la lista ({n})", "renderizar_lista")]],
    "subir": [
        [("✅ Subir", "subir"), ("❌ Rechazar", "rechazar")],
        [("🗑 Eliminar (ya está publicado)", "eliminar")],
    ],
    "reclamo": [[("Sí, tiene reclamo", "si"), ("No tiene", "no")]],
    "reintentar": [[("🔁 Reintentar", "reintentar")]],
    "reprogramar": [[("🔁 Reintentar", "reintentar")]],
    # Borrar el video no se deshace: va en su propia fila, debajo de Conservar.
    "lista_negra": [[("Conservar", "conservar")], [("🗑️ Eliminar video", "eliminar_video")]],
    # Impugnar es una declaración legal a nombre del usuario y borrar no se deshace: cada
    # botón en su fila.
    "impugnar": [[("⚖️ Impugnar", "impugnar")], [("🗑️ Eliminar video", "eliminar_video")]],
    # Una copia retirada nunca se va a poder publicar: no hay nada que conservar ni reintentar.
    "copia_retirada": [[("🗑️ Eliminar video", "eliminar_video")]],
    "impugnacion_rechazada": [[("Conservar", "conservar")], [("🗑️ Eliminar video", "eliminar_video")]],
    # El resumen de /mapear_canal: impugna de a uno los videos de la lista. {n} es cuántos son.
    "impugnar_todos": [[("⚖️ Impugnar todos ({n})", "impugnar_todos")]],
    # Studio pidió "Verifica que eres tú": abre Brave visible en la PC para pasarla ahí.
    "impugnar_ventana": [[("🖥 Impugnar con ventana", "impugnar_ventana")]],
}
AVISO_BOTON = {
    "renderizar": "Lo renderizo y lo subo",
    "a_la_lista": "Va a la lista",
    "renderizar_lista": "Renderizo la lista",
    "subir": "Subiendo…",
    "rechazar": "No lo subo",
    "eliminar": "Borrando…",
    "si": "Programando…",
    "no": "Programando…",
    "reintentar": "Reintentando…",
    "conservar": "Lo conservo",
    "eliminar_video": "Borrando el video…",
    "impugnar": "Impugnando…",
    "impugnar_todos": "Los impugno de a uno; aviso al final",
    # Puede esperar el candado de Studio (`apelar` lo tiene hasta 90 min): la espera de 10 min
    # por reclamo empieza recién con el formulario abierto.
    "impugnar_ventana": "Abro Brave en la PC apenas Studio esté libre: pasá la verificación ahí",
}
# El menú de comandos del bot.
COMANDOS = [
    {"command": "aleatorio", "description": "Prende o apaga la elección al azar del próximo álbum"},
    {"command": "crear_lista", "description": "Junta los álbumes que apruebes y los renderiza todos juntos"},
    {"command": "mapear_canal", "description": "Revisa los videos con reclamo y junta los que falta impugnar"},
]
# Mientras haya una de estas preguntas abiertas no se renderiza otro álbum.
# "reprogramar" no frena: el video ya está subido y no depende del render.
PREGUNTAS_QUE_FRENAN = ("renderizar", "subir", "reintentar", "lista", "fin_lista")
PASOS_DEL_ALBUM = ("buscar", "renderizar", "verificar", "subir", "eliminar")
# Tiempo máximo de cada paso. Los cortables son idempotentes; los demás mueven o
# borran carpetas y cortarlos a medias dejaría un álbum a medio mover: solo se avisa.
LIMITE_PASO_S = 30 * 60  # verificar, eliminar y cualquier paso que no esté abajo
LIMITES_PASO_S = {
    "renderizar": pa.ESPERA_RENDER_MAX_S,
    "subir": 4 * 3600,
    "programar": 8 * 3600,  # espera el procesado de YouTube y el margen del reclamo
    "apelar": 90 * 60,  # tiene el candado de Studio: colgado, frenaría a `programar`
    "seguir_impugnaciones": 45 * 60,  # también; corta solo a los 30 min y sigue al otro día
    # También: corta a los 30 min de tener el candado, que puede esperar a `apelar` (90 min),
    # más un video que ya estaba leyendo.
    "mapear_canal": 150 * 60,
    # Cada reclamo espera hasta 10 min a que pases la verificación. Manda declaraciones
    # legales: no se corta, solo se avisa.
    "impugnar_con_ventana": 4 * 3600,
}
# `apelar` anota cada video antes de pulsar el envío: cortarlo nunca hace que se repita uno.
# `seguir_impugnaciones` y `mapear_canal` solo leen Studio. `buscar` aparta los publicados con
# un rename, que no queda a medias.
PASOS_CORTABLES = ("buscar", "verificar", "programar", "apelar", "seguir_impugnaciones", "mapear_canal")
# Frenan el lote de «Impugnar todos»: con ellos, los demás videos fallarían igual.
FRENAN_EL_LOTE = ("verificacion_de_identidad", "sin_sesion_en_studio", "canal_no_disponible")
# Lo que acompaña al botón «🖥 Impugnar con ventana».
INSTRUCCIONES_VENTANA = (
    "Tocá «🖥 Impugnar con ventana» cuando estés en la PC: se abre Brave en el escritorio. Pasá "
    "ahí «Verifica que eres tú» y no toques nada más; espero 10 minutos por reclamo e impugno "
    "todo en esa ventana."
)
GRACIA_CORTE_S = 10  # entre SIGTERM y SIGKILL
MOTIVOS_SUBIDA = {
    "limite_canal": "YouTube no deja subir más videos por hoy",
    "sin_credenciales": "no hay credenciales con cuota",
    "sin_metadata": "no se pudo armar la descripción",
}
MOTIVOS_MONETIZACION = {
    "cuestionario_ya_enviado": (
        "ya envié el cuestionario de anuncios (no se puede deshacer) pero la monetización no quedó activa"
    ),
    "calificacion_no_adecuada": "el cuestionario no dio «Adecuado para anuncios», no lo envié",
    "no_se_encontro_nada_de_lo_anterior": "no encontré «Nada de lo anterior» en el cuestionario, no lo envié",
    "sin_sesion_en_studio": "Studio no tiene la sesión iniciada",
    "canal_no_disponible": "YouTube pidió elegir canal y no pude entrar con «Daniel Banariba»",
    "no_se_guardo": "Studio no guardó el cambio",
    "no_se_pudo_guardar": "Studio no dejó guardar el cambio",
    "estado_desconocido": "la página de monetización no se parece a la esperada",
    "error_navegador": "falló el navegador",
}

MOTIVOS_IMPUGNACION = {
    "verificacion_de_identidad": "YouTube pidió verificar tu identidad y el servicio no puede pasar ese paso",
    "formulario_no_disponible": "el formulario de Studio no se parece al esperado",
    "no_confirmado": "Studio no mostró el reclamo como impugnado y no sé si salió",
    "lista_de_reclamos_no_disponible": "Studio no mostró la lista de reclamos; no se envió nada",
    "sin_sesion_en_studio": "Studio no tiene la sesión iniciada",
    "canal_no_disponible": "YouTube pidió elegir canal y no pude entrar con «Daniel Banariba»",
    "error_navegador": "falló el navegador",
    "reclamantes_no_disponibles": "no pude leer quién reclama, y sin eso no sé si está en la lista negra",
    "interrumpido": "el servicio se reinició a mitad de la impugnación; revisa en Studio qué salió "
    "(/mapear_canal lo vuelve a encontrar si le falta algo)",
}

MOTIVOS_APELACION = {
    "aviso_no_disponible": "no se abrió el aviso del video",
    "dialogo_desconocido": "el aviso de Studio no se parece al esperado",
    "formulario_no_disponible": "el formulario de Studio no se parece al esperado",
    "lista_no_disponible": "la lista de contenido de Studio no terminó de cargar",
    "dialogo_abierto": "Studio dejó un diálogo abierto y no seguí",
    "demasiados_fallos": "fallaron varios videos seguidos y lo dejé para mañana",
    "tiempo_agotado": "tardó demasiado y lo corté",
    "sin_sesion_en_studio": "Studio no tiene la sesión iniciada",
    "canal_no_disponible": "YouTube pidió elegir canal y no pude entrar con «Daniel Banariba»",
    "error_navegador": "falló el navegador",
}
TIPOS_APELACION = {"edad": "restricción de edad", "anuncios": "anuncios limitados"}

log = logging.getLogger("servicio_publicar")


def causa_legible(motivo):
    """El motivo de un paso de Studio o de la API, dicho para el usuario."""
    return (
        MOTIVOS_IMPUGNACION.get(motivo) or MOTIVOS_APELACION.get(motivo)
        or MOTIVOS_SUBIDA.get(motivo) or motivo
    )


def lineas_de_impugnacion(hechos):
    """Una línea por video del informe de un lote o de una ventana: lo impugnado y los fallos."""
    lineas = []
    for hecho in hechos:
        titulo = hecho.get("titulo") or hecho["video_id"]
        enlace = f"https://studio.youtube.com/video/{hecho['video_id']}/claims"
        if hecho.get("motivo") == "sin_reclamos_impugnables":
            lineas.append(f"• \"{titulo}\": no tenía reclamos que se pudieran impugnar")
            continue
        if hecho.get("motivo") == "interrumpido":
            lineas.append(f"• \"{titulo}\": {causa_legible('interrumpido')}\n  {enlace}")
            continue
        if hecho.get("reclamos"):
            linea = f"• \"{titulo}\": {hecho.get('impugnados', 0)} de {hecho['reclamos']} reclamos"
        else:
            linea = f"• \"{titulo}\": no pude impugnar"
        for fallo in hecho.get("fallos") or []:
            detalle = f" ({fallo['detalle']})" if fallo.get("detalle") else ""
            linea += f"\n  {causa_legible(fallo.get('motivo'))}{detalle}"
        if hecho.get("fallos"):
            linea += f"\n  {enlace}"
        lineas.append(linea)
    return lineas


def pidio_verificacion(hecho):
    return any(fallo.get("motivo") == "verificacion_de_identidad" for fallo in hecho.get("fallos") or [])


def partir_en_mensajes(lineas, limite=LIMITE_TEXTO):
    """Las líneas en textos de hasta `limite` caracteres, sin cortar ninguna: Telegram corta
    el resto de un mensaje más largo."""
    partes, actual = [], ""
    for linea in lineas:
        if actual and len(actual) + 1 + len(linea) > limite:
            partes.append(actual)
            actual = linea
        else:
            actual = f"{actual}\n{linea}" if actual else linea
    return [*partes, actual]


def limite_del_paso(paso):
    return LIMITES_PASO_S.get(paso, LIMITE_PASO_S)


def duracion_legible(segundos):
    horas, minutos = divmod(int(segundos) // 60, 60)
    if not horas:
        return f"{minutos} min"
    return f"{horas} h {minutos} min" if minutos else f"{horas} h"


def acciones(tipo):
    return [accion for fila in BOTONES[tipo] for _, accion in fila]


class ErrorTelegram(Exception):
    pass


ERRORES_TELEGRAM = (ErrorTelegram, ValueError)


class Telegram:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = str(chat_id)

    def llamar(self, metodo, limite_s=30, archivos=None, **datos):
        datos = {
            clave: json.dumps(valor) if isinstance(valor, (dict, list)) else valor
            for clave, valor in datos.items()
        }
        try:
            respuesta = requests.post(
                f"https://api.telegram.org/bot{self.token}/{metodo}",
                data=datos,
                files=archivos,
                timeout=limite_s,
            )
        except requests.RequestException as exc:
            # El mensaje de requests trae la URL, y la URL trae el token.
            raise ErrorTelegram(str(exc).replace(self.token, "<token>")) from None
        cuerpo = respuesta.json()
        if not cuerpo.get("ok"):
            raise ErrorTelegram(f"{metodo}: {cuerpo.get('description')}")
        return cuerpo["result"]

    def actualizaciones(self, offset, espera):
        return self.llamar(
            "getUpdates",
            limite_s=espera + 15,
            offset=offset,
            timeout=espera,
            allowed_updates=["callback_query", "message"],
        )

    def registrar_comandos(self):
        self.llamar("setMyCommands", commands=COMANDOS)

    def mandar(self, texto, botones=None, foto=None):
        datos = {"chat_id": self.chat_id}
        if botones:
            datos["reply_markup"] = {
                "inline_keyboard": [
                    [{"text": t, "callback_data": d} for t, d in fila] for fila in botones
                ]
            }
        if foto and Path(foto).is_file():
            try:
                with open(foto, "rb") as archivo:
                    return self.llamar(
                        "sendPhoto",
                        limite_s=120,
                        archivos={"photo": archivo},
                        caption=texto[:LIMITE_LEYENDA],
                        **datos,
                    )["message_id"]
            except ErrorTelegram as exc:
                # Una foto que Telegram rechaza no puede frenar la pregunta: va sin foto.
                log.warning("No se pudo mandar la foto, va solo el texto: %s", exc)
        return self.llamar("sendMessage", text=texto[:LIMITE_TEXTO], **datos)["message_id"]

    def quitar_botones(self, message_id):
        self.llamar(
            "editMessageReplyMarkup",
            chat_id=self.chat_id,
            message_id=message_id,
            reply_markup={"inline_keyboard": []},
        )

    def responder_boton(self, consulta_id, texto):
        self.llamar("answerCallbackQuery", callback_query_id=consulta_id, text=texto)


def ultimo_json(ruta):
    with open(ruta, "r", encoding="utf-8", errors="replace") as archivo:
        lineas = archivo.read().splitlines()
    for linea in reversed(lineas):
        if linea.startswith("{"):
            try:
                return json.loads(linea)
            except json.JSONDecodeError:
                continue
    return None


class Proceso:
    """Un paso de pipeline_agente.py corriendo aparte; su JSON final es el resultado."""

    def __init__(self, argumentos, comando=None):
        pa.DIR_LOGS.mkdir(parents=True, exist_ok=True)
        marca = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.salida = pa.DIR_LOGS / f"servicio_{argumentos[0]}_{marca}.out"
        with open(self.salida, "w", encoding="utf-8") as archivo:
            self.popen = subprocess.Popen(
                comando or [sys.executable, "pipeline_agente.py", *argumentos],
                cwd=PROJECT_ROOT,
                stdout=archivo,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # grupo propio: cortarlo alcanza a Playwright y Chromium
            )

    def cortar(self):
        """Mata el paso y todo lo que lanzó: SIGTERM al grupo y, si sigue vivo, SIGKILL."""
        grupo = self.popen.pid
        try:
            os.killpg(grupo, signal.SIGTERM)
            self.popen.wait(timeout=GRACIA_CORTE_S)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(grupo, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.popen.wait()

    def resultado(self):
        if self.popen.poll() is None:
            return None
        return ultimo_json(self.salida) or {
            "ok": False,
            "motivo": "sin_respuesta",
            "codigo": self.popen.returncode,
            "log": str(self.salida),
        }


def valor(argumentos, opcion):
    return argumentos[list(argumentos).index(opcion) + 1]


def busqueda_en_youtube(resultado):
    """La captura de la búsqueda va en un mensaje antes de la pregunta; sin captura, el enlace."""
    busqueda = resultado.get("busqueda")
    if resultado.get("captura_busqueda"):
        return [{"texto": f"🔎 Búsqueda en YouTube: {busqueda}", "foto": resultado["captura_busqueda"]}], ""
    if busqueda:
        return [], f"\n\n🔎 https://www.youtube.com/results?search_query={quote(busqueda)}"
    return [], ""


def cantidad_de_albumes(n):
    return f"{n} álbum" if n == 1 else f"{n} álbumes"


def carpetas_de_origen():
    """Las carpetas de donde toma álbumes el servicio: «a/, b/ ni c/»."""
    nombres = [f"{origen.name}/" for origen in pa.DIRS_ORIGEN_RENDER]
    if len(nombres) == 1:
        return nombres[0]
    return f"{', '.join(nombres[:-1])} ni {nombres[-1]}"


class Servicio:
    def __init__(self, telegram, lanzar, estado_json=ESTADO_JSON, reloj=time.monotonic, ahora=time.time):
        self.telegram = telegram
        self.lanzar = lanzar
        self.ruta_estado = Path(estado_json)
        self.reloj = reloj
        self.ahora = ahora  # hora del reloj de pared: la de la última apelación sobrevive reinicios
        # rechazados y subidos: álbumes que siguen en Renderizando/ pero no se vuelven a ofrecer.
        # aceptados: álbumes que el usuario mandó renderizar; se suben sin volver a preguntar.
        # aleatorio: el /aleatorio de Telegram; el próximo álbum se elige al azar.
        # modo_lista y lista: el /crear_lista; lo aprobado con ➕ espera ahí hasta que se toca ▶️.
        # despues_de_lista: el álbum que estaba en pantalla al tocar ▶️; se pregunta al terminar.
        # lote_impugnar: el «Impugnar todos» en curso: los videos que faltan (el primero se está
        # impugnando) y lo que salió con los anteriores.
        # avisos_pendientes: avisos de resultados ya anotados en apelaciones.json o
        # impugnaciones.json que Telegram todavía no recibió: [{"texto", "intentos"}].
        self.estado = {
            "offset": 0, "preguntas": {}, "rechazados": [], "subidos": [], "aceptados": [],
            "aleatorio": False, "modo_lista": False, "lista": [], "despues_de_lista": None,
            "lote_impugnar": None, "avisos_pendientes": [],
        }
        self.tareas = []
        self.inicios = {}  # proceso -> momento en que se lanzó
        self.largos_avisados = set()  # procesos que pasaron su límite y ya se avisó
        self.reintentos = []
        self.fallos_programar = {}  # video_id -> fallos seguidos que llegaron al reintento genérico
        self.album_pausado_hasta = 0.0
        self.avisos = set()
        # Álbumes publicados que no se pudieron mover: se avisan una vez (avisos se vacía en cada render).
        self.no_apartados = set()
        self.pausa_error_s = 5
        self.fallos_telegram = 0  # fallos seguidos de getUpdates; 0 es que Telegram responde
        self.telegram_caido_desde = None

    # -- estado -------------------------------------------------------------

    def iniciar(self):
        if self.ruta_estado.exists():
            with open(self.ruta_estado, "r", encoding="utf-8") as archivo:
                self.estado.update(json.load(archivo))
        con_pregunta = {p.get("video_id") for p in self.estado["preguntas"].values()}
        for pendiente in pa.leer_pendientes():
            if pendiente["video_id"] not in con_pregunta:
                self.lanzar_paso("programar", "--video-id", pendiente["video_id"], "--esperar")
        # El `impugnar` del lote murió con el reinicio. Pudo haber enviado un reclamo que Studio
        # todavía no muestra impugnado: relanzarlo podría impugnarlo dos veces. Queda como
        # interrumpido y el lote sigue con el próximo.
        if self.estado["lote_impugnar"] and self.estado["lote_impugnar"]["videos"]:
            self.anotar_en_el_lote({"ok": False, "motivo": "interrumpido"})
        self.enviar_preguntas()
        self.enviar_avisos_pendientes()

    def guardar(self):
        self.ruta_estado.parent.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta_estado.with_name(f"{self.ruta_estado.name}.tmp")
        with open(temporal, "w", encoding="utf-8") as archivo:
            json.dump(self.estado, archivo, ensure_ascii=False, indent=2)
        temporal.replace(self.ruta_estado)

    # -- ciclo --------------------------------------------------------------

    def paso(self):
        self.revisar_tareas()
        self.revisar_reintentos()
        self.revisar_apelaciones()
        self.enviar_preguntas()
        self.enviar_avisos_pendientes()
        self.avanzar_album()
        self.atender_telegram()

    def lanzar_paso(self, *argumentos):
        log.info("Lanzo: %s", " ".join(argumentos))
        proceso = self.lanzar(argumentos)
        self.inicios[proceso] = self.reloj()
        self.tareas.append((argumentos, proceso))

    def revisar_tareas(self):
        for tarea in list(self.tareas):
            argumentos, proceso = tarea
            resultado = proceso.resultado()
            if resultado is None:
                resultado = self.vigilar_tiempo(argumentos, proceso)
            if resultado is None:
                continue
            self.tareas.remove(tarea)
            self.inicios.pop(proceso, None)
            self.largos_avisados.discard(proceso)
            log.info("Terminó %s: %s", argumentos[0], json.dumps(resultado, ensure_ascii=False)[:500])
            getattr(self, f"tras_{argumentos[0]}")(resultado, argumentos)

    def vigilar_tiempo(self, argumentos, proceso):
        """Un paso que pasó su límite se corta si es seguro; si no, se avisa una vez y se espera."""
        paso = argumentos[0]
        limite = limite_del_paso(paso)
        corrido = self.reloj() - self.inicios[proceso]
        if corrido <= limite:
            return None
        if paso in PASOS_CORTABLES:
            log.warning("%s pasó los %s: lo corto", " ".join(argumentos), duracion_legible(limite))
            proceso.cortar()
            return {"ok": False, "motivo": "tiempo_agotado", "log": str(proceso.salida)}
        if proceso not in self.largos_avisados:
            self.largos_avisados.add(proceso)
            self.avisar(
                f"⏳ {' '.join(argumentos)} lleva {duracion_legible(corrido)} corriendo "
                f"(el máximo es {duracion_legible(limite)}). No lo corto porque mueve o borra "
                f"carpetas; sigo esperando.\nLog: {proceso.salida}"
            )
        return None

    def revisar_reintentos(self):
        ahora = self.reloj()
        for reintento in [r for r in self.reintentos if r[0] <= ahora]:
            self.reintentos.remove(reintento)
            self.lanzar_paso(*reintento[1])

    def revisar_apelaciones(self):
        """Lanza `apelar` una vez por día; la hora de la última corrida queda en el estado."""
        if any(argumentos[0] == "apelar" for argumentos, _ in self.tareas):
            return
        if self.ahora() - self.estado.get("ultima_apelacion", 0) < INTERVALO_APELACIONES_S:
            return
        self.estado["ultima_apelacion"] = self.ahora()
        self.guardar()
        self.lanzar_paso("apelar")

    def avanzar_album(self):
        if any(argumentos[0] in PASOS_DEL_ALBUM for argumentos, _ in self.tareas):
            return
        abiertas = [p for p in self.estado["preguntas"].values() if p["tipo"] in PREGUNTAS_QUE_FRENAN]
        # Un chequeo que falló en un álbum aceptado pregunta sin frenar al resto de la lista.
        sigue_la_lista = self.estado["aceptados"]
        if any(p["tipo"] != "subir" or not p.get("aceptado") or not sigue_la_lista for p in abiertas):
            return
        if self.reloj() < self.album_pausado_hasta:
            return
        # Un render que lanzó otro (a mano o una sesión vieja) puede estar
        # moviendo el mp4 a Renderizando/: se espera a que termine.
        if pa.render_en_curso():
            return
        descartados = set(self.estado["rechazados"]) | set(self.estado["subidos"])
        descartados |= {Path(p["carpeta"]).name for p in abiertas if p.get("carpeta")}
        listos = [c for c in pa.albumes_por_aprobar() if Path(c).name not in descartados]
        if listos:
            self.lanzar_paso("verificar", "--carpeta", listos[0])
            return
        # Un álbum aceptado que sigue en su carpeta de origen se renderiza sin volver a preguntar
        # (por ejemplo, después de un reinicio a mitad del render).
        for nombre in list(self.estado["aceptados"]):
            if any((origen / nombre).is_dir() for origen in pa.DIRS_ORIGEN_RENDER):
                self.lanzar_paso("renderizar", "--carpeta", nombre)
                return
            # Ya no está: el render lo movió a Error404 o alguien lo sacó de la carpeta.
            self.estado["aceptados"].remove(nombre)
            self.guardar()
        # La lista terminó: lo nuevo espera a que se conteste la pregunta que quedó abierta.
        if abiertas:
            return
        if self.preguntar_despues_de_lista():
            return
        if pa.siguiente_album() is not None:
            self.avisos.discard("sin_albumes")
            argumentos = ["--aleatorio"] if self.estado["aleatorio"] else []
            # Con "=": un nombre que empieza con "-" no se lee como otra opción.
            argumentos += [f"--excluir={nombre}" for nombre in self.estado["lista"]]
            self.lanzar_paso("buscar", *argumentos)
        else:
            self.avisar_una_vez(
                "sin_albumes",
                f"📭 No quedan álbumes en {carpetas_de_origen()}. Sigo solo cuando aparezca uno.",
            )

    # -- resultados de cada paso -------------------------------------------

    def avisar_apartados(self, apartados):
        """Lo que el render movió a Ya publicados/ sin renderizar, y lo que no pudo mover."""
        movidos = [a for a in apartados if "error" not in a]
        if movidos:
            lineas = [
                f"• {a['album']} → {a['como']} ({ORIGEN_PUBLICADO.get(a['origen'], a['origen'])})"
                for a in movidos[:MAX_APARTADOS_EN_AVISO]
            ]
            if len(movidos) > MAX_APARTADOS_EN_AVISO:
                lineas.append(f"y {len(movidos) - MAX_APARTADOS_EN_AVISO} más.")
            self.avisar(
                f"📦 Ya estaban publicados: los moví a {pa.DIR_YA_PUBLICADOS.name}/ sin renderizar.\n"
                + "\n".join(lineas)
            )
        for a in apartados:
            if "error" in a and a["album"] not in self.no_apartados:
                self.no_apartados.add(a["album"])
                self.avisar(
                    f"⚠️ {a['album']}: {a['error']}\nYa está publicado y no pude moverlo a "
                    f"{pa.DIR_YA_PUBLICADOS.name}/. Sigue en su carpeta y no lo renderizo."
                )

    def tras_buscar(self, resultado, argumentos):
        apartados = resultado.get("ya_publicados") or []
        self.avisar_apartados(apartados)
        motivo = resultado.get("motivo")
        if resultado.get("ok"):
            self.avisos.clear()
            previas, enlace = busqueda_en_youtube(resultado)
            self.preguntar_por_album(
                resultado["album"], resultado["carpeta"], resultado.get("portada"), previas, enlace
            )
        elif motivo == "sin_albumes":
            if any("error" in a for a in apartados):
                # El álbum que no se pudo mover sigue en la cola: sin pausa, se relanzaría la
                # búsqueda en cada vuelta.
                self.album_pausado_hasta = self.reloj() + REINTENTO_S
            if self.estado["modo_lista"] and self.estado["lista"]:
                # Los que quedan ya están en la lista: esta pregunta frena la búsqueda.
                cantidad = cantidad_de_albumes(len(self.estado["lista"]))
                self.preguntar("fin_lista", f"📭 No quedan más álbumes. En la lista hay {cantidad}.")
        else:
            texto = f"⚠️ No pude buscar el próximo álbum ({motivo}). Reintento en 30 min."
            if resultado.get("log"):
                texto += f"\nLog: {resultado['log']}"
            self.avisar_una_vez(f"buscar_{motivo}", texto)
            self.album_pausado_hasta = self.reloj() + REINTENTO_S

    def tras_renderizar(self, resultado, argumentos):
        apartados = resultado.get("ya_publicados") or []
        self.avisar_apartados(apartados)
        motivo = resultado.get("motivo")
        if motivo == "sin_albumes" and any("error" in a for a in apartados):
            # El álbum que no se pudo mover sigue en la cola: sin pausa, se relanzaría el render
            # en cada vuelta.
            self.album_pausado_hasta = self.reloj() + REINTENTO_S
        if resultado.get("ok"):
            self.avisos.clear()
            # Un render nuevo con el nombre de uno descartado antes se vuelve a preguntar.
            nombre = Path(resultado["carpeta"]).name
            for lista in ("rechazados", "subidos"):
                if nombre in self.estado[lista]:
                    self.estado[lista].remove(nombre)
            self.guardar()
            self.lanzar_paso("verificar", "--carpeta", resultado["carpeta"])
        elif motivo in ("sin_albumes", "render_en_curso"):
            return
        elif motivo == "album_no_encontrado":
            self.avisar(f"⚠️ No encontré {resultado.get('album')} en {carpetas_de_origen()}; sigo con el próximo.")
        elif motivo == "render_fallido":
            texto = f"❌ Falló el render de {resultado.get('album')}."
            if resultado.get("log_final"):
                texto += f"\n\n{resultado['log_final'][-1500:]}"
            if resultado.get("movido_a_error404"):
                self.avisar(f"{texto}\n\nLo moví a Error404; sigo con el próximo.")
            else:
                self.preguntar(
                    "reintentar", f"{texto}\n\nSigue en su carpeta. Arreglalo y apretá Reintentar."
                )
        else:
            detalle = resultado.get("detalle") or resultado.get("libre_gb") or ""
            self.avisar_una_vez(
                f"render_{motivo}",
                f"⚠️ No pude renderizar ({motivo} {detalle}). Reintento cada 30 min.",
            )
            self.album_pausado_hasta = self.reloj() + REINTENTO_S

    def tras_verificar(self, resultado, argumentos):
        carpeta = valor(argumentos, "--carpeta")
        album = Path(carpeta).name
        if resultado.get("motivo") == "tiempo_agotado":
            # Sigue aceptado: al pasar la pausa se verifica otra vez y, si pasa, se sube.
            self.avisar(
                f"⚠️ No pude verificar {album}: el paso pasó los "
                f"{duracion_legible(limite_del_paso('verificar'))} y lo corté. "
                f"Reintento en 30 min.\nLog: {resultado.get('log')}"
            )
            self.album_pausado_hasta = self.reloj() + REINTENTO_S
            return
        aceptado = album in self.estado["aceptados"]
        if aceptado:
            self.estado["aceptados"].remove(album)
            self.guardar()
        if resultado.get("motivo") == "sin_video":
            self.estado["rechazados"].append(album)
            self.guardar()
            self.avisar(f"⚠️ {album}: el render no dejó el mp4. Lo dejo de lado.")
            return
        if aceptado and resultado.get("ok"):
            # El usuario ya dijo que no está publicado y los chequeos pasaron.
            self.lanzar_paso("subir", "--carpeta", carpeta)
            return
        texto = resultado.get("resumen") or album
        if not resultado.get("ok"):
            texto += "\n\n⚠️ Hay tests fallidos."
        previas, enlace = busqueda_en_youtube(resultado)
        texto += enlace
        self.preguntar(
            "subir",
            f"{texto}\n\n¿Lo subo?",
            foto=resultado.get("captura"),
            carpeta=carpeta,
            previas=previas,
            aceptado=aceptado,
        )

    def tras_subir(self, resultado, argumentos):
        carpeta = valor(argumentos, "--carpeta")
        album = Path(carpeta).name
        if resultado.get("ok"):
            # Si la carpeta no se pudo borrar, el mp4 sigue ahí: no se vuelve a ofrecer.
            self.estado["subidos"].append(album)
            self.guardar()
            texto = (
                f"⏫ Subido como privado: {resultado['titulo']}\n"
                "Espero que YouTube lo procese, reviso reclamos, activo la monetización y te aviso cuándo sale."
            )
            if not resultado.get("carpeta_borrada"):
                texto += f"\n⚠️ No pude borrar la carpeta; sigue ocupando espacio: {carpeta}"
            if resultado.get("miniatura") is False:
                texto += (
                    f"\n⚠️ No pude poner la miniatura ({resultado.get('miniatura_error')}); "
                    "YouTube muestra un fotograma del intro. Ponela a mano: "
                    f"https://studio.youtube.com/video/{resultado['video_id']}/edit"
                )
            self.avisar(texto)
            self.lanzar_paso("programar", "--video-id", resultado["video_id"], "--esperar")
            return
        motivo = resultado.get("motivo")
        # Pudo morir después de subir: si el video quedó registrado, no se pregunta otra vez.
        subido = next((p for p in pa.leer_pendientes() if p.get("album") == album), None)
        if subido:
            self.estado["subidos"].append(album)
            self.guardar()
            self.avisar(f"⏫ {album} ya estaba subido ({subido['video_id']}); sigo con la fecha.")
            if not any(valor(a, "--video-id") == subido["video_id"] for a, _ in self.tareas if a[0] == "programar"):
                self.lanzar_paso("programar", "--video-id", subido["video_id"], "--esperar")
            return
        explicacion = MOTIVOS_SUBIDA.get(motivo) or resultado.get("detalle") or motivo
        if not Path(carpeta).is_dir():
            self.avisar(f"❌ No pude subir {album}: {explicacion}. La carpeta ya no existe.")
            return
        self.preguntar("subir", f"❌ No pude subir {album}: {explicacion}.\n\n¿Reintento?", carpeta=carpeta)

    def tras_eliminar(self, resultado, argumentos):
        album = Path(valor(argumentos, "--carpeta")).name
        if resultado.get("ok"):
            self.avisar(f"🗑 Borré {album}: ya estaba publicado. Sigo con el próximo.")
            return
        # Que no se vuelva a ofrecer: la carpeta sigue ahí y el usuario ya decidió.
        self.estado["rechazados"].append(album)
        self.guardar()
        self.avisar(
            f"❌ No pude borrar {album} ({resultado.get('motivo')}). "
            f"Quedó en {Path(valor(argumentos, '--carpeta')).parent.name}/."
        )

    def tras_programar(self, resultado, argumentos):
        video_id = valor(argumentos, "--video-id")
        motivo = resultado.get("motivo")
        if resultado.get("ok"):
            self.fallos_programar.pop(video_id, None)
            self.avisar(resultado["resumen"])
            monetizacion = resultado.get("monetizacion") or {}
            if monetizacion.get("estado") == "fallo":
                titulo = resultado.get("titulo") or video_id
                causa = MOTIVOS_MONETIZACION.get(monetizacion.get("motivo")) or monetizacion.get("motivo")
                self.avisar(
                    f"⚠️ No pude activar la monetización de \"{titulo}\" ({causa}).\n"
                    "No frené el video por esto; actívala a mano:\n"
                    f"https://studio.youtube.com/video/{video_id}/monetization"
                )
            self.preguntar_por_lista_negra(resultado, video_id)
            self.preguntar_por_impugnar(resultado, video_id)
        elif motivo == "reclamo_desconocido":
            titulo = resultado.get("titulo") or video_id
            self.preguntar(
                "reclamo",
                f"🤔 No pude saber si \"{titulo}\" tiene reclamo ({resultado.get('detalle')}).\n"
                f"https://studio.youtube.com/video/{video_id}/copyright\n\n"
                "¿Tiene reclamo de derechos de autor?",
                foto=resultado.get("captura_studio"),
                video_id=video_id,
            )
        elif motivo == "copia_retirada":
            self.preguntar_por_copia_retirada(resultado, video_id)
        elif motivo in ("fallo", "no_existe"):
            self.avisar(f"❌ YouTube rechazó o perdió el video {video_id} ({motivo}).")
        elif motivo == "no_es_privado":
            self.avisar(f"⚠️ {video_id} ya no es privado: no lo programo.")
        else:
            fallos = self.fallos_programar.get(video_id, 0) + 1
            self.fallos_programar[video_id] = fallos
            if fallos < INTENTOS_PROGRAMAR:
                self.avisar(f"⚠️ No pude programar {video_id} ({motivo}). Reintento en 30 min.")
                self.reintentos.append((self.reloj() + REINTENTO_S, argumentos))
            else:
                self.preguntar(
                    "reprogramar",
                    f"⚠️ No pude programar {video_id} ({motivo}) después de {fallos} intentos.\n\n"
                    "¿Reintento?",
                    video_id=video_id,
                    # El mismo comando que falló: si traía --reclamo, se respeta.
                    argumentos=list(argumentos),
                )

    def preguntar_por_lista_negra(self, resultado, video_id):
        """Un reclamante de la lista negra: avisa y ofrece borrar el video. Nunca lo borra solo."""
        reclamantes = resultado.get("reclamantes_en_lista_negra") or []
        if not reclamantes:
            return
        titulo = resultado.get("titulo") or video_id
        lista = "\n".join(f"• {reclamante}" for reclamante in reclamantes)
        self.preguntar(
            "lista_negra",
            f"🚩 \"{titulo}\" tiene reclamo de la lista negra de reclamantes:\n{lista}\n\n"
            f"https://studio.youtube.com/video/{video_id}/claims\n\n"
            "El video sigue programado. ¿Lo elimino de YouTube? No se puede deshacer.",
            foto=resultado.get("captura_studio"),
            video_id=video_id,
        )

    def preguntar_por_copia_retirada(self, resultado, video_id):
        """YouTube retiró el video por ser copia de otro: avisa y solo ofrece borrarlo."""
        titulo = resultado.get("titulo") or video_id
        datos = "".join(f"{linea}\n" for linea in resultado.get("detalle") or [])
        self.preguntar(
            "copia_retirada",
            "🚫 No se puede publicar porque es copia de otro video que YouTube retiró por derechos "
            f"de autor:\n\"{titulo}\"\n{datos}\n"
            "Ya no queda pendiente de programar, y el álbum está en el registro de publicados: no se "
            "vuelve a renderizar ni a subir.\n"
            f"https://studio.youtube.com/video/{video_id}/edit\n\n"
            "¿Lo elimino de YouTube? No se puede deshacer.",
            foto=resultado.get("captura_studio"),
            video_id=video_id,
        )

    def preguntar_por_impugnar(self, resultado, video_id):
        """Un video con reclamo y ninguno de la lista negra: ofrece impugnarlo. Nunca impugna solo."""
        if not resultado.get("con_reclamo") or resultado.get("reclamantes_en_lista_negra"):
            return
        titulo = resultado.get("titulo") or video_id
        reclamantes = resultado.get("reclamantes") or []
        quienes = ":\n" + "\n".join(f"• {reclamante}" for reclamante in reclamantes) if reclamantes else "."
        self.preguntar(
            "impugnar",
            f"⚖️ \"{titulo}\" tiene reclamo de derechos de autor{quienes}\n\n"
            f"https://studio.youtube.com/video/{video_id}/claims\n\n"
            "El video sigue programado. ¿Impugno los reclamos o elimino el video? Impugnar es una "
            "declaración legal a tu nombre: se envía con el motivo \"Licencia\", el texto de licencia "
            "y la firma ya configurados. Eliminar el video no se puede deshacer.",
            foto=resultado.get("captura_studio"),
            video_id=video_id,
        )

    def tras_impugnar(self, resultado, argumentos):
        video_id = valor(argumentos, "--video-id")
        lote = self.estado["lote_impugnar"]
        if lote and lote["videos"] and lote["videos"][0]["video_id"] == video_id:
            self.anotar_en_el_lote(resultado)
            return
        titulo = resultado.get("titulo") or video_id
        enlace = f"https://studio.youtube.com/video/{video_id}/claims"
        foto = resultado.get("captura")
        if resultado.get("motivo") == "sin_reclamos_impugnables":
            self.avisar(
                f"ℹ️ \"{titulo}\" no tiene reclamos que se puedan impugnar ahora "
                f"(ya estaban impugnados o ya no hay reclamos).\n{enlace}",
                foto=foto,
            )
            return
        if resultado.get("reclamos"):
            impugnados = resultado.get("impugnados", 0)
            texto = f"⚖️ Impugné {impugnados} de {resultado['reclamos']} reclamos de \"{titulo}\"."
        else:
            texto = f"❌ No pude impugnar \"{titulo}\"."
        if resultado.get("ok"):
            self.avisar(f"{texto}\n{enlace}", foto=foto)
            return
        fallos = resultado.get("fallos") or []
        if not fallos:  # un fallo antes de empezar trae el motivo en la raíz del resultado
            fallos = [{"motivo": resultado.get("motivo"), "detalle": resultado.get("detalle")}]
        for fallo in fallos:
            causa = MOTIVOS_IMPUGNACION.get(fallo.get("motivo")) or fallo.get("motivo")
            detalle = f" ({fallo['detalle']})" if fallo.get("detalle") else ""
            texto += f"\n• {causa}{detalle}"
        if pidio_verificacion({"fallos": fallos}):
            self.preguntar_por_ventana([f"{texto}\n{enlace}"], [{"video_id": video_id, "titulo": titulo}], foto=foto)
            return
        texto += f"\nRevisa en Studio cuáles quedaron impugnados y termina a mano lo que falte:\n{enlace}"
        if resultado.get("log"):
            texto += f"\nLog: {resultado['log']}"
        self.avisar(texto, foto=foto)

    def tras_apelar(self, resultado, argumentos):
        """Avisa qué se mandó y qué resolvió YouTube; si no hubo nada, no manda nada.

        Después lanza `seguir_impugnaciones`: así las dos lecturas diarias de Studio van seguidas.
        """
        self.lanzar_paso("seguir_impugnaciones")
        videos = resultado.get("videos") or []
        resueltos = resultado.get("resultados") or []
        fallos = resultado.get("fallos") or []

        def renglones(lista):
            return [
                f"• {v.get('titulo') or v['video_id']} ({TIPOS_APELACION[v['tipo']]})\n"
                f"  https://studio.youtube.com/video/{v['video_id']}/edit"
                for v in lista
            ]

        lineas = []
        for estado, encabezado in (
            ("rechazada", "❌ YouTube rechazó la apelación:"),
            ("aceptada", "🎉 YouTube aceptó la apelación (el video salió de la lista de Studio):"),
        ):
            hechos = [r for r in resueltos if r.get("resultado") == estado]
            if hechos:
                lineas.append(encabezado)
                lineas += renglones(hechos)
        for estado, encabezado in (
            ("enviado", "✅ Apelé en hebreo:"),
            ("no_confirmado", "⚠️ Lo envié pero Studio no mostró que llegó (revísalo):"),
            (
                "revisar_a_mano",
                "🔎 Studio solo ofrece «Revisar y tomar medidas» y el formulario no dice de qué "
                "problema es, así que no mandé nada (revísalos a mano):",
            ),
        ):
            hechos = [v for v in videos if v.get("estado") == estado]
            if hechos:
                lineas.append(encabezado)
                lineas += renglones(hechos)
        ya_apelados = sum(1 for v in videos if v.get("estado") == "ya_apelado")
        if ya_apelados:
            lineas.append(f"ℹ️ {ya_apelados} ya tenían una apelación y Studio no deja otra: no les mandé nada.")
        sin_opcion = sum(1 for v in videos if v.get("estado") == "sin_opcion")
        if sin_opcion:
            lineas.append(
                f"ℹ️ {sin_opcion} con anuncios limitados no tienen «Ver comentarios» (sin revisión manual): "
                "no les mandé nada."
            )
        for revision, encabezado in (
            (False, "❌ No pude apelar (no envié nada; lo intento de nuevo mañana):"),
            (True, "❌ No pude ver cómo va la apelación (lo miro de nuevo mañana):"),
        ):
            de_este_tipo = [f for f in fallos if bool(f.get("revision")) == revision]
            if de_este_tipo:
                lineas.append(encabezado)
                for fallo in de_este_tipo:
                    causa = MOTIVOS_APELACION.get(fallo.get("motivo")) or fallo.get("motivo")
                    lineas.append(f"• {fallo.get('titulo') or fallo.get('video_id')}: {causa}")
        motivo = resultado.get("motivo")
        if motivo:
            causa = MOTIVOS_APELACION.get(motivo) or motivo
            lineas.append(f"❌ No terminé las apelaciones de hoy: {causa}. Sigo mañana.")
        if not lineas:
            return
        if motivo or fallos:
            lineas.append(f"Log: {resultado.get('log')}")
        self.avisar_con_reintento("\n".join(["📨 Apelaciones de hoy"] + lineas))

    def tras_seguir_impugnaciones(self, resultado, argumentos):
        """Avisa qué reclamos impugnados cambiaron en Studio; si nada cambió, no manda nada.

        Un video que quedó sin reclamos se vuelve a programar como uno sin reclamo; por uno
        cuya impugnación terminó con algún reclamo pregunta si se borra.
        """
        cambios = resultado.get("cambios") or []
        fallos = resultado.get("fallos") or []
        lineas = []
        por_video = {}
        for cambio in cambios:
            por_video.setdefault(cambio["video_id"], []).append(cambio)
        for video_id, del_video in por_video.items():
            lineas.append(f"\"{del_video[0].get('titulo') or video_id}\"")
            for cambio in del_video:
                if cambio["ahora"] is None:
                    lineas.append(f"• {cambio['reclamo']}: ya no está en la lista (el reclamo se retiró)")
                else:
                    lineas.append(f"• {cambio['reclamo']}: {cambio['ahora']}\n  (antes: {cambio['antes']})")
            lineas.append(f"https://studio.youtube.com/video/{video_id}/claims")
        if lineas:
            lineas.insert(0, "⚖️ Studio cambió el estado de reclamos que impugné:")
        if fallos:
            lineas.append("❌ No pude leer los reclamos (no toqué nada; lo miro de nuevo mañana):")
            for fallo in fallos:
                causa = MOTIVOS_IMPUGNACION.get(fallo.get("motivo")) or fallo.get("motivo")
                lineas.append(
                    f"• {fallo.get('titulo') or fallo['video_id']}: {causa}\n"
                    f"  https://studio.youtube.com/video/{fallo['video_id']}/claims"
                )
        motivo = resultado.get("motivo")
        if motivo:
            causa = MOTIVOS_IMPUGNACION.get(motivo) or motivo
            lineas.append(f"❌ No terminé de mirar las impugnaciones: {causa}. Sigo mañana.")
        if lineas:
            if motivo or fallos:
                lineas.append(f"Log: {resultado.get('log')}")
            self.avisar_con_reintento("\n".join(lineas))
        for liberado in resultado.get("liberados") or []:
            # Studio ya lo leyó dos veces sin reclamos: sin la espera ni otra revisión.
            self.lanzar_paso("programar", "--video-id", liberado["video_id"], "--reclamo", "no")
        for rechazado in resultado.get("rechazados") or []:
            self.preguntar_por_impugnacion_rechazada(rechazado)

    def tras_mapear_canal(self, resultado, argumentos):
        """Un solo mensaje con lo que se liberó, lo que falta impugnar (con «Impugnar todos») y
        lo que no se pudo leer. Los liberados se programan como un video sin reclamo; los de la
        lista negra preguntan aparte, con su «Eliminar video». Nada se impugna sin el botón."""
        listas = {
            clave: resultado.get(clave) or []
            for clave in ("liberados", "sin_impugnar", "lista_negra", "en_proceso", "fallos", "sin_revisar")
        }
        if "aparcados" not in resultado:
            # Falló o se cortó antes de saber qué videos mirar: no hay nada que contar.
            self.avisar(
                f"❌ No pude revisar el canal: {causa_legible(resultado.get('motivo'))}.\n"
                f"Log: {resultado.get('log')}"
            )
            return
        revisados = sum(len(listas[clave]) for clave in ("liberados", "sin_impugnar", "lista_negra", "en_proceso", "fallos"))
        lineas = [f"🗺 Videos aparcados por un reclamo: {resultado['aparcados']}. Revisados: {revisados}."]
        if listas["liberados"]:
            lineas.append(f"✅ Ya sin reclamos (los publico o programo como uno sin reclamo): {len(listas['liberados'])}")
            lineas += [f"• {video.get('titulo') or video['video_id']}" for video in listas["liberados"]]
        if listas["lista_negra"]:
            lineas.append(f"🚩 Con un reclamante de la lista negra (te pregunto aparte): {len(listas['lista_negra'])}")
        if listas["en_proceso"]:
            lineas.append(f"⏳ Con la impugnación en revisión: {len(listas['en_proceso'])}")
        if listas["fallos"]:
            lineas.append("❌ No pude leer estos (no toqué nada):")
            for fallo in listas["fallos"]:
                lineas.append(
                    f"• {fallo.get('titulo') or fallo['video_id']}: {causa_legible(fallo.get('motivo'))}\n"
                    f"  https://studio.youtube.com/video/{fallo['video_id']}/claims"
                )
        if listas["sin_revisar"]:
            lineas.append(f"⏸ Sin revisar (/mapear_canal sigue con ellos): {len(listas['sin_revisar'])}")
        if resultado.get("motivo"):
            lineas.append(f"❌ No terminé de revisar: {causa_legible(resultado['motivo'])}.\nLog: {resultado.get('log')}")
        sin_impugnar = listas["sin_impugnar"]
        if sin_impugnar:
            lineas.append(f"⚖️ Con reclamos sin impugnar: {len(sin_impugnar)}")
            for video in sin_impugnar:
                lineas.append(
                    f"• \"{video.get('titulo') or video['video_id']}\": {', '.join(video['reclamantes'])}\n"
                    f"  https://studio.youtube.com/video/{video['video_id']}/claims"
                )
            lineas.append(
                "¿Los impugno? Van de a uno, con el motivo \"Licencia\", el texto de licencia y la "
                "firma ya configurados: es una declaración legal a tu nombre por cada video."
            )
            *previas, texto = partir_en_mensajes(lineas)
            self.preguntar(
                "impugnar_todos",
                texto,
                previas=[{"texto": previa, "foto": None, "enviada": False} for previa in previas],
                videos=[{"video_id": v["video_id"], "titulo": v.get("titulo")} for v in sin_impugnar],
                n=len(sin_impugnar),
            )
        else:
            for parte in partir_en_mensajes(lineas):
                self.avisar(parte)
        programando = {valor(a, "--video-id") for a, _ in self.tareas if a[0] == "programar"}
        for liberado in listas["liberados"]:
            # Studio ya lo leyó dos veces sin reclamos: sin la espera ni otra revisión.
            if liberado["video_id"] not in programando:
                self.lanzar_paso("programar", "--video-id", liberado["video_id"], "--reclamo", "no")
        ya_preguntados = {
            p.get("video_id") for p in self.estado["preguntas"].values() if p["tipo"] == "lista_negra"
        }
        for video in listas["lista_negra"]:
            if video["video_id"] not in ya_preguntados:
                self.preguntar_por_lista_negra({**video, "captura_studio": video.get("captura")}, video["video_id"])

    def impugnar_siguiente_del_lote(self):
        lote = self.estado["lote_impugnar"]
        if lote and lote["videos"]:
            self.lanzar_paso("impugnar", "--video-id", lote["videos"][0]["video_id"])

    def anotar_en_el_lote(self, resultado):
        """Lo que salió con el video en curso; sigue con el próximo o, al final, avisa una vez."""
        lote = self.estado["lote_impugnar"]
        video = lote["videos"].pop(0)
        fallos = resultado.get("fallos") or []
        if not resultado.get("ok") and not fallos and resultado.get("motivo") != "sin_reclamos_impugnables":
            fallos = [{"motivo": resultado.get("motivo"), "detalle": resultado.get("detalle")}]
        lote["hechos"].append({
            **video, "reclamos": resultado.get("reclamos"), "impugnados": resultado.get("impugnados", 0),
            "motivo": resultado.get("motivo"), "fallos": fallos,
        })
        frena = next((f["motivo"] for f in fallos if f.get("motivo") in FRENAN_EL_LOTE), None)
        if lote["videos"] and not frena:
            self.guardar()
            self.impugnar_siguiente_del_lote()
            return
        self.estado["lote_impugnar"] = None
        self.guardar()
        lineas = [f"⚖️ Terminé el «Impugnar todos». Videos: {len(lote['hechos'])}"]
        lineas += lineas_de_impugnacion(lote["hechos"])
        if frena == "verificacion_de_identidad":
            if lote["videos"]:
                lineas.append(f"⛔ Frené el resto: {causa_legible(frena)}. Quedan {len(lote['videos'])} sin impugnar:")
                lineas += [f"• {v.get('titulo') or v['video_id']}" for v in lote["videos"]]
            ventana = [
                {"video_id": h["video_id"], "titulo": h.get("titulo")} for h in lote["hechos"] if pidio_verificacion(h)
            ]
            self.preguntar_por_ventana(lineas, ventana + lote["videos"])
            return
        if lote["videos"]:
            lineas.append(
                f"⛔ Frené el resto: {causa_legible(frena)}. Quedan {len(lote['videos'])} sin impugnar; "
                "resolvelo en Studio y corré /mapear_canal de nuevo:"
            )
            lineas += [f"• {v.get('titulo') or v['video_id']}" for v in lote["videos"]]
        for parte in partir_en_mensajes(lineas):
            self.avisar(parte)

    def preguntar_por_ventana(self, lineas, videos, foto=None):
        """Studio pidió "Verifica que eres tú": ofrece impugnar esos videos en una ventana visible,
        con una sola verificación. Nada se impugna sin apretar el botón."""
        *previas, texto = partir_en_mensajes(lineas + [INSTRUCCIONES_VENTANA])
        self.preguntar(
            "impugnar_ventana",
            texto,
            foto=foto,
            previas=[{"texto": previa, "foto": None, "enviada": False} for previa in previas],
            videos=[{"video_id": v["video_id"], "titulo": v.get("titulo")} for v in videos],
        )

    def tras_impugnar_con_ventana(self, resultado, argumentos):
        """Un solo informe de lo que salió en la ventana. Si la verificación no se pasó a tiempo,
        vuelve a ofrecer la ventana para lo que quedó."""
        videos = resultado.get("videos") or []
        motivo = resultado.get("motivo")
        sin_revisar = resultado.get("sin_revisar")
        if sin_revisar is None:
            # Terminó sin decir qué hizo: pudo haber impugnado algo, así que no se ofrece de nuevo.
            lineas = ["❌ La ventana terminó sin decir qué impugnó (se cortó o falló). Revisa en Studio:"]
            lineas += [f"• https://studio.youtube.com/video/{v}/claims" for v in argumentos[2::2]]
            if resultado.get("log"):
                lineas.append(f"Log: {resultado['log']}")
            for parte in partir_en_mensajes(lineas):
                self.avisar(parte)
            return
        if videos:
            lineas = [f"🖥 Terminé de impugnar con ventana. Videos: {len(videos)}"]
            lineas += lineas_de_impugnacion(videos)
        else:
            lineas = [f"❌ No pude impugnar con ventana: {causa_legible(motivo)}."]
        if sin_revisar:
            if videos:
                lineas.append(f"⛔ Frené el resto: {causa_legible(motivo)}. Quedan {len(sin_revisar)} sin impugnar:")
            lineas += [f"• {v.get('titulo') or v['video_id']}" for v in sin_revisar]
        if motivo == "verificacion_de_identidad":
            ventana = [{"video_id": v["video_id"], "titulo": v.get("titulo")} for v in videos if pidio_verificacion(v)]
            ventana += sin_revisar
            self.preguntar_por_ventana(lineas, ventana)
            return
        if not resultado.get("ok") and resultado.get("log"):
            lineas.append(f"Log: {resultado['log']}")
        for parte in partir_en_mensajes(lineas):
            self.avisar(parte)

    def preguntar_por_impugnacion_rechazada(self, rechazado):
        """La impugnación terminó y el video sigue con reclamo: ofrece borrarlo. Nunca lo borra solo."""
        video_id = rechazado["video_id"]
        titulo = rechazado.get("titulo") or video_id
        reclamos = "".join(f"• {r['reclamo']}: {r['estado']}\n" for r in rechazado.get("reclamos") or [])
        self.preguntar(
            "impugnacion_rechazada",
            f"❌ La impugnación de \"{titulo}\" terminó y el video sigue con reclamo:\n{reclamos}"
            f"https://studio.youtube.com/video/{video_id}/claims\n\n"
            "Sigue programado a dos años y sin ingresos. ¿Lo elimino de YouTube? No se puede deshacer.",
            foto=rechazado.get("captura"),
            video_id=video_id,
        )

    def tras_eliminar_video(self, resultado, argumentos):
        video_id = valor(argumentos, "--video-id")
        if resultado.get("ok"):
            self.avisar(f"🗑 Borré el video \"{resultado.get('titulo') or video_id}\" de YouTube.")
            return
        if resultado.get("motivo") == "no_existe":
            # Ya lo habían borrado (a mano, o un intento cuya respuesta se perdió).
            self.avisar(f"ℹ️ El video {video_id} ya no está en YouTube: ya estaba borrado.")
            return
        self.avisar(
            f"❌ No pude borrar el video {video_id} ({resultado.get('motivo')}). Sigue en YouTube; "
            f"se puede borrar a mano en Studio:\nhttps://studio.youtube.com/video/{video_id}/edit"
        )

    # -- Telegram -----------------------------------------------------------

    def avisar(self, texto, foto=None):
        try:
            self.telegram.mandar(texto, foto=foto)
        except ERRORES_TELEGRAM as exc:
            log.warning("No se pudo mandar el aviso (%s): %s", exc, texto)

    def avisar_con_reintento(self, texto):
        """Un aviso cuyo resultado ya quedó anotado en disco: se guarda antes de mandarlo y solo
        sale de la lista cuando Telegram lo recibe, así que un rechazo no lo pierde."""
        self.estado["avisos_pendientes"].append({"texto": texto, "intentos": 0})
        self.guardar()
        self.enviar_avisos_pendientes()

    def enviar_avisos_pendientes(self):
        """Manda, en orden, los avisos que Telegram todavía no recibió. Con Telegram sin responder
        no cuenta el intento (el aviso espera a que vuelva); con Telegram respondiendo, un aviso
        que lo rechaza INTENTOS_AVISO veces se descarta."""
        pendientes = self.estado["avisos_pendientes"]
        while pendientes:
            aviso = pendientes[0]
            try:
                self.telegram.mandar(aviso["texto"])
            except ERRORES_TELEGRAM as exc:
                if self.fallos_telegram:
                    return
                aviso["intentos"] += 1
                if aviso["intentos"] < INTENTOS_AVISO:
                    log.warning("Telegram rechazó un aviso (intento %d); reintento: %s", aviso["intentos"], exc)
                    self.guardar()
                    return
                log.warning("Descarto un aviso que Telegram rechazó %d veces (%s): %s", INTENTOS_AVISO, exc, aviso["texto"])
            pendientes.pop(0)
            self.guardar()

    def avisar_una_vez(self, clave, texto):
        if clave not in self.avisos:
            self.avisos.add(clave)
            self.avisar(texto)

    def preguntar(self, tipo, texto, foto=None, **datos):
        qid = uuid.uuid4().hex[:8]
        self.estado["preguntas"][qid] = {
            "tipo": tipo, "texto": texto, "foto": foto, "message_id": None, **datos
        }
        self.guardar()
        self.enviar_preguntas()

    def enviar_preguntas(self):
        """Manda las preguntas que todavía no llegaron (por ejemplo, sin red)."""
        for qid, pregunta in self.estado["preguntas"].items():
            if pregunta["message_id"] is not None:
                continue
            # {n}: lo que cuenta la pregunta (los videos de «Impugnar todos») o los álbumes de la lista.
            n = str(pregunta.get("n", len(self.estado["lista"])))
            botones = [
                [(etiqueta.replace("{n}", n), f"{accion}:{qid}") for etiqueta, accion in fila]
                for fila in BOTONES[pregunta["tipo"]]
            ]
            try:
                # Las fotos previas (la búsqueda en YouTube) van antes, sin botones.
                for previa in pregunta.get("previas", []):
                    if not previa.get("enviada"):
                        self.telegram.mandar(previa["texto"], foto=previa["foto"])
                        previa["enviada"] = True
                        self.guardar()
                pregunta["message_id"] = self.telegram.mandar(
                    pregunta["texto"], botones=botones, foto=pregunta["foto"]
                )
            except ERRORES_TELEGRAM as exc:
                # Con Telegram caído, atender_telegram ya avisó: no se repite cada ciclo.
                if not self.fallos_telegram:
                    log.warning("No se pudo mandar la pregunta; reintento en el próximo ciclo: %s", exc)
                return
            self.guardar()

    def atender_telegram(self):
        try:
            actualizaciones = self.telegram.actualizaciones(self.estado["offset"], ESPERA_TELEGRAM_S)
        except ERRORES_TELEGRAM as exc:
            self.fallo_telegram(exc)
            # 5 s, 10 s, 20 s... hasta PAUSA_TELEGRAM_MAX_S; el exponente se acota para no crecer sin fin.
            pausa = self.pausa_error_s * 2 ** min(self.fallos_telegram - 1, 10)
            time.sleep(min(pausa, PAUSA_TELEGRAM_MAX_S))
            return
        self.telegram_responde()
        for actualizacion in actualizaciones:
            # Se guarda antes de atender: un botón nunca se procesa dos veces.
            self.estado["offset"] = actualizacion["update_id"] + 1
            self.guardar()
            if "callback_query" in actualizacion:
                self.atender_boton(actualizacion["callback_query"])
            elif "message" in actualizacion:
                self.atender_mensaje(actualizacion["message"])

    def fallo_telegram(self, exc):
        """Avisa solo el primer fallo de una caída; los siguientes no llenan el log."""
        if not self.fallos_telegram:
            self.telegram_caido_desde = self.reloj()
            log.warning("Telegram no responde; sigo intentando con esperas cada vez más largas: %s", exc)
        self.fallos_telegram += 1

    def telegram_responde(self):
        if not self.fallos_telegram:
            return
        log.info(
            "Telegram responde de nuevo (%d fallos seguidos, %d s sin respuesta)",
            self.fallos_telegram,
            self.reloj() - self.telegram_caido_desde,
        )
        self.fallos_telegram = 0
        self.telegram_caido_desde = None

    def atender_boton(self, consulta):
        if str(consulta.get("from", {}).get("id")) != self.telegram.chat_id:
            log.warning("Botón de un usuario no autorizado: %s", consulta.get("from"))
            self.responder_boton(consulta, "No autorizado.")
            return
        accion, _, qid = (consulta.get("data") or "").partition(":")
        pregunta = self.estado["preguntas"].get(qid)
        if not pregunta or accion not in acciones(pregunta["tipo"]):
            self.responder_boton(consulta, "Esa pregunta ya no está vigente.")
            return
        del self.estado["preguntas"][qid]
        self.guardar()
        self.responder_boton(consulta, AVISO_BOTON[accion])
        try:
            self.telegram.quitar_botones(pregunta["message_id"])
        except ERRORES_TELEGRAM as exc:
            log.warning("No se pudieron quitar los botones: %s", exc)
        getattr(self, f"respuesta_{pregunta['tipo']}")(accion, pregunta)

    def atender_mensaje(self, mensaje):
        """Lo que se escribe en el chat: /aleatorio, /crear_lista y /mapear_canal; lo demás se ignora."""
        if str(mensaje.get("from", {}).get("id")) != self.telegram.chat_id:
            log.warning("Mensaje de un usuario no autorizado: %s", mensaje.get("from"))
            return
        palabras = (mensaje.get("text") or "").split()
        # En un grupo, Telegram manda el comando como /aleatorio@NombreDelBot.
        comando = palabras[0].split("@")[0] if palabras else ""
        if comando == "/aleatorio":
            self.estado["aleatorio"] = not self.estado["aleatorio"]
            self.guardar()
            self.avisar("🔀 Aleatorio activado" if self.estado["aleatorio"] else "Aleatorio desactivado")
        # El menú de Telegram no acepta guiones: /crear_lista. Escrito con guion también vale.
        elif comando in ("/crear_lista", "/crear-lista"):
            self.crear_lista()
        elif comando in ("/mapear_canal", "/mapear-canal"):
            self.mapear_canal()

    def mapear_canal(self):
        if any(argumentos[0] == "mapear_canal" for argumentos, _ in self.tareas):
            self.avisar("🗺 Ya estoy revisando el canal; aviso cuando termine.")
            return
        self.avisar("🗺 Reviso en Studio los videos aparcados por un reclamo. Solo leo; aviso cuando termine.")
        self.lanzar_paso("mapear_canal")

    def crear_lista(self):
        if self.estado["modo_lista"]:
            # No es un interruptor: escribirlo otra vez no tira lo ya elegido.
            self.avisar(
                f"📝 Ya estás armando una lista: {cantidad_de_albumes(len(self.estado['lista']))}. "
                "Tocá ▶️ en la pregunta para renderizarla."
            )
            return
        self.estado["modo_lista"] = True
        self.guardar()
        self.avisar(
            "📝 Modo lista: lo que apruebes con ➕ queda guardado y no se renderiza. "
            "Cuando quieras, tocá ▶️ y renderizo y subo toda la lista sin preguntarte."
        )
        # La pregunta que ya estaba abierta vuelve con los botones de la lista.
        for qid, pregunta in list(self.estado["preguntas"].items()):
            if pregunta["tipo"] != "renderizar":
                continue
            del self.estado["preguntas"][qid]
            self.guardar()
            if pregunta["message_id"] is not None:
                try:
                    self.telegram.quitar_botones(pregunta["message_id"])
                except ERRORES_TELEGRAM as exc:
                    log.warning("No se pudieron quitar los botones: %s", exc)
            self.preguntar_por_album(
                pregunta.get("album") or Path(pregunta["carpeta"]).name,
                pregunta["carpeta"],
                pregunta["foto"],
                pregunta.get("previas", []),
                pregunta.get("enlace", ""),
            )

    def preguntar_por_album(self, album, carpeta, foto, previas, enlace):
        """La pregunta de antes del render; en modo lista, con los botones de la lista."""
        if self.estado["modo_lista"]:
            tipo, cierre = "lista", "Si no, lo sumo a la lista."
        else:
            tipo, cierre = "renderizar", "Si no, lo renderizo y lo subo."
        self.preguntar(
            tipo,
            f"{album}{enlace}\n\n¿Ya está publicado? {cierre}",
            foto=foto,
            carpeta=carpeta,
            previas=previas,
            album=album,
            enlace=enlace,
        )

    def preguntar_despues_de_lista(self):
        """Cuando la lista terminó, pregunta por el álbum que estaba en pantalla al tocar ▶️."""
        pendiente = self.estado["despues_de_lista"]
        if not pendiente:
            return False
        self.estado["despues_de_lista"] = None
        self.guardar()
        if not Path(pendiente["carpeta"]).is_dir():
            return False
        self.preguntar_por_album(
            pendiente["album"],
            pendiente["carpeta"],
            pendiente["foto"],
            pendiente["previas"],
            pendiente["enlace"],
        )
        return True

    def responder_boton(self, consulta, texto):
        try:
            self.telegram.responder_boton(consulta.get("id"), texto)
        except ERRORES_TELEGRAM as exc:
            log.warning("No se pudo responder el botón: %s", exc)

    def respuesta_renderizar(self, accion, pregunta):
        if accion == "eliminar":
            self.lanzar_paso("eliminar", "--carpeta", pregunta["carpeta"])
            return
        nombre = Path(pregunta["carpeta"]).name
        self.estado["aceptados"].append(nombre)
        self.guardar()
        self.lanzar_paso("renderizar", "--carpeta", nombre)

    def respuesta_lista(self, accion, pregunta):
        if accion == "eliminar":
            self.lanzar_paso("eliminar", "--carpeta", pregunta["carpeta"])
        elif accion == "a_la_lista":
            self.estado["lista"].append(Path(pregunta["carpeta"]).name)
            self.guardar()
        else:
            self.renderizar_lista(pregunta)

    def respuesta_fin_lista(self, accion, pregunta):
        self.renderizar_lista(None)

    def renderizar_lista(self, en_pantalla):
        """▶️: la lista pasa a aceptados, que se renderizan y se suben sin preguntar."""
        lista = self.estado["lista"]
        self.estado["aceptados"].extend(n for n in lista if n not in self.estado["aceptados"])
        self.estado["lista"] = []
        self.estado["modo_lista"] = False
        if en_pantalla:
            # Si la lista tarda, la búsqueda vuelve a llegar con la pregunta.
            previas = [{**p, "enviada": False} for p in en_pantalla.get("previas", [])] if lista else []
            self.estado["despues_de_lista"] = {
                "album": en_pantalla.get("album") or Path(en_pantalla["carpeta"]).name,
                "carpeta": en_pantalla["carpeta"],
                "foto": en_pantalla["foto"],
                "previas": previas,
                "enlace": en_pantalla.get("enlace", ""),
            }
        self.guardar()
        if lista:
            self.avisar(
                f"▶️ Renderizo y subo {cantidad_de_albumes(len(lista))} de la lista. "
                "Solo te pregunto si un chequeo falla."
            )
        else:
            self.avisar("La lista estaba vacía; vuelvo a preguntar de a uno.")

    def respuesta_subir(self, accion, pregunta):
        if accion in ("subir", "eliminar"):
            self.lanzar_paso(accion, "--carpeta", pregunta["carpeta"])
            return
        album = Path(pregunta["carpeta"]).name
        self.estado["rechazados"].append(album)
        self.guardar()
        self.avisar(f"👌 No subo {album}. Quedó en Renderizando/; sigo con el próximo.")

    def respuesta_reclamo(self, accion, pregunta):
        self.lanzar_paso("programar", "--video-id", pregunta["video_id"], "--reclamo", accion)

    def respuesta_lista_negra(self, accion, pregunta):
        if accion == "eliminar_video":
            self.lanzar_paso("eliminar_video", "--video-id", pregunta["video_id"])
        # Conservar: la pregunta ya se cerró y el video queda como está.

    def respuesta_impugnar(self, accion, pregunta):
        if accion == "impugnar":
            self.lanzar_paso("impugnar", "--video-id", pregunta["video_id"])
        elif accion == "eliminar_video":
            self.lanzar_paso("eliminar_video", "--video-id", pregunta["video_id"])

    def respuesta_impugnar_todos(self, accion, pregunta):
        """Suma al lote los videos que no tiene (ni por hacer ni hechos); si no había uno en
        curso, empieza por el primero."""
        lote = self.estado["lote_impugnar"]
        if lote is None:
            lote = self.estado["lote_impugnar"] = {"videos": [], "hechos": []}
        en_el_lote = {video["video_id"] for video in lote["videos"] + lote["hechos"]}
        nuevos = [video for video in pregunta["videos"] if video["video_id"] not in en_el_lote]
        en_curso = bool(lote["videos"])
        lote["videos"] += nuevos
        self.guardar()
        if not en_curso:
            self.impugnar_siguiente_del_lote()

    def respuesta_impugnar_ventana(self, accion, pregunta):
        argumentos = ["impugnar_con_ventana"]
        for video in pregunta["videos"]:
            argumentos += ["--video-id", video["video_id"]]
        self.lanzar_paso(*argumentos)

    def respuesta_copia_retirada(self, accion, pregunta):
        # La API no lista un video retirado por copia: se borra sin la comprobación de siempre.
        self.lanzar_paso("eliminar_video", "--video-id", pregunta["video_id"], "--retirado")

    def respuesta_impugnacion_rechazada(self, accion, pregunta):
        if accion == "eliminar_video":
            self.lanzar_paso("eliminar_video", "--video-id", pregunta["video_id"])
        # Conservar: la pregunta ya se cerró y el video queda como está.

    def respuesta_reprogramar(self, accion, pregunta):
        self.fallos_programar.pop(pregunta["video_id"], None)
        self.lanzar_paso(*pregunta["argumentos"])

    def respuesta_reintentar(self, accion, pregunta):
        """Sacar la pregunta alcanza: el próximo ciclo vuelve a renderizar."""


def main():
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pa.uploader_real().cargar_env()
    token = os.environ.get("PUBLICAR_TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("PUBLICAR_TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.error("Faltan PUBLICAR_TELEGRAM_TOKEN o PUBLICAR_TELEGRAM_CHAT_ID en .env")
        return SIN_CONFIGURACION
    telegram = Telegram(token, chat_id)
    try:
        telegram.registrar_comandos()
    except ERRORES_TELEGRAM as exc:
        # Sin el menú, /aleatorio igual funciona escrito a mano.
        log.warning("No se pudo registrar el menú de comandos: %s", exc)
    servicio = Servicio(telegram, Proceso)
    servicio.iniciar()
    log.info("Servicio en marcha")
    while True:
        servicio.paso()


if __name__ == "__main__":
    sys.exit(main())
