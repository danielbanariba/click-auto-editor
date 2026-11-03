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
LIMITE_LEYENDA = 1024
LIMITE_TEXTO = 4096
SIN_CONFIGURACION = 78  # EX_CONFIG: el unit de systemd no reinicia con este código

# Filas de botones. Eliminar va en su propia fila, lejos de Subir, para no apretarlo sin querer.
BOTONES = {
    "subir": [
        [("✅ Subir", "subir"), ("❌ Rechazar", "rechazar")],
        [("🗑 Eliminar (ya está publicado)", "eliminar")],
    ],
    "reclamo": [[("Sí, tiene reclamo", "si"), ("No tiene", "no")]],
    "reintentar": [[("🔁 Reintentar", "reintentar")]],
    "reprogramar": [[("🔁 Reintentar", "reintentar")]],
    # Borrar el video no se deshace: va en su propia fila, debajo de Conservar.
    "lista_negra": [[("Conservar", "conservar")], [("🗑️ Eliminar video", "eliminar_video")]],
    # Impugnar es una declaración legal a nombre del usuario: cada botón en su fila.
    "impugnar": [[("⚖️ Impugnar", "impugnar")], [("No impugnar", "no_impugnar")]],
}
AVISO_BOTON = {
    "subir": "Subiendo…",
    "rechazar": "No lo subo",
    "eliminar": "Borrando…",
    "si": "Programando…",
    "no": "Programando…",
    "reintentar": "Reintentando…",
    "conservar": "Lo conservo",
    "eliminar_video": "Borrando el video…",
    "impugnar": "Impugnando…",
    "no_impugnar": "No impugno",
}
# Mientras haya una de estas preguntas abiertas no se renderiza otro álbum.
# "reprogramar" no frena: el video ya está subido y no depende del render.
PREGUNTAS_QUE_FRENAN = ("subir", "reintentar")
PASOS_DEL_ALBUM = ("renderizar", "verificar", "subir", "eliminar")
# Tiempo máximo de cada paso. Los cortables son idempotentes; los demás mueven o
# borran carpetas y cortarlos a medias dejaría un álbum a medio mover: solo se avisa.
LIMITE_PASO_S = 30 * 60  # verificar, eliminar y cualquier paso que no esté abajo
LIMITES_PASO_S = {
    "renderizar": pa.ESPERA_RENDER_MAX_S,
    "subir": 4 * 3600,
    "programar": 8 * 3600,  # espera el procesado de YouTube y el margen del reclamo
}
PASOS_CORTABLES = ("verificar", "programar")
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
    "error_navegador": "falló el navegador",
}

log = logging.getLogger("servicio_publicar")


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
            allowed_updates=["callback_query"],
        )

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


def carpetas_de_origen():
    """Las carpetas de donde toma álbumes el servicio: «a/, b/ ni c/»."""
    nombres = [f"{origen.name}/" for origen in pa.DIRS_ORIGEN_RENDER]
    if len(nombres) == 1:
        return nombres[0]
    return f"{', '.join(nombres[:-1])} ni {nombres[-1]}"


class Servicio:
    def __init__(self, telegram, lanzar, estado_json=ESTADO_JSON, reloj=time.monotonic):
        self.telegram = telegram
        self.lanzar = lanzar
        self.ruta_estado = Path(estado_json)
        self.reloj = reloj
        # rechazados y subidos: álbumes que siguen en Renderizando/ pero no se vuelven a ofrecer.
        self.estado = {"offset": 0, "preguntas": {}, "rechazados": [], "subidos": []}
        self.tareas = []
        self.inicios = {}  # proceso -> momento en que se lanzó
        self.largos_avisados = set()  # procesos que pasaron su límite y ya se avisó
        self.reintentos = []
        self.fallos_programar = {}  # video_id -> fallos seguidos que llegaron al reintento genérico
        self.album_pausado_hasta = 0.0
        self.avisos = set()
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
        self.enviar_preguntas()

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
        self.enviar_preguntas()
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

    def avanzar_album(self):
        if any(argumentos[0] in PASOS_DEL_ALBUM for argumentos, _ in self.tareas):
            return
        if any(p["tipo"] in PREGUNTAS_QUE_FRENAN for p in self.estado["preguntas"].values()):
            return
        if self.reloj() < self.album_pausado_hasta:
            return
        # Un render que lanzó otro (a mano o una sesión vieja) puede estar
        # moviendo el mp4 a Renderizando/: se espera a que termine.
        if pa.render_en_curso():
            return
        descartados = set(self.estado["rechazados"]) | set(self.estado["subidos"])
        listos = [c for c in pa.albumes_por_aprobar() if Path(c).name not in descartados]
        if listos:
            self.lanzar_paso("verificar", "--carpeta", listos[0])
        elif pa.siguiente_album() is not None:
            self.avisos.discard("sin_albumes")
            self.lanzar_paso("renderizar")
        else:
            self.avisar_una_vez(
                "sin_albumes",
                f"📭 No quedan álbumes en {carpetas_de_origen()}. Sigo solo cuando aparezca uno.",
            )

    # -- resultados de cada paso -------------------------------------------

    def tras_renderizar(self, resultado, argumentos):
        motivo = resultado.get("motivo")
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
        if resultado.get("motivo") == "sin_video":
            self.estado["rechazados"].append(album)
            self.guardar()
            self.avisar(f"⚠️ {album}: el render no dejó el mp4. Lo dejo de lado.")
            return
        if resultado.get("motivo") == "tiempo_agotado":
            self.avisar(
                f"⚠️ No pude verificar {album}: el paso pasó los "
                f"{duracion_legible(limite_del_paso('verificar'))} y lo corté. "
                f"Reintento en 30 min.\nLog: {resultado.get('log')}"
            )
            self.album_pausado_hasta = self.reloj() + REINTENTO_S
            return
        texto = resultado.get("resumen") or album
        if not resultado.get("ok"):
            texto += "\n\n⚠️ Hay tests fallidos."
        previas = []
        busqueda = resultado.get("busqueda")
        if resultado.get("captura_busqueda"):
            previas.append(
                {"texto": f"🔎 Búsqueda en YouTube: {busqueda}", "foto": resultado["captura_busqueda"]}
            )
        elif busqueda:
            texto += f"\n\n🔎 https://www.youtube.com/results?search_query={quote(busqueda)}"
        self.preguntar(
            "subir",
            f"{texto}\n\n¿Lo subo?",
            foto=resultado.get("captura"),
            carpeta=carpeta,
            previas=previas,
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
        self.avisar(f"❌ No pude borrar {album} ({resultado.get('motivo')}). Quedó en Renderizando/.")

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
            "El video sigue programado. ¿Impugno los reclamos? Es una declaración legal a tu nombre: "
            "se envía con el motivo \"Licencia\", el texto de licencia y la firma ya configurados.",
            foto=resultado.get("captura_studio"),
            video_id=video_id,
        )

    def tras_impugnar(self, resultado, argumentos):
        video_id = valor(argumentos, "--video-id")
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
        texto += f"\nRevisa en Studio cuáles quedaron impugnados y termina a mano lo que falte:\n{enlace}"
        if resultado.get("log"):
            texto += f"\nLog: {resultado['log']}"
        self.avisar(texto, foto=foto)

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
            botones = [
                [(etiqueta, f"{accion}:{qid}") for etiqueta, accion in fila]
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

    def responder_boton(self, consulta, texto):
        try:
            self.telegram.responder_boton(consulta.get("id"), texto)
        except ERRORES_TELEGRAM as exc:
            log.warning("No se pudo responder el botón: %s", exc)

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
        # No impugnar: la pregunta ya se cerró y el video queda como está.

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
    servicio = Servicio(Telegram(token, chat_id), Proceso)
    servicio.iniciar()
    log.info("Servicio en marcha")
    while True:
        servicio.paso()


if __name__ == "__main__":
    sys.exit(main())
