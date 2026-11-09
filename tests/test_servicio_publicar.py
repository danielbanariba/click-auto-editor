"""Tests del servicio que publica álbumes con aprobación por Telegram.

Telegram y los pasos de pipeline_agente.py son falsos; las carpetas y el
estado del servicio son reales (en un directorio temporal).
"""
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

import pipeline_agente as pa
import servicio_publicar as sp

USUARIO = 42


class TelegramFalso:
    chat_id = str(USUARIO)

    def __init__(self):
        self.mensajes = []
        self.sin_botones = []
        self.respuestas = []
        self.updates = []
        self.siguiente_update = 1
        self.falla = None

    def actualizaciones(self, offset, espera):
        if self.falla:
            raise self.falla
        pendientes = [u for u in self.updates if u["update_id"] >= offset]
        self.updates = []
        return pendientes

    def mandar(self, texto, botones=None, foto=None):
        filas = botones or []
        self.mensajes.append(
            {"texto": texto, "botones": [b for fila in filas for b in fila], "foto": foto}
        )
        return len(self.mensajes)

    def quitar_botones(self, message_id):
        self.sin_botones.append(message_id)

    def responder_boton(self, consulta_id, texto):
        self.respuestas.append(texto)

    def apretar(self, mensaje, etiqueta, de=USUARIO):
        datos = dict(mensaje["botones"])[etiqueta]
        self.updates.append(
            {
                "update_id": self.siguiente_update,
                "callback_query": {"id": "c", "from": {"id": de}, "data": datos},
            }
        )
        self.siguiente_update += 1


def escribir_en(telegram, texto, de=USUARIO):
    telegram.updates.append(
        {
            "update_id": telegram.siguiente_update,
            "message": {"message_id": 900, "from": {"id": de}, "chat": {"id": de}, "text": texto},
        }
    )
    telegram.siguiente_update += 1


class ProcesoFalso:
    def __init__(self, argumentos):
        self.argumentos = argumentos
        self.salida = None

    def resultado(self):
        return self.salida


class LanzadorFalso:
    def __init__(self):
        self.procesos = []

    def __call__(self, argumentos):
        proceso = ProcesoFalso(list(argumentos))
        self.procesos.append(proceso)
        return proceso

    def lanzados(self, comando=None):
        return [p.argumentos for p in self.procesos if comando in (None, p.argumentos[0])]

    def terminar(self, comando, resultado):
        proceso = next(
            p for p in self.procesos if p.argumentos[0] == comando and p.salida is None
        )
        proceso.salida = resultado


class BaseServicio(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cola = self.tmp / "cola"
        self.renderizando = self.tmp / "Renderizando"
        self.cola.mkdir()
        self.renderizando.mkdir()
        parches = [
            mock.patch.object(pa, "DIRS_ORIGEN_RENDER", (self.cola,)),
            mock.patch.object(pa, "DIR_RENDERIZANDO", self.renderizando),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "pendientes.json"),
            mock.patch.object(pa, "render_en_curso", return_value=False),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)
        self.addCleanup(shutil.rmtree, self.tmp)
        self.telegram = TelegramFalso()
        self.lanzador = LanzadorFalso()
        # Hora de pared: en 0 la corrida diaria de `apelar` no toca (otro test la mueve).
        self.ahora = 0.0
        self.servicio = self.nuevo_servicio()

    def nuevo_servicio(self):
        servicio = sp.Servicio(
            self.telegram, self.lanzador, estado_json=self.tmp / "servicio.json", ahora=lambda: self.ahora
        )
        servicio.pausa_error_s = 0
        servicio.iniciar()
        return servicio

    def ultima_pregunta(self):
        return [m for m in self.telegram.mensajes if m["botones"]][-1]

    def album_en_cola(self, nombre):
        (self.cola / nombre).mkdir()

    def album_renderizado(self, nombre):
        carpeta = self.renderizando / nombre
        carpeta.mkdir()
        (carpeta / f"{nombre}.mp4").write_bytes(b"video")
        return str(carpeta)

    def verificado(self, nombre, ok=True):
        return {
            "ok": ok,
            "album": nombre,
            "captura": str(self.tmp / "captura.jpg"),
            "captura_busqueda": str(self.tmp / "busqueda.png"),
            "busqueda": f"{nombre} (Full Album)",
            "resumen": f"{nombre} · 45:00\n✓ tiene sonido: sí",
        }

    def pedir_aprobacion(self, nombre="Banda - Disco"):
        """Lleva un álbum renderizado sin aceptar antes hasta el mensaje con Subir."""
        carpeta = self.album_renderizado(nombre)
        self.servicio.paso()
        self.lanzador.terminar("verificar", self.verificado(nombre))
        self.servicio.paso()
        return carpeta, self.telegram.mensajes[-1]

    def preguntar_antes_de_renderizar(self, nombre="Banda - Disco", apartados=()):
        """Lleva un álbum de la cola hasta la pregunta de antes del render."""
        if not (self.cola / nombre).is_dir():
            self.album_en_cola(nombre)
        self.servicio.paso()
        self.lanzador.terminar(
            "buscar",
            {
                "ok": True,
                "album": nombre,
                "carpeta": str(self.cola / nombre),
                "portada": str(self.tmp / "cover.jpg"),
                "captura_busqueda": str(self.tmp / "busqueda.png"),
                "busqueda": f"{nombre} (Full Album)",
                "ya_publicados": list(apartados),
            },
        )
        self.servicio.paso()
        return str(self.cola / nombre), self.ultima_pregunta()

    def aceptar(self, nombre="Banda - Disco"):
        """Acepta el álbum antes del render; devuelve su carpeta en la cola."""
        carpeta, pregunta = self.preguntar_antes_de_renderizar(nombre)
        self.telegram.apretar(pregunta, "✅ Renderizar y subir")
        self.servicio.paso()
        return carpeta

    def renderizar_aceptado(self, nombre="Banda - Disco"):
        """Acepta el álbum y simula su render: queda en Renderizando/ y se lanza verificar."""
        self.aceptar(nombre)
        carpeta = self.album_renderizado(nombre)
        shutil.rmtree(self.cola / nombre)
        self.lanzador.terminar("renderizar", {"ok": True, "album": nombre, "carpeta": carpeta})
        self.servicio.paso()
        return carpeta


class TestAprobacion(BaseServicio):
    def test_album_renderizado_llega_en_un_solo_mensaje_con_foto_y_botones(self):
        carpeta, mensaje = self.pedir_aprobacion()

        self.assertEqual(self.lanzador.lanzados("verificar"), [["verificar", "--carpeta", carpeta]])
        busqueda, pregunta = self.telegram.mensajes
        self.assertEqual(busqueda["foto"], str(self.tmp / "busqueda.png"))
        self.assertIn("Banda - Disco (Full Album)", busqueda["texto"])
        self.assertEqual(busqueda["botones"], [])
        self.assertIs(pregunta, mensaje)
        self.assertEqual(mensaje["foto"], str(self.tmp / "captura.jpg"))
        self.assertIn("Banda - Disco · 45:00", mensaje["texto"])
        self.assertEqual(
            [etiqueta for etiqueta, _ in mensaje["botones"]],
            ["✅ Subir", "❌ Rechazar", "🗑 Eliminar (ya está publicado)"],
        )

    def test_no_sube_ni_renderiza_otro_mientras_no_haya_respuesta(self):
        self.album_en_cola("Banda - Otro")
        self.pedir_aprobacion()
        for _ in range(3):
            self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [])
        self.assertEqual(self.lanzador.lanzados("buscar"), [])

    def test_boton_de_otro_usuario_no_sube_nada(self):
        _, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir", de=99)
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [])

    def test_doble_toque_en_subir_sube_una_sola_vez(self):
        carpeta, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir")
        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [["subir", "--carpeta", carpeta]])

    def test_subir_programa_el_video_y_sigue_con_el_proximo_album(self):
        carpeta, mensaje = self.pedir_aprobacion()
        self.album_en_cola("Banda - Siguiente")

        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        shutil.rmtree(carpeta)
        self.lanzador.terminar(
            "subir", {"ok": True, "video_id": "VID1", "titulo": "Banda - Disco", "carpeta_borrada": True}
        )
        self.servicio.paso()

        self.assertEqual(
            self.lanzador.lanzados("programar"), [["programar", "--video-id", "VID1", "--esperar"]]
        )
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])
        self.assertIn("Banda - Disco", self.telegram.mensajes[-1]["texto"])

    def test_si_falla_la_subida_vuelve_a_preguntar(self):
        carpeta, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        self.lanzador.terminar("subir", {"ok": False, "motivo": "limite_canal"})
        self.servicio.paso()
        self.telegram.apretar(self.ultima_pregunta(), "✅ Subir")
        self.servicio.paso()

        self.assertEqual(
            self.lanzador.lanzados("subir"),
            [["subir", "--carpeta", carpeta], ["subir", "--carpeta", carpeta]],
        )

    def test_album_subido_sin_poder_borrar_la_carpeta_no_se_vuelve_a_ofrecer(self):
        _, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        self.lanzador.terminar(
            "subir", {"ok": True, "video_id": "VID1", "titulo": "T", "carpeta_borrada": False}
        )
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(len(self.lanzador.lanzados("verificar")), 1)
        self.assertEqual(len([m for m in self.telegram.mensajes if m["botones"]]), 1)

    def test_miniatura_que_no_se_pudo_poner_se_avisa_con_el_enlace_de_studio(self):
        _, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        self.lanzador.terminar(
            "subir",
            {
                "ok": True, "video_id": "VID1", "titulo": "T", "carpeta_borrada": True,
                "miniatura": False, "miniatura_error": "HttpError 403",
            },
        )
        self.servicio.paso()

        aviso = next(m["texto"] for m in self.telegram.mensajes if "Subido como privado" in m["texto"])
        self.assertIn("miniatura", aviso)
        self.assertIn("HttpError 403", aviso)
        self.assertIn("https://studio.youtube.com/video/VID1/edit", aviso)

    def test_subida_que_registro_el_video_y_murio_no_pregunta_otra_vez(self):
        _, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        pa.guardar_pendientes([{"video_id": "VID1", "titulo": "T", "album": "Banda - Disco"}])
        self.lanzador.terminar("subir", {"ok": False, "motivo": "sin_respuesta"})
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(len([m for m in self.telegram.mensajes if m["botones"]]), 1)
        self.assertEqual(self.lanzador.lanzados("verificar"), [self.lanzador.lanzados("verificar")[0]])
        self.assertIn(["programar", "--video-id", "VID1", "--esperar"], self.lanzador.lanzados())

    def test_eliminar_borra_el_album_sin_subirlo_y_sigue(self):
        carpeta, mensaje = self.pedir_aprobacion()
        self.album_en_cola("Banda - Siguiente")

        self.telegram.apretar(mensaje, "🗑 Eliminar (ya está publicado)")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [])
        shutil.rmtree(carpeta)
        self.lanzador.terminar("eliminar", {"ok": True, "album": "Banda - Disco"})
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("eliminar"), [["eliminar", "--carpeta", carpeta]])
        self.assertEqual(self.lanzador.lanzados("subir"), [])
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])

    def test_si_no_se_puede_eliminar_no_se_vuelve_a_ofrecer(self):
        _, mensaje = self.pedir_aprobacion()

        self.telegram.apretar(mensaje, "🗑 Eliminar (ya está publicado)")
        self.servicio.paso()
        self.lanzador.terminar("eliminar", {"ok": False, "motivo": "no_se_pudo_borrar"})
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(len(self.lanzador.lanzados("verificar")), 1)
        self.assertTrue(any("No pude borrar" in m["texto"] for m in self.telegram.mensajes))

    def test_rechazar_no_sube_no_vuelve_a_preguntar_y_sigue(self):
        _, mensaje = self.pedir_aprobacion()
        self.album_en_cola("Banda - Siguiente")

        self.telegram.apretar(mensaje, "❌ Rechazar")
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [])
        self.assertEqual(len(self.lanzador.lanzados("verificar")), 1)
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])


class TestAntesDeRenderizar(BaseServicio):
    """El álbum se pregunta con la búsqueda y la portada antes de gastar el render."""

    def test_manda_la_busqueda_y_pregunta_con_la_portada_sin_renderizar(self):
        _, pregunta = self.preguntar_antes_de_renderizar()
        for _ in range(3):
            self.servicio.paso()

        busqueda, mensaje = self.telegram.mensajes
        self.assertEqual(busqueda["foto"], str(self.tmp / "busqueda.png"))
        self.assertIn("Banda - Disco (Full Album)", busqueda["texto"])
        self.assertEqual(busqueda["botones"], [])
        self.assertIs(mensaje, pregunta)
        self.assertEqual(mensaje["foto"], str(self.tmp / "cover.jpg"))
        self.assertIn("Banda - Disco", mensaje["texto"])
        self.assertEqual(
            [etiqueta for etiqueta, _ in mensaje["botones"]],
            ["✅ Renderizar y subir", "🗑 Eliminar (ya está publicado)"],
        )
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])
        self.assertEqual(self.lanzador.lanzados("renderizar"), [])

    def test_eliminar_borra_el_album_de_la_cola_sin_renderizar_y_pregunta_por_el_siguiente(self):
        carpeta, pregunta = self.preguntar_antes_de_renderizar()
        self.album_en_cola("Banda - Siguiente")

        self.telegram.apretar(pregunta, "🗑 Eliminar (ya está publicado)")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("eliminar"), [["eliminar", "--carpeta", carpeta]])
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])
        shutil.rmtree(carpeta)
        self.lanzador.terminar("eliminar", {"ok": True, "album": "Banda - Disco"})
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("renderizar"), [])
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"], ["buscar"]])
        self.assertTrue(any("Borré Banda - Disco" in m["texto"] for m in self.telegram.mensajes))

    def test_si_no_se_puede_eliminar_avisa_que_quedo_en_la_cola(self):
        _, pregunta = self.preguntar_antes_de_renderizar()

        self.telegram.apretar(pregunta, "🗑 Eliminar (ya está publicado)")
        self.servicio.paso()
        self.lanzador.terminar("eliminar", {"ok": False, "motivo": "no_se_pudo_borrar"})
        self.servicio.paso()

        aviso = self.telegram.mensajes[-1]["texto"]
        self.assertIn("No pude borrar Banda - Disco", aviso)
        self.assertIn(f"Quedó en {self.cola.name}/", aviso)

    def test_aceptar_renderiza_verifica_y_sube_sin_volver_a_preguntar(self):
        carpeta_cola = self.aceptar()
        self.assertEqual(
            self.lanzador.lanzados("renderizar"), [["renderizar", "--carpeta", Path(carpeta_cola).name]]
        )
        carpeta = self.album_renderizado("Banda - Disco")
        shutil.rmtree(carpeta_cola)
        self.lanzador.terminar("renderizar", {"ok": True, "album": "Banda - Disco", "carpeta": carpeta})
        self.servicio.paso()
        self.lanzador.terminar("verificar", self.verificado("Banda - Disco"))
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [["subir", "--carpeta", carpeta]])
        self.assertEqual(len([m for m in self.telegram.mensajes if m["botones"]]), 1)

    def test_si_un_chequeo_falla_no_sube_y_pregunta_con_la_captura_del_video(self):
        self.renderizar_aceptado()
        self.lanzador.terminar("verificar", self.verificado("Banda - Disco", ok=False))
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("subir"), [])
        pregunta = self.ultima_pregunta()
        self.assertEqual(pregunta["foto"], str(self.tmp / "captura.jpg"))
        self.assertIn("Hay tests fallidos", pregunta["texto"])
        self.assertIn("✅ Subir", dict(pregunta["botones"]))

    def test_la_pregunta_del_siguiente_sale_cuando_termina_la_subida(self):
        carpeta = self.renderizar_aceptado()
        self.album_en_cola("Banda - Siguiente")
        self.lanzador.terminar("verificar", self.verificado("Banda - Disco"))
        self.servicio.paso()
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])

        shutil.rmtree(carpeta)
        self.lanzador.terminar(
            "subir", {"ok": True, "video_id": "VID1", "titulo": "Banda - Disco", "carpeta_borrada": True}
        )
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"], ["buscar"]])

    def test_un_reinicio_a_mitad_del_render_aceptado_lo_renderiza_sin_volver_a_preguntar(self):
        carpeta_cola = self.aceptar()

        self.lanzador = LanzadorFalso()
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados(), [["renderizar", "--carpeta", Path(carpeta_cola).name]])
        self.assertEqual(len([m for m in self.telegram.mensajes if m["botones"]]), 1)

    def test_un_album_aceptado_que_ya_no_esta_en_la_cola_no_se_vuelve_a_renderizar(self):
        carpeta_cola = self.aceptar()
        shutil.rmtree(carpeta_cola)
        self.lanzador.terminar(
            "renderizar", {"ok": False, "motivo": "render_fallido", "album": "Banda - Disco",
                           "movido_a_error404": True, "log_final": ""}
        )
        self.album_en_cola("Banda - Siguiente")
        self.servicio.paso()

        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 1)
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"], ["buscar"]])


class TestAleatorio(BaseServicio):
    """/aleatorio prende y apaga la elección al azar del próximo álbum."""

    def test_aleatorio_se_prende_y_se_apaga_contesta_y_sobrevive_un_reinicio(self):
        escribir_en(self.telegram, "/aleatorio")
        self.servicio.paso()
        self.assertEqual(self.telegram.mensajes[-1]["texto"], "🔀 Aleatorio activado")

        self.servicio = self.nuevo_servicio()
        self.album_en_cola("Banda - Disco")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar", "--aleatorio"]])

        escribir_en(self.telegram, "/aleatorio@PublicarBot")
        self.servicio.paso()
        self.assertEqual(self.telegram.mensajes[-1]["texto"], "Aleatorio desactivado")
        self.lanzador.terminar("buscar", {"ok": False, "motivo": "sin_albumes", "ya_publicados": []})
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar", "--aleatorio"], ["buscar"]])

    def test_aleatorio_de_otro_usuario_u_otro_texto_no_cambia_nada(self):
        escribir_en(self.telegram, "/aleatorio", de=999)
        escribir_en(self.telegram, "hola")
        self.album_en_cola("Banda - Disco")
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(self.telegram.mensajes, [])
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])


class TestCrearLista(BaseServicio):
    """/crear_lista: lo aprobado se guarda en una lista y ▶️ la renderiza y la sube entera."""

    def etiquetas(self, mensaje):
        return [etiqueta for etiqueta, _ in mensaje["botones"]]

    def en_lista(self, *nombres):
        """Prende el modo lista y suma cada álbum con ➕; no deja ninguna pregunta abierta."""
        escribir_en(self.telegram, "/crear_lista")
        for nombre in nombres:
            _, pregunta = self.preguntar_antes_de_renderizar(nombre)
            self.telegram.apretar(pregunta, "➕ A la lista")
            self.servicio.paso()

    def renderizar_y_subir(self, nombre, video_id, ok=True):
        """Simula render, chequeos y subida del álbum de la lista que se está renderizando."""
        carpeta = self.album_renderizado(nombre)
        shutil.rmtree(self.cola / nombre)
        self.lanzador.terminar("renderizar", {"ok": True, "album": nombre, "carpeta": carpeta})
        self.servicio.paso()
        self.lanzador.terminar("verificar", self.verificado(nombre, ok=ok))
        self.servicio.paso()
        if ok:
            shutil.rmtree(carpeta)
            self.lanzador.terminar(
                "subir", {"ok": True, "video_id": video_id, "titulo": nombre, "carpeta_borrada": True}
            )
            self.servicio.paso()

    def test_a_la_lista_guarda_el_album_sin_renderizar_y_pregunta_el_siguiente(self):
        # Defecto: sin esto, aprobar un álbum lo renderiza enseguida y el teléfono suena cada vez.
        self.en_lista("Banda A - Uno", "Banda B - Dos")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")

        self.assertEqual(self.lanzador.lanzados("renderizar"), [])
        self.assertEqual(
            self.lanzador.lanzados("buscar")[-1],
            ["buscar", "--excluir=Banda A - Uno", "--excluir=Banda B - Dos"],
        )
        self.assertEqual(
            self.etiquetas(pregunta),
            ["➕ A la lista", "🗑 Eliminar (ya está publicado)", "▶️ Renderizar la lista (2)"],
        )

    def test_renderizar_la_lista_sube_todo_sin_preguntar_y_despues_vuelve_al_de_la_pantalla(self):
        self.en_lista("Banda A - Uno", "Banda B - Dos")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")
        preguntas_antes = len([m for m in self.telegram.mensajes if m["botones"]])

        self.telegram.apretar(pregunta, "▶️ Renderizar la lista (2)")
        self.servicio.paso()
        self.servicio.paso()
        self.renderizar_y_subir("Banda A - Uno", "VID1")
        self.renderizar_y_subir("Banda B - Dos", "VID2")
        self.servicio.paso()

        self.assertEqual(
            self.lanzador.lanzados("renderizar"),
            [["renderizar", "--carpeta", "Banda A - Uno"], ["renderizar", "--carpeta", "Banda B - Dos"]],
        )
        self.assertEqual(len(self.lanzador.lanzados("subir")), 2)
        con_botones = [m for m in self.telegram.mensajes if m["botones"]]
        # Una sola pregunta después de ▶️: la del álbum que estaba en pantalla, ya sin la lista.
        self.assertEqual(len(con_botones), preguntas_antes + 1)
        self.assertIn("Banda C - Tres", con_botones[-1]["texto"])
        self.assertEqual(
            self.etiquetas(con_botones[-1]), ["✅ Renderizar y subir", "🗑 Eliminar (ya está publicado)"]
        )
        # Pasaron horas: la búsqueda vuelve a llegar antes de la pregunta.
        self.assertEqual(self.telegram.mensajes[-2]["foto"], str(self.tmp / "busqueda.png"))

    def test_un_chequeo_que_falla_pregunta_y_la_lista_sigue(self):
        # Defecto: la pregunta de un chequeo fallido frenaba el resto de la lista hasta contestarla.
        self.en_lista("Banda A - Uno", "Banda B - Dos")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")
        self.telegram.apretar(pregunta, "▶️ Renderizar la lista (2)")
        self.servicio.paso()
        self.servicio.paso()

        self.renderizar_y_subir("Banda A - Uno", "VID1", ok=False)
        self.servicio.paso()

        self.assertIn("✅ Subir", self.etiquetas(self.ultima_pregunta()))
        self.assertEqual(
            self.lanzador.lanzados("renderizar")[-1], ["renderizar", "--carpeta", "Banda B - Dos"]
        )
        self.assertEqual(len(self.lanzador.lanzados("verificar")), 1)
        self.assertEqual(self.lanzador.lanzados("subir"), [])

    def test_al_terminar_la_lista_la_pregunta_del_chequeo_fallido_frena_lo_nuevo(self):
        # Defecto: con un aceptado ya movido a Error404, se preguntaba por el álbum siguiente
        # mientras la pregunta del chequeo fallido seguía abierta.
        self.en_lista("Banda A - Uno", "Banda B - Dos")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")
        self.telegram.apretar(pregunta, "▶️ Renderizar la lista (2)")
        self.servicio.paso()
        self.servicio.paso()
        self.renderizar_y_subir("Banda A - Uno", "VID1", ok=False)
        fallida = self.ultima_pregunta()
        shutil.rmtree(self.cola / "Banda B - Dos")
        self.lanzador.terminar(
            "renderizar",
            {"ok": False, "motivo": "render_fallido", "album": "Banda B - Dos", "movido_a_error404": True},
        )
        self.servicio.paso()
        self.servicio.paso()

        self.assertIs(self.ultima_pregunta(), fallida)
        self.telegram.apretar(fallida, "❌ Rechazar")
        self.servicio.paso()
        self.servicio.paso()
        self.assertIn("Banda C - Tres", self.ultima_pregunta()["texto"])

    def test_una_subida_que_falla_frena_el_resto_de_la_lista(self):
        # Defecto: con el límite del canal, cada álbum restante se renderizaría y fallaría igual.
        self.en_lista("Banda A - Uno", "Banda B - Dos")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")
        self.telegram.apretar(pregunta, "▶️ Renderizar la lista (2)")
        self.servicio.paso()
        self.servicio.paso()
        carpeta = self.album_renderizado("Banda A - Uno")
        shutil.rmtree(self.cola / "Banda A - Uno")
        self.lanzador.terminar("renderizar", {"ok": True, "album": "Banda A - Uno", "carpeta": carpeta})
        self.servicio.paso()
        self.lanzador.terminar("verificar", self.verificado("Banda A - Uno"))
        self.servicio.paso()

        self.lanzador.terminar("subir", {"ok": False, "motivo": "limite_canal"})
        self.servicio.paso()
        self.servicio.paso()

        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 1)

    def test_sin_mas_albumes_llega_solo_el_boton_de_renderizar(self):
        # Defecto: sin álbumes nuevos no había pregunta con ▶️ y buscar se relanzaba en cada vuelta.
        self.en_lista("Banda A - Uno")
        self.servicio.paso()
        self.lanzador.terminar("buscar", {"ok": False, "motivo": "sin_albumes", "ya_publicados": []})
        self.servicio.paso()
        self.servicio.paso()

        final = self.ultima_pregunta()
        self.assertEqual(self.etiquetas(final), ["▶️ Renderizar la lista (1)"])
        self.assertEqual(len(self.lanzador.lanzados("buscar")), 2)
        self.telegram.apretar(final, "▶️ Renderizar la lista (1)")
        self.servicio.paso()
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("renderizar"), [["renderizar", "--carpeta", "Banda A - Uno"]])

    def test_crear_lista_otra_vez_no_borra_la_lista_y_dice_cuantos_hay(self):
        # Defecto: si funcionara como /aleatorio, volver a escribirlo tiraría lo ya elegido.
        self.en_lista("Banda A - Uno")
        self.preguntar_antes_de_renderizar("Banda C - Tres")
        escribir_en(self.telegram, "/crear-lista")
        self.servicio.paso()

        self.assertIn("1 álbum", [m for m in self.telegram.mensajes if not m["botones"]][-1]["texto"])
        self.assertIn("▶️ Renderizar la lista (1)", self.etiquetas(self.ultima_pregunta()))

    def test_la_pregunta_abierta_al_prender_el_modo_vuelve_con_los_botones_de_la_lista(self):
        # Defecto: la pregunta vieja seguía con "Renderizar y subir" y renderizaba al tocarla.
        _, vieja = self.preguntar_antes_de_renderizar("Banda A - Uno")
        escribir_en(self.telegram, "/crear_lista")
        self.servicio.paso()

        self.assertIn(self.telegram.mensajes.index(vieja) + 1, self.telegram.sin_botones)
        nueva = self.ultima_pregunta()
        self.assertIn("Banda A - Uno", nueva["texto"])
        self.assertEqual(self.etiquetas(nueva)[0], "➕ A la lista")

    def test_la_lista_y_el_modo_sobreviven_un_reinicio(self):
        self.en_lista("Banda A - Uno")
        self.servicio = self.nuevo_servicio()
        self.album_en_cola("Banda C - Tres")
        _, pregunta = self.preguntar_antes_de_renderizar("Banda C - Tres")

        self.assertIn(["buscar", "--excluir=Banda A - Uno"], self.lanzador.lanzados("buscar"))
        self.assertEqual(self.etiquetas(pregunta)[0], "➕ A la lista")


class TestReclamos(BaseServicio):
    def test_programado_manda_el_resumen(self):
        carpeta, mensaje = self.pedir_aprobacion()
        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()
        shutil.rmtree(carpeta)
        self.lanzador.terminar("subir", {"ok": True, "video_id": "VID1", "titulo": "T", "carpeta_borrada": True})
        self.servicio.paso()

        self.lanzador.terminar(
            "programar", {"ok": True, "video_id": "VID1", "resumen": "📅 T se publica el 02/11 10:00"}
        )
        self.servicio.paso()

        self.assertEqual(self.telegram.mensajes[-1]["texto"], "📅 T se publica el 02/11 10:00")

    def terminar_programar_con(self, monetizacion):
        self.servicio.lanzar_paso("programar", "--video-id", "VID1", "--esperar")
        self.lanzador.terminar(
            "programar",
            {
                "ok": True,
                "video_id": "VID1",
                "titulo": "Banda - Disco",
                "resumen": "✓ Banda - Disco\nSin reclamos: programado para 2026-11-02 18:00.",
                "monetizacion": monetizacion,
            },
        )
        self.servicio.paso()
        # El mismo paso puede avisar otras cosas (la cola de álbumes vacía).
        return [m["texto"] for m in self.telegram.mensajes if "Banda - Disco" in m["texto"]]

    def test_si_falla_la_monetizacion_avisa_con_el_link_de_studio_para_hacerlo_a_mano(self):
        textos = self.terminar_programar_con(
            {"estado": "fallo", "motivo": "cuestionario_ya_enviado", "captura": "/tmp/m.png"}
        )
        self.assertEqual(len(textos), 2, textos)
        self.assertIn("Sin reclamos: programado", textos[0])
        self.assertIn("Banda - Disco", textos[1])
        # Es irreversible: el aviso tiene que decir que el cuestionario ya salió.
        self.assertIn("ya envié el cuestionario", textos[1])
        self.assertIn("https://studio.youtube.com/video/VID1/monetization", textos[1])

    def test_monetizacion_activada_u_omitida_no_manda_aviso_aparte(self):
        for monetizacion in ({"estado": "activada"}, {"estado": "ya_activa"}, {"estado": "omitida"}):
            with self.subTest(monetizacion=monetizacion):
                self.telegram.mensajes.clear()
                self.assertEqual(len(self.terminar_programar_con(monetizacion)), 1)

    def test_reclamo_dudoso_pregunta_y_programa_con_la_respuesta(self):
        self.servicio.lanzar_paso("programar", "--video-id", "VID1", "--esperar")
        self.lanzador.terminar(
            "programar",
            {
                "ok": False,
                "motivo": "reclamo_desconocido",
                "video_id": "VID1",
                "titulo": "Banda - Disco",
                "detalle": "texto_no_reconocido",
                "captura_studio": str(self.tmp / "studio.png"),
            },
        )
        self.servicio.paso()
        pregunta = self.ultima_pregunta()

        self.telegram.apretar(pregunta, "Sí, tiene reclamo")
        self.servicio.paso()

        self.assertEqual(pregunta["foto"], str(self.tmp / "studio.png"))
        self.assertIn("studio.youtube.com/video/VID1/copyright", pregunta["texto"])
        self.assertEqual(
            self.lanzador.lanzados("programar")[-1],
            ["programar", "--video-id", "VID1", "--reclamo", "si"],
        )


class TestListaNegra(BaseServicio):
    """Un video con reclamo de un sello de la lista negra: aviso con botón para eliminarlo."""

    ELIMINAR = "🗑️ Eliminar video"
    CONSERVAR = "Conservar"

    def terminar_programar(self, en_lista_negra=("Believe Music, UMG",)):
        resultado = {
            "ok": True,
            "video_id": "VID1",
            "titulo": "Banda - Disco",
            "con_reclamo": True,
            "reclamantes": ["Believe Music", "Believe Music, UMG"],
            "captura_studio": str(self.tmp / "studio.png"),
            "monetizacion": {"estado": "omitida", "motivo": "con_reclamo"},
            "resumen": "⚠️ Banda - Disco\nTiene reclamo (Believe Music, UMG): programado para 2028-11-02 10:00.",
        }
        if en_lista_negra is not None:
            resultado["reclamantes_en_lista_negra"] = list(en_lista_negra)
        self.servicio.lanzar_paso("programar", "--video-id", "VID1", "--esperar")
        self.lanzador.terminar("programar", resultado)
        self.servicio.paso()

    def preguntas(self):
        return [m for m in self.telegram.mensajes if m["botones"]]

    def test_pregunta_con_el_video_el_reclamante_el_link_de_studio_y_la_captura(self):
        # Defecto: avisar sin decir qué video, quién reclama o dónde mirarlo, o sin los botones.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()

        self.assertEqual(len(self.preguntas()), 1)
        self.assertIn("Banda - Disco", pregunta["texto"])
        self.assertIn("Believe Music, UMG", pregunta["texto"])
        self.assertIn("https://studio.youtube.com/video/VID1/claims", pregunta["texto"])
        self.assertEqual(pregunta["foto"], str(self.tmp / "studio.png"))
        self.assertEqual([etiqueta for etiqueta, _ in pregunta["botones"]], [self.CONSERVAR, self.ELIMINAR])

    def test_el_resumen_de_programado_sigue_llegando_antes_de_la_pregunta(self):
        # Comportamiento previo: el aviso de que el video quedó programado no se pierde.
        self.terminar_programar()
        mensajes = [m for m in self.telegram.mensajes if "Banda - Disco" in m["texto"]]
        self.assertEqual(len(mensajes), 2, mensajes)
        self.assertIn("programado para 2028-11-02", mensajes[0]["texto"])
        self.assertFalse(mensajes[0]["botones"])
        self.assertTrue(mensajes[1]["botones"])

    def test_eliminar_video_lanza_el_paso_que_lo_borra_y_avisa_que_se_borro(self):
        # Defecto: apretar el botón sin que se borre nada, o borrar sin decirle al usuario.
        self.terminar_programar()
        self.telegram.apretar(self.ultima_pregunta(), self.ELIMINAR)
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [["eliminar_video", "--video-id", "VID1"]])

        self.lanzador.terminar("eliminar_video", {"ok": True, "video_id": "VID1", "titulo": "Banda - Disco"})
        self.servicio.paso()
        self.assertIn("Borré", self.telegram.mensajes[-1]["texto"])
        self.assertIn("Banda - Disco", self.telegram.mensajes[-1]["texto"])

    def test_si_no_se_pudo_borrar_el_aviso_dice_que_el_video_sigue_en_youtube(self):
        # Defecto: dar por borrado un video que YouTube no borró.
        self.terminar_programar()
        self.telegram.apretar(self.ultima_pregunta(), self.ELIMINAR)
        self.servicio.paso()
        self.lanzador.terminar("eliminar_video", {"ok": False, "motivo": "error_api", "video_id": "VID1"})
        self.servicio.paso()

        aviso = self.telegram.mensajes[-1]["texto"]
        self.assertIn("No pude borrar", aviso)
        self.assertIn("error_api", aviso)
        self.assertIn("https://studio.youtube.com/video/VID1/edit", aviso)

    def test_si_el_video_ya_no_existe_el_aviso_no_dice_que_sigue_en_youtube(self):
        # Defecto: decir «Sigue en YouTube» de un video que YouTube ya no tiene
        # (borrado a mano, o borrado en un intento cuya respuesta se perdió).
        self.terminar_programar()
        self.telegram.apretar(self.ultima_pregunta(), self.ELIMINAR)
        self.servicio.paso()
        self.lanzador.terminar("eliminar_video", {"ok": False, "motivo": "no_existe", "video_id": "VID1"})
        self.servicio.paso()

        aviso = self.telegram.mensajes[-1]["texto"]
        self.assertIn("VID1", aviso)
        self.assertIn("ya no está en YouTube", aviso)
        self.assertNotIn("Sigue en YouTube", aviso)

    def test_conservar_cierra_la_pregunta_y_un_boton_viejo_ya_no_borra(self):
        # Defecto: que «Conservar» deje la pregunta abierta y un toque posterior en
        # «Eliminar video» borre un video que el usuario decidió conservar.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()

        self.telegram.apretar(pregunta, self.CONSERVAR)
        self.servicio.paso()
        self.telegram.apretar(pregunta, self.ELIMINAR)
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [])
        self.assertEqual(self.servicio.estado["preguntas"], {})

    def test_el_boton_sigue_valiendo_tras_reiniciar_el_servicio(self):
        # Defecto: perder la pregunta (o el id del video) en un reinicio.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()

        self.lanzador = LanzadorFalso()
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()
        self.assertEqual(len(self.preguntas()), 1)
        self.telegram.apretar(pregunta, self.ELIMINAR)
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [["eliminar_video", "--video-id", "VID1"]])

    def test_un_video_con_reclamo_fuera_de_la_lista_negra_solo_recibe_la_pregunta_de_impugnar(self):
        # Sin coincidencia (también con el JSON de antes, que no traía el campo) no llega la
        # pregunta de la lista negra: llega una sola, la de impugnar o eliminar el video.
        for en_lista_negra in ([], None):
            with self.subTest(en_lista_negra=en_lista_negra):
                self.telegram.mensajes.clear()
                self.terminar_programar(en_lista_negra)
                botones = [[etiqueta for etiqueta, _ in pregunta["botones"]] for pregunta in self.preguntas()]
                self.assertEqual(botones, [["⚖️ Impugnar", self.ELIMINAR]])
                # El resumen de programado sigue llegando una sola vez.
                resumenes = [m for m in self.telegram.mensajes if "programado para" in m["texto"]]
                self.assertEqual(len(resumenes), 1, resumenes)
                self.lanzador.procesos.clear()


class TestCopiaRetirada(BaseServicio):
    ELIMINAR = "🗑️ Eliminar video"

    def terminar_programar(self, resultado):
        self.servicio.lanzar_paso("programar", "--video-id", "VID1", "--esperar")
        self.lanzador.terminar("programar", resultado)
        self.servicio.paso()

    def test_una_copia_retirada_avisa_por_que_y_solo_ofrece_eliminarla(self):
        # Defecto: el aviso genérico "rechazó o perdió", sin decir por qué ni ofrecer borrarlo; un
        # botón de más (no hay nada que conservar ni que reintentar); o un "Eliminar" que corre la
        # comprobación de siempre, que con este video dice "ya estaba borrado" sin borrar nada.
        self.terminar_programar({
            "ok": False,
            "motivo": "copia_retirada",
            "video_id": "VID1",
            "titulo": "🇩🇪 Naked Whipper - Chapel Defilement (2025)",
            "detalle": ["De: Naked Whipper Official", "Solicitud recibida: 2 feb 2025"],
            "captura_studio": str(self.tmp / "retiro.png"),
        })
        (pregunta,) = [m for m in self.telegram.mensajes if m["botones"]]
        self.assertEqual([etiqueta for etiqueta, _ in pregunta["botones"]], [self.ELIMINAR])
        texto = pregunta["texto"]
        self.assertIn("Naked Whipper - Chapel Defilement (2025)", texto)
        self.assertIn("No se puede publicar porque es copia de otro", texto)
        self.assertIn("De: Naked Whipper Official", texto)
        self.assertIn("https://studio.youtube.com/video/VID1/edit", texto)
        self.assertEqual(pregunta["foto"], str(self.tmp / "retiro.png"))
        self.assertFalse(any("rechazó o perdió" in m["texto"] for m in self.telegram.mensajes))

        self.telegram.apretar(pregunta, self.ELIMINAR)
        self.servicio.paso()
        self.assertEqual(
            self.lanzador.lanzados("eliminar_video"), [["eliminar_video", "--video-id", "VID1", "--retirado"]]
        )

    def copia_retirada(self, **extra):
        self.terminar_programar({
            "ok": False, "motivo": "copia_retirada", "video_id": "VID1",
            "titulo": "🇩🇪 Naked Whipper - Chapel Defilement (2025)", "detalle": [], "captura_studio": None, **extra,
        })
        (pregunta,) = [m for m in self.telegram.mensajes if m["botones"]]
        return pregunta

    def eliminar_y_terminar(self, resultado):
        """Aprieta Eliminar, deja que `eliminar_video` termine con `resultado` y devuelve el aviso."""
        self.telegram.apretar(self.ultima_pregunta(), self.ELIMINAR)
        self.servicio.paso()
        self.lanzador.terminar("eliminar_video", {"video_id": "VID1", "log": "/logs/eliminar.out", **resultado})
        self.servicio.paso()
        return self.telegram.mensajes[-1]["texto"]

    def test_el_aviso_de_borrado_nombra_el_album_y_no_el_id(self):
        # Defecto: con --retirado `eliminar_video` no consulta el video y devuelve el id como
        # título: el aviso decía «Borré el video "VID1"» en vez del nombre del álbum.
        self.copia_retirada()
        aviso = self.eliminar_y_terminar({"ok": True, "titulo": "VID1"})
        self.assertIn("🇩🇪 Naked Whipper - Chapel Defilement (2025)", aviso)
        self.assertNotIn("VID1", aviso)

    def test_una_pregunta_guardada_sin_titulo_borra_y_avisa_con_el_id_como_antes(self):
        # Comportamiento previo: una pregunta de antes del cambio no trae `titulo`.
        self.copia_retirada()
        for pregunta in self.servicio.estado["preguntas"].values():
            pregunta.pop("titulo", None)
        aviso = self.eliminar_y_terminar({"ok": True, "titulo": "VID1"})
        self.assertIn("Borré el video \"VID1\"", aviso)

    def test_un_404_al_borrar_dice_que_youtube_no_lo_encuentra_y_no_que_sigue(self):
        # Defecto: un 404 de videos.delete puede ser un video que ya no existía, pero el aviso
        # afirmaba «Sigue en YouTube». Cualquier otro error de la API conserva el aviso de siempre.
        self.copia_retirada()
        aviso = self.eliminar_y_terminar({"ok": False, "motivo": "no_encontrado"})
        self.assertIn("YouTube no encuentra el video VID1", aviso)
        self.assertIn("https://studio.youtube.com/video/VID1/edit", aviso)
        self.assertNotIn("Sigue en YouTube", aviso)

    def test_otro_error_de_la_api_al_borrar_sigue_diciendo_que_el_video_sigue_en_youtube(self):
        # Comportamiento previo: solo el 404 cambia; un 403 no borró nada y el video sigue.
        self.copia_retirada()
        aviso = self.eliminar_y_terminar({"ok": False, "motivo": "error_api"})
        self.assertIn("Sigue en YouTube", aviso)

    def test_el_aviso_solo_dice_que_el_album_esta_en_el_registro_si_quedo_anotado(self):
        # Defecto: sin entrada en pendientes (o sin álbum) `programar` no anota nada, pero el
        # aviso aseguraba que el álbum no se vuelve a renderizar ni a subir.
        for registrado in (True, False):
            with self.subTest(album_registrado=registrado):
                self.telegram.mensajes.clear()
                texto = self.copia_retirada(album_registrado=registrado)["texto"]
                self.assertIn("Ya no queda pendiente de programar", texto)
                self.assertEqual("registro de publicados" in texto, registrado)
                self.servicio.estado["preguntas"].clear()

    def test_un_video_que_no_existe_sigue_con_el_aviso_de_siempre(self):
        # Comportamiento previo: `no_existe` avisa una vez, sin pregunta.
        self.terminar_programar({"ok": False, "motivo": "no_existe", "status": {}, "video_id": "VID1"})
        self.assertEqual([m for m in self.telegram.mensajes if m["botones"]], [])
        avisos = [m["texto"] for m in self.telegram.mensajes if "VID1" in m["texto"]]
        self.assertEqual(avisos, ["❌ YouTube rechazó o perdió el video VID1 (no_existe)."])


class TestImpugnarDesdeTelegram(BaseServicio):
    """Un video con reclamo fuera de la lista negra: pregunta con botón para impugnarlo.

    Impugnar es una declaración legal a nombre del usuario: nada se impugna sin que apriete
    el «⚖️ Impugnar» de ese video.
    """

    IMPUGNAR = "⚖️ Impugnar"
    ELIMINAR = "🗑️ Eliminar video"
    ENLACE = "https://studio.youtube.com/video/VID1/claims"

    def terminar_programar(self, con_reclamo=True, en_lista_negra=(), reclamantes=("Believe Music", "UMG")):
        resultado = {
            "ok": True,
            "video_id": "VID1",
            "titulo": "Banda - Disco",
            "con_reclamo": con_reclamo,
            "reclamantes": list(reclamantes) if con_reclamo else [],
            "reclamantes_en_lista_negra": list(en_lista_negra),
            "captura_studio": str(self.tmp / "studio.png"),
            "monetizacion": {"estado": "omitida", "motivo": "con_reclamo"},
            "resumen": "⚠️ Banda - Disco\nTiene reclamo: programado para 2028-11-02 10:00.",
        }
        self.servicio.lanzar_paso("programar", "--video-id", "VID1", "--esperar")
        self.lanzador.terminar("programar", resultado)
        self.servicio.paso()

    def preguntas(self):
        return [m for m in self.telegram.mensajes if m["botones"]]

    def etiquetas(self, mensaje):
        return [etiqueta for etiqueta, _ in mensaje["botones"]]

    def lanzar_impugnar(self):
        self.terminar_programar()
        self.telegram.apretar(self.ultima_pregunta(), self.IMPUGNAR)
        self.servicio.paso()

    def test_pregunta_con_el_video_los_reclamantes_el_link_y_la_captura(self):
        # Defecto: preguntar sin decir qué video, quién reclama ni dónde mirarlo, sin
        # advertir que es una declaración con la licencia y la firma ya configuradas, o sin
        # la opción de borrar el video.
        self.terminar_programar()
        (pregunta,) = self.preguntas()
        self.assertEqual(self.etiquetas(pregunta), [self.IMPUGNAR, self.ELIMINAR])
        self.assertIn("Banda - Disco", pregunta["texto"])
        self.assertIn("Believe Music", pregunta["texto"])
        self.assertIn("UMG", pregunta["texto"])
        self.assertIn(self.ENLACE, pregunta["texto"])
        self.assertIn("Licencia", pregunta["texto"])
        self.assertEqual(pregunta["foto"], str(self.tmp / "studio.png"))

    def test_sin_reclamantes_conocidos_pregunta_igual_y_no_deja_una_lista_vacia(self):
        # Defecto: el usuario contestó «Sí, tiene reclamo» a mano y no hay nombres que mostrar;
        # no preguntar dejaría ese video sin la opción, y una lista vacía confunde.
        self.terminar_programar(reclamantes=())
        (pregunta,) = self.preguntas()
        self.assertIn("Banda - Disco", pregunta["texto"])
        self.assertNotIn("•", pregunta["texto"])
        self.assertIn(self.ENLACE, pregunta["texto"])

    def test_impugnar_lanza_el_paso_de_ese_video_y_nada_mas(self):
        # Defecto: apretar el botón sin que se impugne nada, o impugnar otro video o borrarlo.
        self.lanzar_impugnar()
        self.assertEqual(self.lanzador.lanzados("impugnar"), [["impugnar", "--video-id", "VID1"]])
        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [])
        self.assertEqual(self.servicio.estado["preguntas"], {})

    def test_eliminar_video_borra_ese_video_y_un_impugnar_tardio_se_rechaza(self):
        # Defecto: que «Eliminar video» no borre nada, borre otro video o además impugne, o
        # que deje la pregunta abierta y un toque posterior en «Impugnar» envíe una
        # declaración legal sobre el video que el usuario decidió borrar.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()

        self.telegram.apretar(pregunta, self.ELIMINAR)
        self.servicio.paso()
        self.telegram.apretar(pregunta, self.IMPUGNAR)
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [["eliminar_video", "--video-id", "VID1"]])
        self.assertEqual(self.lanzador.lanzados("impugnar"), [])
        self.assertEqual(self.servicio.estado["preguntas"], {})
        self.assertIn("ya no está vigente", self.telegram.respuestas[-1])

    def test_impugnar_dos_veces_seguidas_lanza_un_solo_paso(self):
        # Defecto: un doble toque en el botón enviaría dos veces la misma declaración legal.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()
        self.telegram.apretar(pregunta, self.IMPUGNAR)
        self.telegram.apretar(pregunta, self.IMPUGNAR)
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("impugnar"), [["impugnar", "--video-id", "VID1"]])

    def test_un_video_de_la_lista_negra_solo_pregunta_por_eliminarlo(self):
        # Defecto: sumar la pregunta de impugnar a la de la lista negra: el usuario ya decidió
        # que ese reclamante no se discute y se le pediría una segunda decisión.
        self.terminar_programar(en_lista_negra=("Believe Music",))
        (pregunta,) = self.preguntas()
        self.assertEqual(self.etiquetas(pregunta), ["Conservar", "🗑️ Eliminar video"])
        self.assertNotIn(self.IMPUGNAR, self.etiquetas(pregunta))

    def test_un_video_sin_reclamo_no_pregunta_nada(self):
        # Defecto: ofrecer impugnar un video sin reclamos.
        self.terminar_programar(con_reclamo=False)
        self.assertEqual(self.preguntas(), [])

    def test_la_pregunta_sigue_valiendo_tras_reiniciar_el_servicio(self):
        # Defecto: perder la pregunta (o el id del video) en un reinicio.
        self.terminar_programar()
        pregunta = self.ultima_pregunta()

        self.lanzador = LanzadorFalso()
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()
        self.assertEqual(len(self.preguntas()), 1)
        self.telegram.apretar(pregunta, self.IMPUGNAR)
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("impugnar"), [["impugnar", "--video-id", "VID1"]])

    def test_el_resumen_de_programado_sigue_llegando_antes_de_la_pregunta(self):
        # Comportamiento previo: el aviso de que el video quedó programado no se pierde.
        self.terminar_programar()
        mensajes = [m for m in self.telegram.mensajes if "programado para" in m["texto"]]
        self.assertEqual(len(mensajes), 1)
        self.assertFalse(mensajes[0]["botones"])
        self.assertLess(self.telegram.mensajes.index(mensajes[0]), self.telegram.mensajes.index(self.ultima_pregunta()))

    def terminar_impugnar(self, resultado):
        self.lanzar_impugnar()
        self.lanzador.terminar("impugnar", {"video_id": "VID1", "titulo": "Banda - Disco", **resultado})
        self.servicio.paso()
        return self.telegram.mensajes[-1]

    def test_informa_cuantos_reclamos_se_impugnaron(self):
        # Defecto: no decirle al usuario cómo salió, o no mostrar la captura de /claims.
        aviso = self.terminar_impugnar(
            {"ok": True, "reclamos": 5, "impugnados": 5, "fallos": [], "captura": str(self.tmp / "claims.png")}
        )
        self.assertIn("Impugné 5 de 5", aviso["texto"])
        self.assertIn("Banda - Disco", aviso["texto"])
        self.assertEqual(aviso["foto"], str(self.tmp / "claims.png"))
        self.assertFalse(aviso["botones"])

    def test_un_resultado_parcial_dice_cuantos_salieron_por_que_fallo_y_donde_terminar(self):
        # Defecto: dar por impugnado todo el video, o callar la causa y el link para terminar
        # a mano los reclamos que quedaron.
        aviso = self.terminar_impugnar(
            {
                "ok": False,
                "reclamos": 5,
                "impugnados": 4,
                "fallos": [{"motivo": "verificacion_de_identidad", "detalle": "2fa_no_completado (timeout)"}],
            }
        )
        self.assertIn("Impugné 4 de 5", aviso["texto"])
        self.assertIn("verific", aviso["texto"].lower())
        self.assertIn(self.ENLACE, aviso["texto"])

    def test_un_envio_no_confirmado_pide_revisar_studio(self):
        # Defecto: avisar «Impugné» cuando Studio no mostró el reclamo como impugnado.
        aviso = self.terminar_impugnar(
            {"ok": False, "reclamos": 2, "impugnados": 0, "fallos": [{"motivo": "no_confirmado", "detalle": "x"}]}
        )
        self.assertIn("Impugné 0 de 2", aviso["texto"])
        self.assertIn("no sé si salió", aviso["texto"])
        self.assertIn(self.ENLACE, aviso["texto"])
        self.assertFalse(aviso["botones"])  # solo la verificación ofrece la ventana

    def test_una_verificacion_de_identidad_ofrece_impugnar_con_ventana(self):
        # Defecto: el aviso decía que el servicio no puede pasar la verificación y el video
        # quedaba sin impugnar, sin forma de terminarlo desde Telegram.
        aviso = self.terminar_impugnar(
            {"ok": False, "reclamos": 3, "impugnados": 0, "log": "/logs/impugnar.out",
             "fallos": [{"motivo": "verificacion_de_identidad", "detalle": "2fa_no_completado (timeout)"}]}
        )
        self.assertEqual([etiqueta for etiqueta, _ in aviso["botones"]], ["🖥 Impugnar con ventana"])
        self.assertIn("Verifica que eres tú", aviso["texto"])
        self.assertIn(self.ENLACE, aviso["texto"])
        # Defecto: esta rama del aviso perdió la línea con el log del paso que traen los demás.
        self.assertIn("Log: /logs/impugnar.out", aviso["texto"])
        self.assertEqual(self.lanzador.lanzados("impugnar_con_ventana"), [])
        self.telegram.apretar(aviso, "🖥 Impugnar con ventana")
        self.servicio.paso()
        self.assertEqual(
            self.lanzador.lanzados("impugnar_con_ventana"), [["impugnar_con_ventana", "--video-id", "VID1"]]
        )

    def test_sin_reclamos_impugnables_dice_que_no_habia_nada_que_impugnar(self):
        # Defecto: reportar un fallo (o «Impugné 0 de 0») cuando simplemente no había nada.
        aviso = self.terminar_impugnar({"ok": False, "motivo": "sin_reclamos_impugnables", "reclamos": 0})
        self.assertIn("no tiene reclamos que se puedan impugnar", aviso["texto"])
        self.assertNotIn("Impugné", aviso["texto"])

    def test_un_fallo_antes_de_empezar_no_dice_que_se_impugno_algo(self):
        # Defecto: sin sesión de Studio (o sin respuesta del paso) no hay conteos: el aviso no
        # debe inventar «Impugné 0 de 0» y sí decir qué pasó y dónde revisar.
        for resultado, causa in (
            ({"ok": False, "motivo": "sin_sesion_en_studio"}, "sesión"),
            ({"ok": False, "motivo": "sin_respuesta", "codigo": -9, "log": "/tmp/x.out"}, "sin_respuesta"),
        ):
            with self.subTest(motivo=resultado["motivo"]):
                aviso = self.terminar_impugnar(resultado)
                self.assertIn("No pude impugnar", aviso["texto"])
                self.assertIn(causa, aviso["texto"])
                self.assertIn(self.ENLACE, aviso["texto"])
                self.assertNotIn("Impugné", aviso["texto"])
                self.lanzador.procesos.clear()
                self.telegram.mensajes.clear()


class TestApelaciones(BaseServicio):
    DIA = sp.INTERVALO_APELACIONES_S

    def test_apela_una_vez_por_dia_y_lo_recuerda_tras_un_reinicio(self):
        # Defecto: lanzar `apelar` en cada vuelta del ciclo, o de nuevo tras cada reinicio.
        self.ahora = self.DIA
        self.servicio.paso()
        self.servicio.paso()  # sigue corriendo: no se lanza otro
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        self.servicio = self.nuevo_servicio()
        self.ahora += self.DIA - 1
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("apelar")), 1)
        self.ahora += 1
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("apelar")), 2)

    def test_avisa_lo_que_mando_y_lo_que_no_pudo(self):
        # Defecto: un aviso que no dice qué video se apeló ni cuál quedó sin confirmar o falló.
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar(
            "apelar",
            {
                "ok": False,
                "log": "/logs/apelar.log",
                "videos": [
                    {"tipo": "edad", "video_id": "ID1", "titulo": "Banda - Uno", "estado": "enviado"},
                    {"tipo": "anuncios", "video_id": "ID2", "titulo": "Banda - Dos", "estado": "no_confirmado"},
                    {"tipo": "edad", "video_id": "ID3", "titulo": "Banda - Tres", "estado": "ya_apelado"},
                    {"tipo": "edad", "video_id": "ID5", "titulo": "Banda - Cinco", "estado": "revisar_a_mano"},
                ],
                "fallos": [
                    {"tipo": "anuncios", "video_id": "ID4", "titulo": "Banda - Cuatro",
                     "motivo": "formulario_no_disponible"},
                ],
            },
        )
        self.servicio.paso()
        texto = self.telegram.mensajes[-1]["texto"]
        self.assertIn("Banda - Uno (restricción de edad)", texto)
        self.assertIn("https://studio.youtube.com/video/ID1/edit", texto)
        self.assertIn("revísalo", texto.split("Banda - Dos")[0])
        self.assertIn("1 ya tenían una apelación", texto)
        a_mano = texto.split("a mano")[1]
        self.assertIn("Banda - Cinco (restricción de edad)", a_mano)
        self.assertIn("https://studio.youtube.com/video/ID5/edit", a_mano)
        self.assertIn("Banda - Cuatro: el formulario de Studio no se parece al esperado", texto)
        self.assertIn("/logs/apelar.log", texto)

    def test_avisa_las_apelaciones_que_youtube_resolvio(self):
        # Defecto: el resultado se anotaba pero el mensaje diario no lo decía (o no salía).
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar(
            "apelar",
            {
                "ok": False,
                "log": "/logs/apelar.log",
                "videos": [],
                "resultados": [
                    {"tipo": "edad", "video_id": "ID6", "titulo": "Banda - Seis", "resultado": "rechazada"},
                    {"tipo": "anuncios", "video_id": "ID7", "titulo": "Banda - Siete", "resultado": "aceptada"},
                ],
                "fallos": [
                    {"tipo": "edad", "video_id": "ID8", "titulo": "Banda - Ocho", "motivo": "dialogo_desconocido",
                     "revision": True},
                ],
            },
        )
        self.servicio.paso()
        texto = self.telegram.mensajes[-1]["texto"]
        rechazadas, resto = texto.split("Banda - Seis (restricción de edad)")
        self.assertIn("rechazó", rechazadas)
        self.assertIn("https://studio.youtube.com/video/ID6/edit", resto.split("Banda - Siete")[0])
        self.assertIn("aceptó", resto.split("Banda - Siete (anuncios limitados)")[0])
        self.assertNotIn("No pude apelar", texto)
        self.assertIn("No pude ver", texto.split("Banda - Ocho")[0])

    def test_tras_apelar_sigue_las_impugnaciones_y_avisa_lo_que_cambio(self):
        # Defecto: el seguimiento de las impugnaciones no se lanzaba, o lo que cambió no llegaba.
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("seguir_impugnaciones")), 1)
        en_revision = "Impugnación en proceso de revisión. Caduca dentro de 20 días."
        self.lanzador.terminar(
            "seguir_impugnaciones",
            {
                "ok": False,
                "log": "/logs/seguir.log",
                "cambios": [
                    {"video_id": "V1", "titulo": "Banda - Uno", "reclamo": "Canción B — Artista",
                     "antes": en_revision, "ahora": "El reclamante ha rechazado tu impugnación."},
                    {"video_id": "V1", "titulo": "Banda - Uno", "reclamo": "Canción C — Artista",
                     "antes": en_revision, "ahora": None},
                ],
                "fallos": [{"video_id": "V2", "titulo": "Banda - Dos", "motivo": "lista_de_reclamos_no_disponible"}],
            },
        )
        self.servicio.paso()
        texto = self.telegram.mensajes[-1]["texto"]
        uno, dos = texto.split("Banda - Dos")
        self.assertIn("Banda - Uno", uno)
        self.assertIn("El reclamante ha rechazado tu impugnación.", uno.split("Canción B — Artista")[1])
        self.assertIn("ya no está", uno.split("Canción C — Artista")[1])
        self.assertIn("https://studio.youtube.com/video/V1/claims", uno)
        self.assertIn("https://studio.youtube.com/video/V2/claims", dos)
        self.assertIn("/logs/seguir.log", texto)

    def test_un_seguimiento_sin_cambios_no_manda_ningun_mensaje(self):
        # Defecto: un mensaje diario vacío sobre las impugnaciones.
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        antes = len(self.telegram.mensajes)
        self.lanzador.terminar("seguir_impugnaciones", {"ok": True, "cambios": [], "fallos": []})
        self.servicio.paso()
        self.assertEqual(len(self.telegram.mensajes), antes)

    def terminar_seguimiento(self, **resultado):
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        self.lanzador.terminar("seguir_impugnaciones", {"ok": True, "cambios": [], "fallos": [], **resultado})
        self.servicio.paso()

    def test_un_video_liberado_se_vuelve_a_programar_sin_reclamo_y_sin_preguntar(self):
        # Defecto: un video que quedó sin reclamos sigue a dos años y sin ingresos, o se relanza
        # con la espera de una hora y otra revisión, o pide un botón que nadie pidió.
        self.terminar_seguimiento(liberados=[{"video_id": "V1", "titulo": "Banda - Uno"},
                                             {"video_id": "V2", "titulo": "Banda - Dos"}])
        self.assertEqual(self.lanzador.lanzados("programar"), [
            ["programar", "--video-id", "V1", "--reclamo", "no"],
            ["programar", "--video-id", "V2", "--reclamo", "no"],
        ])
        self.assertEqual([m for m in self.telegram.mensajes if m["botones"]], [])

    def test_una_impugnacion_rechazada_pregunta_y_solo_borra_con_eliminar(self):
        # Defecto: un rechazo sin pregunta o sin decir qué video, qué reclamo ni dónde mirarlo;
        # "Conservar" que borra; o un "Eliminar" sin la comprobación de siempre (--retirado).
        estado = "El reclamante ha rechazado tu impugnación."
        self.terminar_seguimiento(rechazados=[
            {"video_id": video_id, "titulo": titulo, "captura": str(self.tmp / f"{video_id}.png"),
             "reclamos": [{"reclamo": "Canción A — Artista", "estado": estado}]}
            for video_id, titulo in (("V1", "Banda - Uno"), ("V2", "Banda - Dos"))
        ])
        uno, dos = [m for m in self.telegram.mensajes if m["botones"]]
        self.assertIn("Banda - Uno", uno["texto"])
        self.assertIn(estado, uno["texto"].split("Canción A — Artista")[1])
        self.assertIn("https://studio.youtube.com/video/V1/claims", uno["texto"])
        self.assertEqual(uno["foto"], str(self.tmp / "V1.png"))
        self.assertEqual([etiqueta for etiqueta, _ in uno["botones"]], ["Conservar", "🗑️ Eliminar video"])
        self.assertEqual(self.lanzador.lanzados("programar"), [])

        self.telegram.apretar(uno, "Conservar")
        self.telegram.apretar(dos, "🗑️ Eliminar video")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("eliminar_video"), [["eliminar_video", "--video-id", "V2"]])

    def test_sin_nada_que_apelar_no_manda_ningun_mensaje(self):
        # Defecto: un mensaje diario vacío.
        self.ahora = self.DIA
        self.servicio.paso()
        antes = len(self.telegram.mensajes)
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        self.assertEqual(len(self.telegram.mensajes), antes)

    def test_sin_sesion_en_studio_no_lanza_el_seguimiento_y_avisa_el_problema_una_vez(self):
        # Defecto: `seguir_impugnaciones` se lanzaba tras cada `apelar`, también cuando `apelar`
        # no tenía sesión: el segundo paso falla igual y el mismo problema llegaba dos veces.
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar(
            "apelar",
            {"ok": False, "motivo": "sin_sesion_en_studio", "videos": [], "fallos": [], "log": "/logs/apelar.log"},
        )
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("seguir_impugnaciones"), [])
        (aviso,) = [m for m in self.telegram.mensajes if sp.MOTIVOS_APELACION["sin_sesion_en_studio"] in m["texto"]]
        self.assertIn("Sigo mañana", aviso["texto"])


class TestAvisosPendientes(BaseServicio):
    """`apelar` y `seguir_impugnaciones` anotan su resultado antes de que el servicio lo avise:
    si Telegram rechaza el mensaje, el aviso tiene que llegar en un paso posterior."""

    DIA = sp.INTERVALO_APELACIONES_S
    APELACION = {
        "ok": True, "fallos": [],
        "videos": [{"tipo": "edad", "video_id": "ID1", "titulo": "Banda - Uno", "estado": "enviado"}],
    }
    SEGUIMIENTO = {
        "ok": True, "fallos": [],
        "cambios": [{"video_id": "V1", "titulo": "Banda - Dos", "reclamo": "Canción B — Artista",
                     "antes": "En revisión", "ahora": "El reclamante ha rechazado tu impugnación."}],
    }

    def telegram_que_rechaza(self, rechaza):
        """`mandar` lanza ErrorTelegram mientras `rechaza(texto)` sea verdadero."""
        def mandar(texto, botones=None, foto=None):
            if rechaza(texto):
                raise sp.ErrorTelegram("sendMessage: Bad Request")
            return TelegramFalso.mandar(self.telegram, texto, botones=botones, foto=foto)

        self.telegram.mandar = mandar

    def telegram_que_rechaza_una_vez(self, fragmento):
        """El primer mensaje que trae `fragmento` es rechazado; los demás pasan. Devuelve los
        rechazos que faltan: vacío cuando ya rechazó ese mensaje."""
        rechazos = [True]
        self.telegram_que_rechaza(lambda texto: fragmento in texto and bool(rechazos) and rechazos.pop())
        return rechazos

    def recibidos(self, fragmento):
        return [m["texto"] for m in self.telegram.mensajes if fragmento in m["texto"]]

    def test_el_aviso_de_la_apelacion_que_telegram_rechaza_llega_en_el_paso_siguiente_una_sola_vez(self):
        # Defecto: `avisar` traga el error de Telegram y el resultado ya quedó anotado en
        # apelaciones.json: el aviso no salía nunca.
        rechazos = self.telegram_que_rechaza_una_vez("Banda - Uno")
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", self.APELACION)
        for _ in range(3):
            self.servicio.paso()
        self.assertEqual(rechazos, [])  # el primer envío sí fue rechazado
        self.assertEqual(len(self.recibidos("Banda - Uno")), 1)
        self.assertEqual(self.servicio.estado["avisos_pendientes"], [])

    def test_el_aviso_del_seguimiento_que_telegram_rechaza_llega_en_el_paso_siguiente_una_sola_vez(self):
        # Defecto: el mismo, con el resultado ya anotado en impugnaciones.json.
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", {"ok": True, "videos": [], "fallos": []})
        self.servicio.paso()
        rechazos = self.telegram_que_rechaza_una_vez("Canción B")
        self.lanzador.terminar("seguir_impugnaciones", self.SEGUIMIENTO)
        for _ in range(3):
            self.servicio.paso()
        self.assertEqual(rechazos, [])  # el primer envío sí fue rechazado
        self.assertEqual(len(self.recibidos("Canción B")), 1)

    def test_el_aviso_que_no_salio_sobrevive_a_un_reinicio(self):
        # Defecto: la cola solo en memoria: reiniciar el servicio con el aviso sin enviar lo perdía.
        self.telegram_que_rechaza(lambda texto: "Banda - Uno" in texto)
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", self.APELACION)
        self.servicio.paso()
        self.telegram_que_rechaza(lambda texto: False)
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()
        self.assertEqual(len(self.recibidos("Banda - Uno")), 1)

    def test_con_telegram_caido_el_aviso_espera_sin_gastar_intentos(self):
        # Defecto: contar como intento cada paso con Telegram sin responder; una caída larga
        # descartaba el aviso, que es justo cuando más hace falta que llegue después.
        self.telegram.falla = sp.ErrorTelegram("sin red")
        self.telegram_que_rechaza(lambda texto: True)
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", self.APELACION)
        for _ in range(sp.INTENTOS_AVISO + 3):
            self.servicio.paso()
        self.telegram.falla = None
        self.telegram_que_rechaza(lambda texto: False)
        self.servicio.paso()
        self.assertEqual(len(self.recibidos("Banda - Uno")), 1)

    def test_un_aviso_que_telegram_rechaza_siempre_se_descarta_y_no_bloquea_los_demas(self):
        # Defecto: sin tope de intentos, un texto que Telegram nunca acepta quedaba primero de la
        # cola para siempre y los avisos de después no salían.
        self.telegram_que_rechaza(lambda texto: "Banda - Uno" in texto)
        self.ahora = self.DIA
        self.servicio.paso()
        self.lanzador.terminar("apelar", self.APELACION)
        self.servicio.paso()
        self.lanzador.terminar("seguir_impugnaciones", self.SEGUIMIENTO)
        for _ in range(sp.INTENTOS_AVISO + 1):
            self.servicio.paso()
        self.assertEqual(len(self.recibidos("Canción B")), 1)
        self.assertEqual(self.recibidos("Banda - Uno"), [])
        self.assertEqual(self.servicio.estado["avisos_pendientes"], [])


class TestProgramarFallido(BaseServicio):
    ARGUMENTOS = ["programar", "--video-id", "VID1", "--esperar"]
    REINTENTO = "⚠️ No pude programar VID1 (error_api). Reintento en 30 min."

    def setUp(self):
        super().setUp()
        self.ahora = 0.0
        self.servicio.reloj = lambda: self.ahora

    def fallar(self, motivo="error_api"):
        """Lanza `programar` si no hay uno corriendo y lo termina con un motivo no previsto."""
        if not any(p.salida is None for p in self.lanzador.procesos if p.argumentos[0] == "programar"):
            self.servicio.lanzar_paso(*self.ARGUMENTOS)
        self.lanzador.terminar("programar", {"ok": False, "motivo": motivo, "video_id": "VID1"})
        self.servicio.paso()

    def esperar_reintento(self):
        self.ahora += sp.REINTENTO_S
        self.servicio.paso()

    def textos(self):
        return [m["texto"] for m in self.telegram.mensajes]

    def preguntas(self):
        return [m for m in self.telegram.mensajes if m["botones"]]

    def fallar_tres_veces(self):
        self.fallar()
        self.esperar_reintento()
        self.fallar()
        self.esperar_reintento()
        self.fallar()

    def test_cortes_por_tiempo_y_conexiones_caidas_cuentan_como_fallos(self):
        # Un corte por tiempo máximo, una conexión caída y un proceso que muere
        # sin respuesta son el mismo video sin programar: los tres suman.
        self.fallar("tiempo_agotado")
        self.esperar_reintento()
        self.fallar("sin_conexion")
        self.esperar_reintento()
        self.fallar("sin_respuesta")

        self.assertEqual(len(self.preguntas()), 1)
        self.assertIn("después de 3 intentos", self.preguntas()[0]["texto"])

    def test_tercer_fallo_pregunta_en_vez_de_reintentar_otra_vez(self):
        self.fallar()
        self.assertEqual(self.textos().count(self.REINTENTO), 1)
        self.assertEqual(len(self.lanzador.lanzados("programar")), 1)
        self.esperar_reintento()
        self.assertEqual(self.lanzador.lanzados("programar"), [self.ARGUMENTOS, self.ARGUMENTOS])
        self.fallar()
        self.assertEqual(self.textos().count(self.REINTENTO), 2)
        self.esperar_reintento()
        self.fallar()

        # El tercer fallo ya no avisa "Reintento en 30 min": pregunta una sola vez.
        self.assertEqual(self.textos().count(self.REINTENTO), 2)
        pregunta = self.ultima_pregunta()
        self.assertEqual(len(self.preguntas()), 1)
        self.assertIn("VID1", pregunta["texto"])
        self.assertIn("error_api", pregunta["texto"])
        self.assertIn("3 intentos", pregunta["texto"])
        self.assertIn("🔁 Reintentar", dict(pregunta["botones"]))
        # Y no vuelve a lanzar `programar` por su cuenta, por mucho que pase el tiempo.
        lanzados = len(self.lanzador.lanzados("programar"))
        self.ahora += 10 * sp.REINTENTO_S
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("programar")), lanzados)

    def test_boton_reintentar_relanza_programar_y_reinicia_la_cuenta(self):
        self.fallar_tres_veces()
        self.telegram.apretar(self.ultima_pregunta(), "🔁 Reintentar")
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("programar")[-1], self.ARGUMENTOS)
        self.assertEqual(len(self.lanzador.lanzados("programar")), 4)
        # La cuenta arrancó de cero: el próximo fallo es el primero, no vuelve a preguntar.
        self.fallar()
        self.assertEqual(self.textos().count(self.REINTENTO), 3)
        self.assertEqual(len(self.preguntas()), 1)

    def test_reintentar_repite_el_programar_que_fallo_con_la_respuesta_del_reclamo(self):
        # Si ya contestaste si tenía reclamo, el botón no vuelve a revisar Studio:
        # relanza el mismo programar que falló, con su --reclamo.
        self.ARGUMENTOS = ["programar", "--video-id", "VID1", "--reclamo", "si"]
        self.fallar_tres_veces()
        self.telegram.apretar(self.ultima_pregunta(), "🔁 Reintentar")
        self.servicio.paso()

        self.assertEqual(
            self.lanzador.lanzados("programar")[-1],
            ["programar", "--video-id", "VID1", "--reclamo", "si"],
        )

    def test_programar_exitoso_reinicia_la_cuenta(self):
        self.fallar()
        self.esperar_reintento()
        self.fallar()
        self.esperar_reintento()
        self.lanzador.terminar("programar", {"ok": True, "video_id": "VID1", "resumen": "📅 T se publica"})
        self.servicio.paso()

        # Dos fallos más de un video que ya se programó una vez no suman al tercero.
        self.fallar()
        self.esperar_reintento()
        self.fallar()

        self.assertEqual(self.preguntas(), [])

    def test_pregunta_abierta_no_frena_el_render_de_otros_albumes(self):
        self.fallar_tres_veces()
        self.assertEqual(len(self.preguntas()), 1)
        self.album_en_cola("Banda - Disco")
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])

    def test_reinicio_con_la_pregunta_abierta_no_relanza_programar(self):
        pa.guardar_pendientes([{"video_id": "VID1", "titulo": "T"}])
        self.fallar_tres_veces()
        pregunta = self.ultima_pregunta()

        self.lanzador = LanzadorFalso()
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("programar"), [])
        self.assertEqual(len(self.preguntas()), 1)
        # El botón del mensaje de antes del reinicio sigue valiendo.
        self.telegram.apretar(pregunta, "🔁 Reintentar")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("programar"), [self.ARGUMENTOS])


class TestReinicio(BaseServicio):
    def test_reinicio_retoma_la_pregunta_y_los_videos_sin_programar(self):
        carpeta, mensaje = self.pedir_aprobacion()
        pa.guardar_pendientes([{"video_id": "VID0", "titulo": "Viejo"}])

        self.lanzador = LanzadorFalso()
        self.servicio = self.nuevo_servicio()
        self.servicio.paso()
        self.telegram.apretar(mensaje, "✅ Subir")
        self.servicio.paso()

        self.assertEqual(len(self.telegram.mensajes), 2)
        self.assertEqual(self.lanzador.lanzados("verificar"), [])
        self.assertIn(["programar", "--video-id", "VID0", "--esperar"], self.lanzador.lanzados())
        self.assertIn(["subir", "--carpeta", carpeta], self.lanzador.lanzados())

    def test_album_que_dejo_otro_render_se_pregunta_sin_renderizar(self):
        carpeta = self.album_renderizado("Banda - Ya renderizado")
        self.album_en_cola("Banda - En cola")

        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados(), [["verificar", "--carpeta", carpeta]])

    def test_render_ajeno_en_curso_espera_sin_lanzar_otro(self):
        # El mp4 puede estar copiándose todavía: no se verifica hasta que termine.
        self.album_renderizado("Banda - Copiándose")
        self.album_en_cola("Banda - En cola")
        with mock.patch.object(pa, "render_en_curso", return_value=True):
            self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados(), [])


class TestTiempoMaximo(BaseServicio):
    """Un paso colgado no puede frenar el servicio para siempre.

    Los pasos son procesos reales (un guion de Python que duerme) y el reloj
    del servicio es falso, así nadie espera 30 minutos de verdad.
    """

    def setUp(self):
        super().setUp()
        parche = mock.patch.object(pa, "DIR_LOGS", self.tmp / "logs")
        parche.start()
        self.addCleanup(parche.stop)
        self.ahora = 1000.0
        self.servicio.reloj = lambda: self.ahora
        self.servicio.lanzar = self.lanzar_real
        self.guion = "import time; time.sleep(600)"

    def lanzar_real(self, argumentos):
        proceso = sp.Proceso(argumentos, comando=[sys.executable, "-c", self.guion])
        self.addCleanup(self.matar, proceso)
        return proceso

    @staticmethod
    def matar(proceso):
        try:
            os.killpg(proceso.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proceso.popen.kill()
        proceso.popen.wait()

    def primera_linea(self, proceso):
        """Espera a que el guion avise que ya arrancó (instaló su señal, lanzó su hijo)."""
        limite = time.monotonic() + 10
        while time.monotonic() < limite:
            texto = proceso.salida.read_text(encoding="utf-8")
            if texto.strip():
                return texto.splitlines()[0]
            time.sleep(0.05)
        self.fail("el paso no escribió nada")

    @staticmethod
    def esta_vivo(pid):
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as archivo:
                return archivo.read().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    def correr(self, segundos):
        """Deja pasar `segundos` desde el lanzamiento y revisa las tareas."""
        self.ahora = 1000.0 + segundos
        self.servicio.revisar_tareas()

    def paso_lanzado(self, *argumentos):
        self.servicio.lanzar_paso(*argumentos)
        return self.servicio.tareas[-1][1]

    def test_verificar_colgado_se_corta_y_avisa_sin_ofrecer_subir(self):
        carpeta = self.album_renderizado("Banda - Disco")
        proceso = self.paso_lanzado("verificar", "--carpeta", carpeta)
        limite = sp.limite_del_paso("verificar")

        self.correr(limite - 1)
        self.assertIsNone(proceso.popen.poll())
        self.assertEqual(self.telegram.mensajes, [])

        self.correr(limite + 1)
        self.assertIsNotNone(proceso.popen.poll())
        self.assertEqual(self.servicio.tareas, [])
        self.assertEqual(len(self.telegram.mensajes), 1)
        aviso = self.telegram.mensajes[0]
        self.assertEqual(aviso["botones"], [])  # no "¿Lo subo?" de un álbum sin verificar
        self.assertIn("Banda - Disco", aviso["texto"])
        self.assertIn(str(proceso.salida), aviso["texto"])

    def test_buscar_colgado_se_corta_avisa_y_se_reintenta_al_pasar_la_pausa(self):
        self.album_en_cola("Banda - Disco")
        proceso = self.paso_lanzado("buscar")

        self.correr(sp.limite_del_paso("buscar") + 1)
        self.assertIsNotNone(proceso.popen.poll())
        self.assertEqual(self.servicio.tareas, [])
        aviso = self.telegram.mensajes[-1]
        self.assertEqual(aviso["botones"], [])
        self.assertIn("No pude buscar el próximo álbum", aviso["texto"])

        self.servicio.lanzar = self.lanzador
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [])
        self.ahora += sp.REINTENTO_S + 1
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])

    def test_verificar_cortado_se_reintenta_al_pasar_la_pausa(self):
        carpeta = self.album_renderizado("Banda - Disco")
        self.servicio.lanzar = self.lanzador
        argumentos = ("verificar", "--carpeta", carpeta)
        self.servicio.tras_verificar(
            {"ok": False, "motivo": "tiempo_agotado", "log": "paso.out"}, argumentos
        )

        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("verificar"), [])

        self.ahora += sp.REINTENTO_S + 1
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("verificar"), [list(argumentos)])

    def test_programar_aguanta_mas_que_un_paso_corto_y_se_corta_al_pasar_el_suyo(self):
        proceso = self.paso_lanzado("programar", "--video-id", "VID1", "--esperar")

        self.correr(sp.limite_del_paso("verificar") + 1)
        self.assertIsNone(proceso.popen.poll())
        self.assertEqual(self.telegram.mensajes, [])

        self.correr(sp.limite_del_paso("programar") + 1)
        self.assertIsNotNone(proceso.popen.poll())
        self.assertIn("VID1", self.telegram.mensajes[0]["texto"])

        self.servicio.lanzar = self.lanzador
        self.ahora += sp.REINTENTO_S + 1
        self.servicio.revisar_reintentos()
        self.assertEqual(
            self.lanzador.lanzados("programar"), [["programar", "--video-id", "VID1", "--esperar"]]
        )

    def test_cortar_mata_tambien_a_los_hijos_del_paso(self):
        self.guion = (
            "import subprocess, sys, time\n"
            "hijo = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "print(hijo.pid, flush=True)\n"
            "time.sleep(600)\n"
        )
        proceso = self.lanzar_real(("verificar", "--carpeta", "x"))
        pid_hijo = int(self.primera_linea(proceso))
        self.assertTrue(self.esta_vivo(pid_hijo))

        proceso.cortar()

        limite = time.monotonic() + 5
        while self.esta_vivo(pid_hijo) and time.monotonic() < limite:
            time.sleep(0.05)
        self.assertFalse(self.esta_vivo(pid_hijo))

    def test_paso_que_ignora_sigterm_muere_con_sigkill(self):
        self.guion = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('listo', flush=True)\n"
            "time.sleep(600)\n"
        )
        proceso = self.lanzar_real(("verificar", "--carpeta", "x"))
        self.primera_linea(proceso)

        with mock.patch.object(sp, "GRACIA_CORTE_S", 0.3):
            proceso.cortar()

        self.assertEqual(proceso.popen.returncode, -signal.SIGKILL)

    def comprobar_sin_cortar(self, argumentos, limite):
        proceso = self.paso_lanzado(*argumentos)

        self.correr(limite - 1)
        self.assertEqual(self.telegram.mensajes, [])

        self.correr(limite + 60)
        self.correr(limite + 3600)

        self.assertIsNone(proceso.popen.poll())
        self.assertEqual(len(self.servicio.tareas), 1)
        self.assertEqual(len(self.telegram.mensajes), 1)  # un solo aviso por corrida
        texto = self.telegram.mensajes[0]["texto"]
        self.assertIn(" ".join(argumentos), texto)
        self.assertIn(sp.duracion_legible(limite + 60), texto)
        self.assertIn(str(proceso.salida), texto)

    def test_render_que_pasa_su_limite_no_se_corta_y_avisa_una_vez(self):
        # Mismo tiempo que ya espera pipeline_agente.py por un render.
        self.comprobar_sin_cortar(("renderizar",), pa.ESPERA_RENDER_MAX_S)

    def test_subida_que_pasa_su_limite_no_se_corta_y_avisa_una_vez(self):
        carpeta = self.album_renderizado("Banda - Disco")
        self.comprobar_sin_cortar(("subir", "--carpeta", carpeta), sp.limite_del_paso("subir"))

    def test_borrado_que_pasa_su_limite_no_se_corta_y_avisa_una_vez(self):
        carpeta = self.album_renderizado("Banda - Disco")
        self.comprobar_sin_cortar(("eliminar", "--carpeta", carpeta), sp.limite_del_paso("eliminar"))

    def test_duracion_legible_en_horas_y_minutos(self):
        self.assertEqual(sp.duracion_legible(45 * 60), "45 min")
        self.assertEqual(sp.duracion_legible(6 * 3600), "6 h")
        self.assertEqual(sp.duracion_legible(6 * 3600 + 90), "6 h 1 min")


class TestAvisos(BaseServicio):
    def test_con_la_cola_vacia_busca_desde_album_o_ep(self):
        album = self.tmp / "Album"
        (album / "Banda - Disco (2020)").mkdir(parents=True)
        with mock.patch.object(pa, "DIRS_ORIGEN_RENDER", (self.cola, album)):
            self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])

    def test_sin_albumes_avisa_una_sola_vez(self):
        origenes = tuple(self.tmp / nombre for nombre in ("01_limpieza_de_impurezas", "Album", "EP"))
        with mock.patch.object(pa, "DIRS_ORIGEN_RENDER", origenes):
            for _ in range(3):
                self.servicio.paso()

        self.assertEqual(len(self.telegram.mensajes), 1)
        self.assertIn(
            "No quedan álbumes en 01_limpieza_de_impurezas/, Album/ ni EP/.",
            self.telegram.mensajes[0]["texto"],
        )
        self.assertEqual(self.lanzador.lanzados(), [])

    def test_render_fallido_sin_error404_no_manda_a_una_cola_que_no_existe(self):
        self.aceptar("Banda - Disco")
        self.lanzador.terminar(
            "renderizar",
            {"ok": False, "motivo": "render_fallido", "album": "Banda - Disco",
             "movido_a_error404": False, "log_final": ""},
        )
        self.servicio.paso()

        pregunta = self.ultima_pregunta()
        self.assertIn("Falló el render de Banda - Disco", pregunta["texto"])
        self.assertIn("Sigue en su carpeta", pregunta["texto"])
        self.assertNotIn("la cola", pregunta["texto"])
        self.assertIn("🔁 Reintentar", dict(pregunta["botones"]))

    def test_avisa_los_albumes_que_buscar_aparto_por_ya_publicados(self):
        # Defecto: buscar mueve álbumes a Ya publicados/ en silencio, y una reedición
        # legítima apartada por error no se ve nunca.
        apartados = [
            {"album": f"Banda {n} - Disco (2020)", "origen": "canal",
             "como": f"🇺🇸 BANDA {n} - DISCO (2020) ⟨FULL ALBUM⟩", "destino": "x"}
            for n in range(12)
        ] + [{"album": "Banda - Roto (2020)", "origen": "eliminar", "como": "Banda - Roto (2020)",
              "error": "sin permiso"}]
        _, pregunta = self.preguntar_antes_de_renderizar("Banda - Disco", apartados=apartados)

        movidos, no_movido, _busqueda, _pregunta = (m["texto"] for m in self.telegram.mensajes)
        self.assertIn("Ya publicados", movidos)
        self.assertIn("Banda 0 - Disco (2020) → 🇺🇸 BANDA 0 - DISCO (2020) ⟨FULL ALBUM⟩", movidos)
        self.assertIn("Banda 9 - Disco (2020)", movidos)
        self.assertNotIn("Banda 10 - Disco (2020)", movidos)
        self.assertIn("y 2 más", movidos)
        self.assertNotIn("Banda - Roto", movidos)
        self.assertIn("Banda - Roto (2020): sin permiso", no_movido)
        self.assertIn("Banda - Disco", pregunta["texto"])

    def test_sin_albumes_que_preguntar_igual_avisa_lo_que_aparto(self):
        # Defecto: el aviso sale solo cuando hay un álbum que preguntar, y cuando todo lo que
        # quedaba ya estaba publicado no se entera nadie.
        self.album_en_cola("Banda - Disco (2020)")
        self.servicio.paso()
        self.lanzador.terminar(
            "buscar",
            {"ok": False, "motivo": "sin_albumes", "ya_publicados": [
                {"album": "Banda - Disco (2020)", "origen": "subir", "como": "Banda - Disco (2020)",
                 "destino": "x"},
            ]},
        )
        self.servicio.paso()

        self.assertEqual(len(self.telegram.mensajes), 1)
        self.assertIn("Banda - Disco (2020) → Banda - Disco (2020)", self.telegram.mensajes[0]["texto"])

    def test_un_album_que_no_se_pudo_apartar_no_relanza_la_busqueda_en_cada_vuelta(self):
        # Defecto: el álbum sigue en la cola, la búsqueda se relanza en cada vuelta del servicio
        # (recorriendo miles de carpetas del disco) y el mismo aviso llega una y otra vez.
        self.album_en_cola("Banda - Roto (2020)")
        self.servicio.paso()
        self.lanzador.terminar(
            "buscar",
            {"ok": False, "motivo": "sin_albumes", "ya_publicados": [
                {"album": "Banda - Roto (2020)", "origen": "canal", "como": "Banda - Roto (2020)",
                 "error": "sin permiso"},
            ]},
        )
        for _ in range(3):
            self.servicio.paso()

        self.assertEqual(len(self.lanzador.lanzados("buscar")), 1)
        self.assertEqual(len(self.telegram.mensajes), 1)
        self.assertIn("Banda - Roto (2020): sin permiso", self.telegram.mensajes[0]["texto"])

    def test_un_album_que_no_se_pudo_apartar_se_avisa_una_sola_vez(self):
        # Defecto: el álbum sigue al frente de la cola, cada búsqueda vuelve a intentar moverlo y
        # el mismo aviso llega después de cada una.
        no_movido = {"album": "Banda - Roto (2020)", "origen": "canal", "como": "Banda - Roto (2020)",
                     "error": "sin permiso"}
        for nombre in ("Banda - Uno", "Banda - Dos"):
            self.servicio.tras_buscar(
                {"ok": True, "album": nombre, "carpeta": str(self.cola / nombre), "ya_publicados": [no_movido]},
                ["buscar"],
            )

        avisos = [m for m in self.telegram.mensajes if "sin permiso" in m["texto"]]
        self.assertEqual(len(avisos), 1)

    def test_error_de_red_en_telegram_no_frena_el_servicio(self):
        self.album_en_cola("Banda - Disco")
        self.telegram.falla = sp.ErrorTelegram("sin red")

        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("buscar"), [["buscar"]])


class TestTelegramCaido(BaseServicio):
    """Con Telegram sin responder, la espera crece y el log avisa una sola vez."""

    def setUp(self):
        super().setUp()
        self.servicio.pausa_error_s = 5
        self.tiempo = [1000.0]
        self.servicio.reloj = lambda: self.tiempo[0]
        parche = mock.patch.object(sp.time, "sleep")
        self.dormir = parche.start()
        self.addCleanup(parche.stop)
        self.telegram.falla = sp.ErrorTelegram("sin red")

    def esperas(self):
        return [llamada.args[0] for llamada in self.dormir.call_args_list]

    def avisos_en(self, capturado, nivel):
        return [r.getMessage() for r in capturado.records if r.levelno == nivel]

    def test_la_espera_se_duplica_hasta_un_minuto(self):
        for _ in range(7):
            self.servicio.atender_telegram()

        self.assertEqual(self.esperas(), [5, 10, 20, 40, 60, 60, 60])

    def test_al_responder_la_espera_vuelve_a_la_inicial(self):
        for _ in range(3):
            self.servicio.atender_telegram()
        self.telegram.falla = None
        self.servicio.atender_telegram()
        self.telegram.falla = sp.ErrorTelegram("sin red")
        self.servicio.atender_telegram()

        self.assertEqual(self.esperas(), [5, 10, 20, 5])

    def test_avisa_una_sola_vez_cuando_telegram_deja_de_responder(self):
        with self.assertLogs("servicio_publicar", "INFO") as capturado:
            for _ in range(5):
                self.servicio.atender_telegram()

        avisos = self.avisos_en(capturado, logging.WARNING)
        self.assertEqual(len(avisos), 1)
        self.assertIn("sin red", avisos[0])

    def test_avisa_una_sola_vez_cuando_telegram_vuelve(self):
        with self.assertLogs("servicio_publicar", "INFO") as capturado:
            for _ in range(3):
                self.servicio.atender_telegram()
                self.tiempo[0] += 10
            self.telegram.falla = None
            self.servicio.atender_telegram()
            self.servicio.atender_telegram()

        avisos = self.avisos_en(capturado, logging.INFO)
        self.assertEqual(len(avisos), 1)
        self.assertIn("3 fallos", avisos[0])
        self.assertIn("30 s", avisos[0])

    def test_un_segundo_corte_vuelve_a_avisar(self):
        with self.assertLogs("servicio_publicar", "INFO") as capturado:
            self.servicio.atender_telegram()
            self.telegram.falla = None
            self.servicio.atender_telegram()
            self.telegram.falla = sp.ErrorTelegram("otra vez sin red")
            self.servicio.atender_telegram()

        avisos = self.avisos_en(capturado, logging.WARNING)
        self.assertEqual(len(avisos), 2)
        self.assertIn("otra vez sin red", avisos[1])

    def test_con_telegram_caido_la_pregunta_pendiente_no_repite_el_aviso(self):
        caida = sp.ErrorTelegram("sin red")
        with mock.patch.object(self.telegram, "mandar", side_effect=caida):
            self.servicio.preguntar("reintentar", "¿Reintento?")
            self.servicio.atender_telegram()
            with self.assertNoLogs("servicio_publicar", "WARNING"):
                for _ in range(3):
                    self.servicio.enviar_preguntas()

        # La pregunta no se perdió: sale en cuanto Telegram vuelve.
        self.telegram.falla = None
        self.servicio.atender_telegram()
        self.servicio.enviar_preguntas()
        self.assertEqual([m["texto"] for m in self.telegram.mensajes], ["¿Reintento?"])

    def test_con_telegram_respondiendo_la_pregunta_que_falla_sigue_avisando(self):
        self.telegram.falla = None
        with mock.patch.object(self.telegram, "mandar", side_effect=sp.ErrorTelegram("PHOTO_INVALID")):
            self.servicio.preguntar("reintentar", "¿Reintento?")
            self.servicio.atender_telegram()
            with self.assertLogs("servicio_publicar", "WARNING") as capturado:
                self.servicio.enviar_preguntas()

        self.assertIn("PHOTO_INVALID", capturado.output[0])

    def test_los_avisos_de_caida_no_dejan_el_token_en_el_log(self):
        self.servicio.telegram = sp.Telegram("123:SECRETO", "42")
        error = requests.ConnectionError("Max retries exceeded with url: /bot123:SECRETO/getUpdates")
        with mock.patch.object(sp.requests, "post", side_effect=error):
            with self.assertLogs("servicio_publicar", "INFO") as capturado:
                for _ in range(2):
                    self.servicio.atender_telegram()
        with mock.patch.object(sp.requests, "post", return_value=mock.Mock(json=lambda: {"ok": True, "result": []})):
            with self.assertLogs("servicio_publicar", "INFO") as vuelta:
                self.servicio.atender_telegram()

        self.assertNotIn("SECRETO", "\n".join(capturado.output + vuelta.output))


class TestConexiones(unittest.TestCase):
    def test_error_de_red_no_deja_el_token_en_el_log(self):
        telegram = sp.Telegram("123:SECRETO", "42")
        error = requests.ConnectionError("Max retries exceeded with url: /bot123:SECRETO/getUpdates")
        with mock.patch.object(sp.requests, "post", side_effect=error):
            with self.assertRaises(sp.ErrorTelegram) as contexto:
                telegram.actualizaciones(0, 1)
        self.assertNotIn("SECRETO", str(contexto.exception))

    def test_foto_rechazada_por_telegram_manda_la_pregunta_igual(self):
        foto = Path(tempfile.mkdtemp()) / "captura.jpg"
        foto.write_bytes(b"no es una imagen")
        self.addCleanup(shutil.rmtree, foto.parent)
        rechazo = mock.Mock(json=lambda: {"ok": False, "description": "PHOTO_INVALID_DIMENSIONS"})
        enviado = mock.Mock(json=lambda: {"ok": True, "result": {"message_id": 7}})
        telegram = sp.Telegram("123:SECRETO", "42")

        with mock.patch.object(sp.requests, "post", side_effect=[rechazo, enviado]) as post:
            message_id = telegram.mandar("¿Lo subo?", botones=[[("✅ Subir", "subir:q1")]], foto=foto)

        self.assertEqual(message_id, 7)
        self.assertTrue(post.call_args.args[0].endswith("/sendMessage"))
        self.assertIn("subir:q1", post.call_args.kwargs["data"]["reply_markup"])

    def test_proceso_devuelve_el_json_final_del_paso(self):
        proceso = sp.Proceso(["estado"])
        proceso.popen.wait(timeout=60)
        self.addCleanup(proceso.salida.unlink)

        resultado = proceso.resultado()

        self.assertTrue(resultado["ok"])
        self.assertIn("por_aprobar", resultado)

    def test_un_id_que_empieza_con_guion_llega_entero_al_paso(self):
        # YouTube da ids que empiezan con «-». Suelto después de --video-id, argparse lo toma
        # por otra opción y el paso termina con «expected one argument» sin hacer nada.
        logs = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, logs)
        casos = {
            "programar": (["programar", "--video-id", "-trLTTpeDEs", "--esperar"], ["-trLTTpeDEs"]),
            "un id normal": (["eliminar_video", "--video-id", "VID1"], ["VID1"]),
            "varios ids": (
                ["impugnar_con_ventana", "--video-id", "-xiG2-NDCGY", "--video-id", "VID2"],
                ["-xiG2-NDCGY", "VID2"],
            ),
        }
        for caso, (argumentos, esperados) in casos.items():
            with self.subTest(caso), mock.patch.object(pa, "DIR_LOGS", logs), \
                    mock.patch.object(sp.subprocess, "Popen") as popen:
                sp.Proceso(argumentos)
                comando = popen.call_args.args[0]
                self.assertEqual(comando[1], "pipeline_agente.py")
                leidos = pa.parse_args(comando[2:]).video_id
                self.assertEqual(leidos if isinstance(leidos, list) else [leidos], esperados)


class TestImpugnarConVentana(BaseServicio):
    """La ventana impugna varios videos con una sola verificación y avisa una vez."""

    def terminar_ventana(self, resultado, videos=("V1", "V2")):
        argumentos = ["impugnar_con_ventana"]
        for video in videos:
            argumentos += ["--video-id", video]
        self.servicio.lanzar_paso(*argumentos)
        self.antes = len(self.telegram.mensajes)
        self.lanzador.terminar("impugnar_con_ventana", {"ventana": True, **resultado})
        self.servicio.paso()
        return [m for m in self.telegram.mensajes[self.antes:] if "ventana" in m["texto"]]

    def test_avisa_una_vez_lo_que_impugno_y_lo_que_fallo(self):
        # Defecto: un aviso por video, o callar el fallo y el link para terminarlo a mano.
        (aviso,) = self.terminar_ventana({"ok": False, "sin_revisar": [], "videos": [
            {"ok": True, "video_id": "V1", "titulo": "Banda - V1", "reclamos": 2, "impugnados": 2, "fallos": []},
            {"ok": False, "video_id": "V2", "titulo": "Banda - V2", "reclamos": 1, "impugnados": 0,
             "fallos": [{"motivo": "no_confirmado", "detalle": "x"}]},
        ]})
        self.assertIn("\"Banda - V1\": 2 de 2", aviso["texto"])
        self.assertIn("\"Banda - V2\": 0 de 1", aviso["texto"])
        self.assertIn(sp.MOTIVOS_IMPUGNACION["no_confirmado"], aviso["texto"])
        self.assertIn("https://studio.youtube.com/video/V2/claims", aviso["texto"])
        self.assertFalse(aviso["botones"])

    def test_si_no_paso_la_verificacion_a_tiempo_ofrece_la_ventana_otra_vez(self):
        # Defecto: lo que quedó sin impugnar se perdía, o se relanzaba solo sin nadie en la PC.
        (pregunta,) = self.terminar_ventana({
            "ok": False, "motivo": "verificacion_de_identidad",
            "videos": [{"ok": False, "video_id": "V1", "titulo": "Banda - V1", "reclamos": 1, "impugnados": 0,
                        "fallos": [{"motivo": "verificacion_de_identidad"}]}],
            "sin_revisar": [{"video_id": "V2", "titulo": "Banda - V2"}],
        })
        self.assertEqual([e for e, _ in pregunta["botones"]], ["🖥 Impugnar con ventana"])
        self.assertIn("Banda - V2", pregunta["texto"])
        self.assertEqual(len(self.lanzador.lanzados("impugnar_con_ventana")), 1)
        self.telegram.apretar(pregunta, "🖥 Impugnar con ventana")
        self.servicio.paso()
        self.assertEqual(
            self.lanzador.lanzados("impugnar_con_ventana")[-1],
            ["impugnar_con_ventana", "--video-id", "V1", "--video-id", "V2"],
        )

    def ofrecer_ventana(self, *videos):
        """La pregunta «Impugnar con ventana» de esos videos."""
        self.servicio.preguntar_por_ventana(
            ["⚠️ Studio pidió verificar"], [{"video_id": v, "titulo": f"Banda - {v}"} for v in videos]
        )
        return self.ultima_pregunta()

    def apretar_ventana(self, pregunta):
        self.telegram.apretar(pregunta, "🖥 Impugnar con ventana")
        self.servicio.paso()

    def test_la_ventana_no_se_lanza_para_un_video_que_ya_esta_en_otra_ventana(self):
        # Defecto: dos ventanas para el mismo video: una corrida de más en Studio, que además
        # comparte el perfil con todo lo demás.
        primera = self.ofrecer_ventana("V1", "V2")
        segunda = self.ofrecer_ventana("V2", "V3")
        tercera = self.ofrecer_ventana("V1")
        self.apretar_ventana(primera)
        self.apretar_ventana(segunda)
        antes = len(self.telegram.mensajes)
        self.apretar_ventana(tercera)
        self.assertEqual(self.lanzador.lanzados("impugnar_con_ventana"), [
            ["impugnar_con_ventana", "--video-id", "V1", "--video-id", "V2"],
            ["impugnar_con_ventana", "--video-id", "V3"],
        ])
        (aviso,) = self.telegram.mensajes[antes:]
        self.assertIn("ya se están impugnando", aviso["texto"])

    def test_la_ventana_no_se_lanza_para_un_video_que_ya_esta_en_el_lote_de_impugnar_todos(self):
        # Defecto: la ventana y el lote para el mismo video corren uno detrás del otro con el
        # candado de Studio: una corrida de más.
        sin_impugnar = [{"video_id": v, "titulo": f"Banda - {v}", "impugnables": 1, "reclamantes": ["Sello"]}
                        for v in ("V1", "V2")]
        listas = dict.fromkeys(("liberados", "lista_negra", "en_proceso", "fallos", "sin_revisar"), [])
        self.servicio.lanzar_paso("mapear_canal")
        self.lanzador.terminar("mapear_canal", {"ok": True, "aparcados": 2, **listas, "sin_impugnar": sin_impugnar})
        self.servicio.paso()
        self.telegram.apretar(self.ultima_pregunta(), "⚖️ Impugnar todos (2)")
        self.servicio.paso()
        ventana = self.ofrecer_ventana("V1", "V3")
        self.apretar_ventana(ventana)
        self.assertEqual(
            self.lanzador.lanzados("impugnar_con_ventana"), [["impugnar_con_ventana", "--video-id", "V3"]]
        )
        self.assertEqual(self.lanzador.lanzados("impugnar"), [["impugnar", "--video-id", "V1"]])

    def test_un_reinicio_con_la_ventana_abierta_avisa_con_los_links_y_no_la_relanza(self):
        # Defecto: la ventana muere con el reinicio y nadie lo sabe: los videos quedaban sin
        # impugnar, o se relanzaba la ventana y se impugnaba dos veces lo que pudo salir.
        self.apretar_ventana(self.ofrecer_ventana("V1", "V2"))
        antes = len(self.telegram.mensajes)
        self.servicio = self.nuevo_servicio()
        (aviso,) = self.telegram.mensajes[antes:]
        self.assertIn(sp.MOTIVOS_IMPUGNACION["interrumpido"], aviso["texto"])
        for video in ("V1", "V2"):
            self.assertIn(f"Banda - {video}", aviso["texto"])
            self.assertIn(f"https://studio.youtube.com/video/{video}/claims", aviso["texto"])
        self.assertEqual(len(self.lanzador.lanzados("impugnar_con_ventana")), 1)
        # Se avisa una sola vez: otro reinicio ya no encuentra nada.
        antes = len(self.telegram.mensajes)
        self.servicio = self.nuevo_servicio()
        self.assertEqual(self.telegram.mensajes[antes:], [])

    def test_una_ventana_que_termino_no_se_avisa_como_interrumpida_al_reiniciar(self):
        # Defecto: no sacar la ventana del estado al terminar: cada reinicio avisaría un corte
        # que no pasó.
        self.apretar_ventana(self.ofrecer_ventana("V1", "V2"))
        self.lanzador.terminar("impugnar_con_ventana", {"ok": True, "sin_revisar": [], "videos": [
            {"ok": True, "video_id": v, "titulo": f"Banda - {v}", "reclamos": 1, "impugnados": 1, "fallos": []}
            for v in ("V1", "V2")
        ]})
        self.servicio.paso()
        antes = len(self.telegram.mensajes)
        self.servicio = self.nuevo_servicio()
        self.assertEqual(self.telegram.mensajes[antes:], [])

    def test_una_ventana_que_termina_sin_responder_manda_a_studio_y_no_se_ofrece_de_nuevo(self):
        # Defecto: ofrecer otra ventana cuando la primera pudo haber impugnado algo, y
        # mandarlo dos veces.
        (aviso,) = self.terminar_ventana({"ok": False, "motivo": "sin_respuesta", "codigo": -9, "log": "/x.out"})
        self.assertIn("https://studio.youtube.com/video/V1/claims", aviso["texto"])
        self.assertIn("https://studio.youtube.com/video/V2/claims", aviso["texto"])
        self.assertFalse(aviso["botones"])
        self.assertNotIn("impugnar_ventana", [p["tipo"] for p in self.servicio.estado["preguntas"].values()])


class TestMapearCanal(BaseServicio):
    """/mapear_canal relee los videos aparcados por un reclamo y junta en un mensaje lo que falta
    impugnar. Impugnar es una declaración legal: nada sale sin apretar «Impugnar todos»."""

    TODOS = "⚖️ Impugnar todos (2)"

    def mapear(self, **resultado):
        escribir_en(self.telegram, "/mapear_canal")
        self.servicio.paso()
        listas = ("liberados", "sin_impugnar", "lista_negra", "en_proceso", "fallos", "sin_revisar")
        self.antes_del_resultado = len(self.telegram.mensajes)
        self.lanzador.terminar("mapear_canal", {"ok": True, "aparcados": 0, **dict.fromkeys(listas, []), **resultado})
        self.servicio.paso()

    def preguntas(self):
        return [m for m in self.telegram.mensajes if m["botones"]]

    def etiquetas(self, mensaje):
        return [etiqueta for etiqueta, _ in mensaje["botones"]]

    def sin_impugnar(self, *ids):
        return [
            {"video_id": v, "titulo": f"Banda - {v}", "impugnables": 1,
             "reclamantes": ["Sello Independiente", "Otro Sello"] if v == "V1" else ["Sello"]}
            for v in ids
        ]

    def impugnar_todos(self):
        self.mapear(aparcados=2, sin_impugnar=self.sin_impugnar("V1", "V2"))
        self.telegram.apretar(self.ultima_pregunta(), self.TODOS)
        self.servicio.paso()

    def test_el_comando_lanza_el_mapeo_una_sola_vez_y_tambien_escrito_con_guion(self):
        # Defecto: el comando no hace nada, o dos mapeos a la vez se pisan en Studio.
        escribir_en(self.telegram, "/mapear-canal")
        self.servicio.paso()
        escribir_en(self.telegram, "/mapear_canal")
        self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("mapear_canal"), [["mapear_canal"]])
        self.assertIn("mapear_canal", [c["command"] for c in sp.COMANDOS])

    def test_lo_que_falta_impugnar_llega_en_un_solo_mensaje_con_un_boton(self):
        # Defecto: un mensaje por video (S3: "Un solo mensaje"), un resumen sin links ni
        # reclamantes, o una impugnación que sale sin apretar nada.
        self.mapear(
            ok=False,  # como lo manda el paso cuando hay fallos
            aparcados=6,
            sin_impugnar=self.sin_impugnar("V1", "V2"),
            en_proceso=[{"video_id": "V3", "titulo": "Banda - V3"}],
            fallos=[{"video_id": "V4", "titulo": "Banda - V4", "motivo": "lista_de_reclamos_no_disponible"}],
            sin_revisar=[{"video_id": "V5", "titulo": "Banda - V5"}],
        )
        (pregunta,) = self.telegram.mensajes[self.antes_del_resultado:]
        self.assertEqual(self.etiquetas(pregunta), [self.TODOS])
        for texto in ("Banda - V1", "Sello Independiente", "Otro Sello", "Banda - V2",
                      "https://studio.youtube.com/video/V1/claims", "https://studio.youtube.com/video/V2/claims",
                      "https://studio.youtube.com/video/V4/claims", "Licencia"):
            self.assertIn(texto, pregunta["texto"])
        self.assertIn("Con la impugnación en revisión: 1", pregunta["texto"])
        self.assertIn("Sin revisar (/mapear_canal sigue con ellos): 1", pregunta["texto"])
        self.assertEqual(self.lanzador.lanzados("impugnar"), [])

    def test_los_liberados_se_programan_sin_reclamo_y_la_lista_negra_pregunta_aparte(self):
        # Defecto: un video sin reclamos que sigue aparcado dos años (S2), o uno de la lista
        # negra ofrecido para impugnar o preguntado dos veces.
        self.servicio.preguntar("lista_negra", "ya abierta", video_id="V3")
        negro = {"titulo": "Banda - V2", "reclamantes_en_lista_negra": ["UMG"], "captura": str(self.tmp / "v2.png")}
        self.mapear(
            aparcados=3,
            liberados=[{"video_id": "V1", "titulo": "Banda - V1"}],
            lista_negra=[{"video_id": "V2", **negro}, {"video_id": "V3", **negro, "titulo": "Banda - V3"}],
        )
        self.assertEqual(self.lanzador.lanzados("programar"), [["programar", "--video-id", "V1", "--reclamo", "no"]])
        de_lista_negra = [p for p in self.servicio.estado["preguntas"].values() if p["tipo"] == "lista_negra"]
        self.assertEqual(sorted(p["video_id"] for p in de_lista_negra), ["V2", "V3"])
        (nueva,) = [m for m in self.preguntas() if "Banda - V2" in m["texto"]]
        self.assertEqual(self.etiquetas(nueva), ["Conservar", "🗑️ Eliminar video"])
        self.assertEqual(nueva["foto"], str(self.tmp / "v2.png"))
        self.assertEqual(self.lanzador.lanzados("impugnar"), [])

    def test_impugnar_todos_va_de_a_uno_y_avisa_una_sola_vez_al_final(self):
        # Defecto: varios `impugnar` a la vez en Studio, o un aviso por cada video.
        self.impugnar_todos()
        self.assertEqual(self.lanzador.lanzados("impugnar"), [["impugnar", "--video-id", "V1"]])
        antes = len(self.telegram.mensajes)
        self.lanzador.terminar("impugnar", {"ok": True, "video_id": "V1", "titulo": "Banda - V1", "reclamos": 2, "impugnados": 2, "fallos": []})
        self.servicio.paso()
        self.assertEqual(len(self.telegram.mensajes), antes)
        self.assertEqual(self.lanzador.lanzados("impugnar")[1:], [["impugnar", "--video-id", "V2"]])
        self.lanzador.terminar("impugnar", {"ok": False, "motivo": "sin_reclamos_impugnables", "reclamos": 0, "impugnados": 0, "fallos": []})
        self.servicio.paso()
        (aviso,) = self.telegram.mensajes[antes:]
        self.assertIn("Banda - V1", aviso["texto"])
        self.assertIn("2 de 2", aviso["texto"])
        self.assertIn("Banda - V2", aviso["texto"])
        self.assertEqual(len(self.lanzador.lanzados("impugnar")), 2)
        self.assertIsNone(self.servicio.estado["lote_impugnar"])

    def test_una_verificacion_de_identidad_frena_el_lote_y_dice_que_falta(self):
        # Defecto: seguir abriendo formularios con Studio pidiendo verificar la identidad (todos
        # fallarían), o frenar sin decir por qué ni qué quedó sin impugnar.
        self.impugnar_todos()
        antes = len(self.telegram.mensajes)
        self.lanzador.terminar("impugnar", {
            "ok": False, "video_id": "V1", "titulo": "Banda - V1", "reclamos": 1, "impugnados": 0,
            "fallos": [{"motivo": "verificacion_de_identidad"}],
        })
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("impugnar")), 1)
        (aviso,) = self.telegram.mensajes[antes:]
        self.assertIn(sp.MOTIVOS_IMPUGNACION["verificacion_de_identidad"], aviso["texto"])
        self.assertIn("Banda - V2", aviso["texto"])
        self.assertIsNone(self.servicio.estado["lote_impugnar"])

    def test_un_reinicio_no_relanza_el_video_en_curso_y_sigue_con_el_resto(self):
        # Defecto: un reinicio a mitad del lote lo dejaba sin terminar, o relanzaba el video en
        # curso, que pudo haber enviado un reclamo que Studio todavía no muestra (dos veces).
        self.impugnar_todos()
        self.servicio = self.nuevo_servicio()
        self.assertEqual([a[2] for a in self.lanzador.lanzados("impugnar")], ["V1", "V2"])
        antes = len(self.telegram.mensajes)
        self.lanzador.terminar("impugnar", {"ok": True, "video_id": "V1", "reclamos": 1, "impugnados": 1, "fallos": []})
        self.lanzador.terminar("impugnar", {"ok": True, "video_id": "V2", "reclamos": 1, "impugnados": 1, "fallos": []})
        self.servicio.paso()
        (aviso,) = [m for m in self.telegram.mensajes[antes:] if "«Impugnar todos»" in m["texto"]]
        self.assertIn(sp.MOTIVOS_IMPUGNACION["interrumpido"], aviso["texto"])
        self.assertIn("1 de 1", aviso["texto"])
        self.assertIsNone(self.servicio.estado["lote_impugnar"])

    def test_sin_sesion_en_studio_tambien_frena_el_lote(self):
        # Defecto: el fallo de todo el paso (sin "fallos" por reclamo) no frenaba el lote, y
        # cada video siguiente fallaba igual.
        self.impugnar_todos()
        self.lanzador.terminar("impugnar", {"ok": False, "motivo": "sin_sesion_en_studio", "fallos": []})
        self.servicio.paso()
        self.assertEqual(len(self.lanzador.lanzados("impugnar")), 1)
        self.assertIn(sp.MOTIVOS_IMPUGNACION["sin_sesion_en_studio"], self.telegram.mensajes[-1]["texto"])
        self.assertIn("Banda - V2", self.telegram.mensajes[-1]["texto"])
        self.assertFalse(self.telegram.mensajes[-1]["botones"])  # una ventana no arregla la sesión

    def test_el_lote_frenado_por_la_verificacion_ofrece_la_ventana_para_lo_que_quedo(self):
        # Defecto: el lote frenado solo decía que corriera /mapear_canal de nuevo, que chocaría
        # con la misma verificación.
        self.impugnar_todos()
        self.lanzador.terminar("impugnar", {
            "ok": False, "video_id": "V1", "titulo": "Banda - V1", "reclamos": 1, "impugnados": 0,
            "fallos": [{"motivo": "verificacion_de_identidad"}],
        })
        self.servicio.paso()
        pregunta = self.ultima_pregunta()
        self.assertEqual(self.etiquetas(pregunta), ["🖥 Impugnar con ventana"])
        self.assertIn("Banda - V2", pregunta["texto"])
        self.telegram.apretar(pregunta, "🖥 Impugnar con ventana")
        self.servicio.paso()
        self.assertEqual(
            self.lanzador.lanzados("impugnar_con_ventana"),
            [["impugnar_con_ventana", "--video-id", "V1", "--video-id", "V2"]],
        )

    def test_un_liberado_que_ya_se_esta_programando_no_se_programa_dos_veces(self):
        # Defecto: dos /mapear_canal seguidos lanzaban dos `programar` del mismo video.
        liberado = [{"video_id": "V1", "titulo": "Banda - V1"}]
        self.mapear(aparcados=1, liberados=liberado)
        self.mapear(aparcados=1, liberados=liberado)
        self.assertEqual(len(self.lanzador.lanzados("programar")), 1)

    def test_un_resumen_largo_no_pierde_videos_y_el_boton_va_al_final(self):
        # Defecto: Telegram corta el texto en 4096 caracteres y los últimos videos no se ven.
        ids = [f"V{i:02d}" for i in range(60)]
        videos = [{**v, "titulo": f"Banda con un nombre bastante largo {v['video_id']} - Álbum (2026)"}
                  for v in self.sin_impugnar(*ids)]
        self.mapear(aparcados=60, sin_impugnar=videos)
        nuevos = self.telegram.mensajes[self.antes_del_resultado:]
        textos = [m["texto"] for m in nuevos]
        self.assertTrue(all(len(t) <= sp.LIMITE_TEXTO for t in textos))
        for v in ids:
            self.assertIn(f"https://studio.youtube.com/video/{v}/claims", "\n".join(textos))
        self.assertGreater(len(nuevos), 1)
        self.assertEqual([self.etiquetas(m) for m in nuevos if m["botones"]], [["⚖️ Impugnar todos (60)"]])
        self.assertTrue(nuevos[-1]["botones"])

    def test_un_segundo_impugnar_todos_no_repite_lo_que_el_lote_ya_hizo(self):
        # Defecto: un video ya impugnado en el lote se volvía a correr y salía dos veces.
        self.impugnar_todos()
        self.lanzador.terminar("impugnar", {"ok": True, "video_id": "V1", "reclamos": 1, "impugnados": 1, "fallos": []})
        self.servicio.paso()
        self.mapear(aparcados=2, sin_impugnar=self.sin_impugnar("V1", "V2"))
        self.telegram.apretar(self.ultima_pregunta(), self.TODOS)
        self.servicio.paso()
        self.lanzador.terminar("impugnar", {"ok": True, "video_id": "V2", "reclamos": 1, "impugnados": 1, "fallos": []})
        self.servicio.paso()
        self.assertEqual([a[2] for a in self.lanzador.lanzados("impugnar")], ["V1", "V2"])
        self.assertEqual(self.telegram.mensajes[-1]["texto"].count("Banda - V1"), 1)

    def test_un_mapeo_cortado_antes_de_empezar_no_dice_que_reviso_cero(self):
        # Defecto: "Revisé 0 de 0" cuando el paso ni llegó a leer la lista de subidas.
        escribir_en(self.telegram, "/mapear_canal")
        self.servicio.paso()
        self.lanzador.terminar("mapear_canal", {"ok": False, "motivo": "tiempo_agotado", "log": "x.log"})
        self.servicio.paso()
        texto = self.telegram.mensajes[-1]["texto"]
        self.assertIn(sp.MOTIVOS_APELACION["tiempo_agotado"], texto)
        self.assertNotRegex(texto, r"\b0\b")

    def test_un_mapeo_que_no_termina_dice_por_que(self):
        # Defecto: silencio después de que el usuario escribió el comando.
        self.mapear(ok=False, motivo="sin_sesion_en_studio", aparcados=4,
                    sin_revisar=[{"video_id": "V1", "titulo": "Banda - V1"}])
        self.assertIn(sp.MOTIVOS_IMPUGNACION["sin_sesion_en_studio"], self.telegram.mensajes[-1]["texto"])


if __name__ == "__main__":
    unittest.main()
