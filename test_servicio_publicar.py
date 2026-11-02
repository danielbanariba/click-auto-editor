"""Tests del servicio que publica álbumes con aprobación por Telegram.

Telegram y los pasos de pipeline_agente.py son falsos; las carpetas y el
estado del servicio son reales (en un directorio temporal).
"""
import shutil
import tempfile
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
            mock.patch.object(pa, "DIR_AUDIO_SCRIPTS", self.cola),
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
        self.servicio = self.nuevo_servicio()

    def nuevo_servicio(self):
        servicio = sp.Servicio(
            self.telegram, self.lanzador, estado_json=self.tmp / "servicio.json"
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

    def pedir_aprobacion(self, nombre="Banda - Disco"):
        """Lleva un álbum de la cola hasta el mensaje con los botones."""
        self.album_en_cola(nombre)
        self.servicio.paso()
        carpeta = self.album_renderizado(nombre)
        shutil.rmtree(self.cola / nombre)
        self.lanzador.terminar("renderizar", {"ok": True, "album": nombre, "carpeta": carpeta})
        self.servicio.paso()
        self.lanzador.terminar(
            "verificar",
            {
                "ok": True,
                "album": nombre,
                "captura": str(self.tmp / "captura.jpg"),
                "captura_busqueda": str(self.tmp / "busqueda.png"),
                "busqueda": f"{nombre} (Full Album)",
                "resumen": f"{nombre} · 45:00\n✓ tiene sonido: sí",
            },
        )
        self.servicio.paso()
        return carpeta, self.telegram.mensajes[-1]


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
        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 1)

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
        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 2)
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
        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 1)
        shutil.rmtree(carpeta)
        self.lanzador.terminar("eliminar", {"ok": True, "album": "Banda - Disco"})
        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("eliminar"), [["eliminar", "--carpeta", carpeta]])
        self.assertEqual(self.lanzador.lanzados("subir"), [])
        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 2)

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
        self.assertEqual(len(self.lanzador.lanzados("renderizar")), 2)


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


class TestAvisos(BaseServicio):
    def test_sin_albumes_avisa_una_sola_vez(self):
        for _ in range(3):
            self.servicio.paso()

        self.assertEqual(len(self.telegram.mensajes), 1)
        self.assertIn("No quedan álbumes", self.telegram.mensajes[0]["texto"])
        self.assertEqual(self.lanzador.lanzados(), [])

    def test_error_de_red_en_telegram_no_frena_el_servicio(self):
        self.album_en_cola("Banda - Disco")
        self.telegram.falla = sp.ErrorTelegram("sin red")

        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("renderizar"), [["renderizar"]])


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


if __name__ == "__main__":
    unittest.main()
