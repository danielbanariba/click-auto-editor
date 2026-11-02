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

        self.assertEqual(self.lanzador.lanzados("renderizar"), [["renderizar"]])

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
    def test_con_la_cola_vacia_renderiza_desde_album_o_ep(self):
        album = self.tmp / "Album"
        (album / "Banda - Disco (2020)").mkdir(parents=True)
        with mock.patch.object(pa, "DIRS_ORIGEN_RENDER", (self.cola, album)):
            self.servicio.paso()
        self.assertEqual(self.lanzador.lanzados("renderizar"), [["renderizar"]])

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
        self.album_en_cola("Banda - Disco")
        self.servicio.paso()
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

    def test_error_de_red_en_telegram_no_frena_el_servicio(self):
        self.album_en_cola("Banda - Disco")
        self.telegram.falla = sp.ErrorTelegram("sin red")

        self.servicio.paso()

        self.assertEqual(self.lanzador.lanzados("renderizar"), [["renderizar"]])


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


if __name__ == "__main__":
    unittest.main()
