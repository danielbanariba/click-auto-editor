"""Tests de la censura de portadas: la geometría de las cajas y la llamada al modelo.

El modelo nunca se llama: `requests` es un Ollama falso que contesta como el de verdad
(chat, generate y ps), así que el cuerpo de cada pedido y la descarga se prueban reales.
"""
import itertools
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlparse

import requests
from PIL import Image

import censura_portadas as cp


class RespuestaFalsa:
    def __init__(self, codigo, cuerpo):
        self.status_code = codigo
        self._cuerpo = cuerpo
        self.text = json.dumps(cuerpo)

    def json(self):
        return self._cuerpo


class OllamaFalso:
    """Contesta /api/chat con las respuestas dadas, en orden; un Exception se lanza, un int es un HTTP.

    Las preguntas por lo sexual toman de `respuestas`; las del gore, de `gore`, y cuando se acaba
    contesta "[]" (nada de gore).
    """

    def __init__(self, respuestas=(), ignora_la_descarga=False, gore=()):
        self.respuestas = list(respuestas)
        self.gore = list(gore)
        self.pedidos = []
        self.cargado = False
        self.ignora_la_descarga = ignora_la_descarga
        self.apagado_a_la_fuerza = 0

    def chats(self):
        return [cuerpo for ruta, cuerpo, _ in self.pedidos if ruta == "/api/chat"]

    def post(self, url, json=None, timeout=None):
        ruta = urlparse(url).path
        self.pedidos.append((ruta, json, timeout))
        if ruta == "/api/chat":
            pregunta = json["messages"][0]["content"]
            if pregunta in (cp.PROMPT_GORE_IMAGEN_ENTERA, cp.PROMPT_GORE_PEDAZO):
                respuesta = self.gore.pop(0) if self.gore else "[]"
            else:
                respuesta = self.respuestas.pop(0)
            if isinstance(respuesta, Exception):
                raise respuesta
            if isinstance(respuesta, int):
                return RespuestaFalsa(respuesta, {"error": "no"})
            self.cargado = True
            return RespuestaFalsa(200, {"message": {"content": respuesta}})
        if ruta == "/api/generate" and json.get("keep_alive") == 0 and not self.ignora_la_descarga:
            self.cargado = False
        return RespuestaFalsa(200, {})

    def get(self, url, timeout=None):
        modelos = [{"name": cp.MODELO_CENSURA, "model": cp.MODELO_CENSURA}] if self.cargado else []
        return RespuestaFalsa(200, {"models": modelos})

    def parchar(self, test):
        for nombre in ("post", "get"):
            parche = mock.patch.object(requests, nombre, getattr(self, nombre))
            parche.start()
            test.addCleanup(parche.stop)
        # `ollama stop` es el último recurso de la descarga: aquí solo apaga el modelo falso.
        def parar(comando, **kwargs):
            self.apagado_a_la_fuerza += 1
            self.cargado = False
            return SimpleNamespace(returncode=0)

        for parche in (
            mock.patch.object(cp.subprocess, "run", side_effect=parar),
            mock.patch.object(cp.shutil, "which", return_value="/usr/bin/ollama"),
        ):
            parche.start()
            test.addCleanup(parche.stop)
        return self


def respuesta_con(*cajas):
    """Lo que contesta el modelo: [{"label", "bbox_2d"}] con coordenadas de 0 a 1000."""
    return json.dumps([{"label": etiqueta, "bbox_2d": list(caja)} for etiqueta, caja in cajas])


class TestAmpliarCajas(unittest.TestCase):
    def test_cada_caja_crece_la_mitad_de_su_ancho_y_de_su_alto_a_cada_lado(self):
        # Defecto: un margen aplicado una sola vez (o repartido entre los dos lados) deja la caja
        # más chica de lo que el usuario pidió: «más grande la censura».
        (caja,) = cp.ampliar_cajas([(400, 400, 500, 460)], 1000, 1000, margen=0.5)
        self.assertEqual(caja, (350, 370, 550, 490))
        ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
        self.assertEqual((ancho, alto), (200, 120))
        (con_otro_margen,) = cp.ampliar_cajas([(400, 400, 500, 460)], 1000, 1000, margen=0.25)
        self.assertEqual(con_otro_margen, (375, 385, 525, 475))

    def test_sin_margen_dicho_la_caja_mide_el_doble_en_cada_dimension(self):
        # Defecto: la constante por defecto no cumple lo que se pidió (el doble en cada dimensión).
        (caja,) = cp.ampliar_cajas([(400, 400, 500, 460)], 1000, 1000)
        self.assertEqual((caja[2] - caja[0], caja[3] - caja[1]), (200, 120))

    def test_la_caja_se_recorta_al_borde_de_la_imagen(self):
        # Defecto: una caja fuera de la imagen hace que el relleno se salga o falle al pintar.
        cajas = cp.ampliar_cajas([(0, 0, 100, 100), (900, 950, 1000, 1000)], 1000, 1000, margen=0.5)
        self.assertEqual(sorted(cajas), [(0, 0, 150, 150), (850, 925, 1000, 1000)])

    def test_las_cajas_que_se_superponen_o_se_tocan_se_unen_en_una(self):
        # Defecto: dos rellenos pegados dejan una rendija sin tapar entre ellos.
        superpuestas = cp.ampliar_cajas([(100, 100, 200, 200), (230, 100, 330, 200)], 1000, 1000)
        self.assertEqual(superpuestas, [(50, 50, 380, 250)])
        # Las ampliadas apenas se tocan por el borde: (50..250) y (250..450).
        tocandose = cp.ampliar_cajas([(100, 100, 200, 200), (300, 100, 400, 200)], 1000, 1000)
        self.assertEqual(tocandose, [(50, 50, 450, 250)])

    def test_la_union_encadena_cajas_que_solo_se_tocan_a_traves_de_otra(self):
        # Defecto: unir de a pares una sola vez deja dos cajas que ya se tocan tras la primera unión.
        # La del medio llega al final: recién al unirla con la primera toca a la tercera.
        cajas = [(100, 100, 200, 200), (500, 100, 600, 200), (300, 100, 400, 200)]
        self.assertEqual(cp.ampliar_cajas(cajas, 1000, 1000), [(50, 50, 650, 250)])

    def test_las_cajas_lejanas_siguen_separadas_y_sin_cajas_no_hay_nada_que_tapar(self):
        # Defecto: unir todo en un solo bloque taparía media portada por una sola zona.
        cajas = cp.ampliar_cajas([(50, 50, 100, 100), (800, 800, 850, 850)], 1000, 1000)
        self.assertEqual(sorted(cajas), [(25, 25, 125, 125), (775, 775, 875, 875)])
        self.assertEqual(cp.ampliar_cajas([], 1000, 1000), [])


class TestPedazos(unittest.TestCase):
    def test_el_primer_intento_mira_la_imagen_entera(self):
        self.assertEqual(cp.pedazos(2000, 1000, 1), [((0, 0, 2000, 1000), True)])

    def test_el_segundo_intento_son_cuatro_pedazos_de_60_por_ciento_que_cubren_todo(self):
        # Defecto: pedazos que no se superponen parten por la mitad lo que está justo en el corte
        # (la advertencia pasó porque el modelo lo vio recién en un pedazo).
        pedazos = cp.pedazos(1000, 500, 2)
        self.assertEqual(len(pedazos), 4)
        self.assertFalse(any(entera for _, entera in pedazos))
        cajas = sorted(caja for caja, _ in pedazos)
        self.assertEqual(cajas, [(0, 0, 600, 300), (0, 200, 600, 500), (400, 0, 1000, 300), (400, 200, 1000, 500)])

    def test_del_tercer_intento_en_adelante_son_nueve_pedazos_mas_la_imagen_entera(self):
        for intento in (3, 4, 7):
            pedazos = cp.pedazos(1000, 1000, intento)
            self.assertEqual(len(pedazos), 10, intento)
            self.assertEqual([c for c, entera in pedazos if entera], [(0, 0, 1000, 1000)])
            trozos = [c for c, entera in pedazos if not entera]
            self.assertEqual(len(trozos), 9)
            # Cada pedazo ronda el 45 % del lado, y los de una fila se superponen con el vecino.
            self.assertTrue(all(440 <= c[2] - c[0] <= 460 and 440 <= c[3] - c[1] <= 460 for c in trozos))
            self.assertEqual(min(c[0] for c in trozos), 0)
            self.assertEqual(max(c[2] for c in trozos), 1000)
            fila = sorted(c for c in trozos if c[1] == 0)
            self.assertTrue(all(a[2] > b[0] for a, b in zip(fila, fila[1:])))


class TestUrlDeOllama(unittest.TestCase):
    def test_un_ollama_host_sin_esquema_ni_puerto_va_al_11434_como_el_cli_de_ollama(self):
        # Defecto: OLLAMA_HOST=0.0.0.0 o localhost (como lo muestra ollama) apuntaba al puerto 80 y
        # todo análisis fallaba con «Ollama no responde».
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0"}):
            self.assertEqual(cp.ClienteOllama().url, "http://0.0.0.0:11434")
        self.assertEqual(cp.url_de_ollama("localhost"), "http://localhost:11434")
        self.assertEqual(cp.url_de_ollama("[::1]"), "http://[::1]:11434")
        # Lo que ya funcionaba sigue igual: con puerto se respeta, y con esquema el puerto que
        # falta es el del esquema (así lo lee ollama).
        self.assertEqual(cp.url_de_ollama("otra-pc:8080"), "http://otra-pc:8080")
        self.assertEqual(cp.url_de_ollama("https://ollama.example/"), "https://ollama.example")


class TestLeerDetecciones(unittest.TestCase):
    def test_lee_el_json_aunque_venga_con_texto_o_comillas_de_codigo_alrededor(self):
        # Defecto: el modelo envuelve el JSON en ```json ...```; leerlo tal cual lo descartaba.
        texto = 'Aquí está:\n```json\n[{"label": "penis", "bbox_2d": [100, 200, 300, 400]}]\n```'
        self.assertEqual(cp.leer_detecciones(texto), [("penis", (100, 200, 300, 400))])

    def test_una_lista_vacia_es_una_respuesta_valida_y_un_texto_sin_json_no(self):
        # Defecto: confundir «no hay nada» con «no entendí» hace fallar el intento cuando no había nada.
        self.assertEqual(cp.leer_detecciones("[]"), [])
        self.assertIsNone(cp.leer_detecciones("I cannot help with that."))
        self.assertIsNone(cp.leer_detecciones('[{"label": "penis", "bbox_2d": [1, 2'))
        self.assertIsNone(cp.leer_detecciones('{"label": "penis"}'))

    def test_un_objeto_suelto_con_su_caja_se_lee_como_una_deteccion(self):
        # Defecto: sin la lista alrededor, se leía solo el [x1, y1, x2, y2] de adentro y la
        # portada salía «sin nada que tapar» aunque el modelo hubiera ubicado la zona.
        texto = 'Encontré esto: {"label": "penis", "bbox_2d": [100, 200, 300, 400]}'
        self.assertEqual(cp.leer_detecciones(texto), [("penis", (100, 200, 300, 400))])

    def test_una_lista_con_elementos_pero_ninguna_caja_usable_no_es_nada_que_tapar(self):
        # Defecto: si el modelo cambia el formato (otra clave, números como texto, pares), todo se
        # salteaba y el pedazo contaba como limpio en vez de como no entendido.
        self.assertIsNone(cp.leer_detecciones('[{"label": "penis", "box": [100, 200, 300, 400]}]'))
        self.assertIsNone(cp.leer_detecciones('[{"label": "penis", "bbox_2d": ["100", "200", "300", "400"]}]'))
        self.assertIsNone(cp.leer_detecciones("[[100, 200, 300, 400]]"))

    def test_una_respuesta_rota_o_con_dos_valores_no_se_lee_a_medias(self):
        # Defecto: con la lista de afuera rota (una coma de más, cortada) o con dos valores
        # seguidos, se tomaba solo la primera caja y el pedazo contaba como leído: las otras zonas
        # que marcó el modelo quedaban sin tapar y sin aviso.
        caja = '{"label": "penis", "bbox_2d": [100, 200, 300, 400]}'
        otra = '{"label": "anus", "bbox_2d": [500, 500, 600, 600]}'
        self.assertIsNone(cp.leer_detecciones(f"[{caja}, {otra},]"))
        self.assertIsNone(cp.leer_detecciones(f'[{caja}, {{"label": "anus", "bbox_2d": [5'))
        self.assertIsNone(cp.leer_detecciones(f"{caja}\n{otra}"))
        self.assertIsNone(cp.leer_detecciones(f"[{caja}]\n[{otra}]"))

    def test_salta_las_cajas_rotas_sin_tirar_las_buenas(self):
        # Defecto: una sola caja mal formada tiraba todas las del mismo pedazo.
        texto = json.dumps([
            {"label": "vulva", "bbox_2d": [10, 20, 30]},
            {"label": "anus", "bbox_2d": [10, 20, "x", 40]},
            {"label": "anus", "bbox_2d": [50, 50, 50, 90]},
            "penis",
            {"label": "penis", "bbox_2d": [100, 200, 300, 400]},
        ])
        self.assertEqual(cp.leer_detecciones(texto), [("penis", (100, 200, 300, 400))])

    def test_corrige_coordenadas_dadas_al_reves_y_las_acota_a_0_1000(self):
        texto = json.dumps([{"label": "penis", "bbox_2d": [300, 400, -20, 1200]}])
        self.assertEqual(cp.leer_detecciones(texto), [("penis", (0, 400, 300, 1000))])


class TestEtiquetaLegible(unittest.TestCase):
    def test_el_gore_sale_en_castellano_por_lo_que_nombra_y_lo_sexual_gana(self):
        # Defecto: el aviso decía «gore element: exposed organs or entrails» en inglés, «gore» donde
        # el modelo dijo sangre, o «sangre» donde había un pene. Las etiquetas son del modelo real.
        casos = {
            "gore element: exposed organs or entrails": "vísceras",
            "gore element: blood": "sangre",
            "open wounds": "heridas",
            "corpses": "cadáver",
            "severed body parts": "restos desmembrados",
            "gore": "gore",
            "bloody exposed penis": "pene",
        }
        for etiqueta, texto in casos.items():
            with self.subTest(etiqueta):
                self.assertEqual(cp.etiqueta_legible(etiqueta), texto)


class TestBuscarCajas(unittest.TestCase):
    def setUp(self):
        self.imagen = Image.new("RGB", (2000, 1000), "white")

    def buscar(self, ollama, intento=1, **kwargs):
        ollama.parchar(self)
        return cp.buscar_cajas(self.imagen, intento, cp.ClienteOllama(host="http://ollama.test"), **kwargs)

    def test_primer_intento_manda_la_imagen_entera_achicada_y_pasa_las_cajas_a_pixeles(self):
        # Defecto: devolver coordenadas 0-1000 sin pasarlas a píxeles taparía la esquina de la imagen.
        ollama = OllamaFalso([respuesta_con(("penis", (100, 200, 300, 400)))])

        cajas = self.buscar(ollama).cajas

        self.assertEqual(cajas, [("penis", 200.0, 200.0, 600.0, 400.0)])
        for pedido in ollama.chats():
            self.assertIs(pedido["think"], False)
            self.assertEqual(pedido["options"]["temperature"], 0)
            self.assertIs(pedido["stream"], False)
            self.assertEqual(len(pedido["messages"][0]["images"]), 1)

    def test_la_imagen_va_achicada_a_1024_de_lado(self):
        # Defecto: mandar la portada de 4000 px entera satura la VRAM y tarda minutos.
        import base64
        import io

        ollama = OllamaFalso(["[]"])
        self.buscar(ollama)
        enviada = Image.open(io.BytesIO(base64.b64decode(ollama.chats()[0]["messages"][0]["images"][0])))
        self.assertEqual(max(enviada.size), 1024)
        self.assertEqual(enviada.size, (1024, 512))

    def test_segundo_intento_manda_cuatro_pedazos_y_suma_su_posicion_a_cada_caja(self):
        # Defecto: sin sumar el origen del pedazo, la caja del pedazo de abajo a la derecha queda arriba a la izquierda.
        ollama = OllamaFalso([respuesta_con(("penis", (0, 0, 1000, 1000)))] + ["[]"] * 3)

        cajas = self.buscar(ollama, intento=2).cajas

        # Dos preguntas por pedazo: lo sexual y el gore.
        self.assertEqual(len(ollama.chats()), 8)
        # El primer pedazo es el de arriba a la izquierda, de 1200 x 600 px.
        self.assertEqual(cajas, [("penis", 0.0, 0.0, 1200.0, 600.0)])

    def test_un_pedazo_con_respuesta_ilegible_no_tira_las_cajas_de_los_demas(self):
        # Defecto: un pedazo con texto en vez de JSON hacía fallar el intento entero.
        ollama = OllamaFalso(["no puedo", respuesta_con(("anus", (0, 0, 500, 500))), "[]", "[]"])

        cajas = self.buscar(ollama, intento=2).cajas

        # Segundo pedazo: nace en x=800 (0.4 de 2000), y=0.
        self.assertEqual(cajas, [("anus", 800.0, 0.0, 1400.0, 300.0)])

    def test_cuenta_los_pedazos_que_no_se_pudieron_leer(self):
        # Defecto: un pedazo que el modelo no contestó se daba por limpio sin avisar, y el usuario
        # veía «no se encontró nada que tapar» justo cuando ese pedazo tenía lo explícito.
        ollama = OllamaFalso(["no puedo", "[]", "I cannot help with that.", "[]"])

        resultado = self.buscar(ollama, intento=2)

        self.assertEqual((resultado.cajas, resultado.pedazos, resultado.sin_leer), ([], 4, 2))

    def test_cada_pedazo_se_pregunta_aparte_por_el_gore_y_sus_cajas_se_tapan(self):
        # Defecto: una portada con un cadáver, sangre o vísceras salía «sin nada que tapar», porque
        # solo se buscaba lo sexual.
        ollama = OllamaFalso(
            [respuesta_con(("penis", (100, 200, 300, 400)))],
            gore=[respuesta_con(("blood", (500, 0, 1000, 1000)))],
        )

        cajas = self.buscar(ollama).cajas

        self.assertEqual(cajas, [("penis", 200.0, 200.0, 600.0, 400.0), ("blood", 1000.0, 0.0, 2000.0, 1000.0)])
        self.assertEqual(len(ollama.chats()), 2)

    def test_una_pregunta_sin_leer_marca_el_pedazo_sin_tirar_lo_que_contesto_la_otra(self):
        # Defecto: si el modelo se negaba a la pregunta del gore, el pedazo se daba por limpio sin
        # aviso; si se negaba a la sexual, el intento fallaba y se perdía la sangre que sí marcó.
        casos = {
            "gore sin leer": (["[]"], ["no puedo"], []),
            "sexual sin leer": (
                ["no puedo"], [respuesta_con(("blood", (0, 0, 500, 500)))], [("blood", 0.0, 0.0, 1000.0, 500.0)]
            ),
        }
        for caso, (sexual, gore, cajas) in casos.items():
            with self.subTest(caso):
                resultado = self.buscar(OllamaFalso(sexual, gore=gore))
                self.assertEqual((resultado.cajas, resultado.pedazos, resultado.sin_leer), (cajas, 1, 1))

    def test_si_ningun_pedazo_se_entiende_el_intento_falla(self):
        # Defecto: dar «no hay nada que tapar» cuando el modelo no contestó nada utilizable
        # deja pasar una portada sin censurar.
        ollama = OllamaFalso(["no puedo"] * 4, gore=["no puedo"] * 4)
        with self.assertRaises(cp.ErrorDeCensura) as error:
            self.buscar(ollama, intento=2)
        self.assertEqual(error.exception.motivo, "respuesta_invalida")

    def test_ollama_caido_falta_el_modelo_y_tiempo_agotado_tienen_su_motivo(self):
        # Defecto: todo fallo llegaba como «error» y el aviso no decía qué arreglar.
        casos = {
            "ollama_caido": requests.exceptions.ConnectionError("rechazada"),
            "tiempo_agotado": requests.exceptions.ReadTimeout("lento"),
            "falta_el_modelo": 404,
            "ollama_error": 500,
        }
        for motivo, respuesta in casos.items():
            with self.subTest(motivo):
                with self.assertRaises(cp.ErrorDeCensura) as error:
                    self.buscar(OllamaFalso([respuesta]))
                self.assertEqual(error.exception.motivo, motivo)

    def test_un_analisis_que_pasa_su_presupuesto_se_corta_antes_de_la_siguiente_pregunta(self):
        # Defecto: diez pedazos con un modelo lento mantienen la GPU ocupada horas.
        ollama = OllamaFalso(["[]"] * 4)
        reloj = itertools.count(0, 6).__next__
        with self.assertRaises(cp.ErrorDeCensura) as error:
            self.buscar(ollama, intento=2, reloj=reloj, presupuesto_s=10)
        self.assertEqual(error.exception.motivo, "tiempo_agotado")
        self.assertEqual(len(ollama.chats()), 1)


class TestDescargarElModelo(unittest.TestCase):
    def descargar(self, ollama):
        ollama.parchar(self)
        pausas = []
        cliente = cp.ClienteOllama(host="http://ollama.test", pausa=pausas.append, espera_descarga_s=3)
        return cliente.descargar()

    def test_pide_la_descarga_y_confirma_que_ya_no_esta_en_la_gpu(self):
        # Defecto: dar por descargado el modelo sin mirar /api/ps deja 17 GB ocupados para el render.
        ollama = OllamaFalso()
        ollama.cargado = True

        self.assertIs(self.descargar(ollama), True)

        self.assertFalse(ollama.cargado)
        generar = [cuerpo for ruta, cuerpo, _ in ollama.pedidos if ruta == "/api/generate"]
        self.assertEqual(generar, [{"model": cp.MODELO_CENSURA, "keep_alive": 0}])
        self.assertEqual(ollama.apagado_a_la_fuerza, 0)

    def test_si_sigue_cargado_lo_descarga_a_la_fuerza(self):
        # Defecto: un Ollama que ignora keep_alive 0 dejaba el modelo en la GPU para siempre.
        ollama = OllamaFalso(ignora_la_descarga=True)
        ollama.cargado = True

        self.assertIs(self.descargar(ollama), True)

        self.assertEqual(ollama.apagado_a_la_fuerza, 1)

    def test_si_no_se_puede_descargar_dice_que_sigue_cargado(self):
        # Defecto: ocultar que el modelo sigue en la GPU hace fallar el próximo render por memoria sin aviso.
        ollama = OllamaFalso(ignora_la_descarga=True)
        ollama.cargado = True
        ollama.parchar(self)
        with mock.patch.object(cp.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            cliente = cp.ClienteOllama(host="http://ollama.test", pausa=lambda s: None, espera_descarga_s=3)
            self.assertIs(cliente.descargar(), False)

    def test_con_ollama_caido_no_hay_nada_cargado(self):
        ollama = OllamaFalso()
        ollama.parchar(self)
        error = requests.exceptions.ConnectionError("rechazada")
        with mock.patch.object(requests, "post", side_effect=error), mock.patch.object(
            requests, "get", side_effect=error
        ):
            cliente = cp.ClienteOllama(host="http://ollama.test", pausa=lambda s: None, espera_descarga_s=3)
            self.assertIs(cliente.descargar(), True)


class TestPintar(unittest.TestCase):
    def test_el_relleno_es_negro_opaco_y_no_toca_lo_que_esta_fuera_de_la_caja(self):
        # Defecto: un desenfoque o un pixelado dejan el cuerpo reconocible para un clasificador.
        imagen = Image.new("RGB", (100, 100), (200, 100, 50))

        pintada = cp.pintar_cajas(imagen, [(20, 20, 60, 60)])

        self.assertEqual(pintada.getpixel((20, 20)), (0, 0, 0))
        self.assertEqual(pintada.getpixel((59, 59)), (0, 0, 0))
        self.assertEqual(pintada.getpixel((40, 40)), (0, 0, 0))
        for fuera in ((19, 20), (60, 60), (20, 19), (99, 99)):
            self.assertEqual(pintada.getpixel(fuera), (200, 100, 50), fuera)
        # La imagen que se pasa no se modifica: es la original de la portada.
        self.assertEqual(imagen.getpixel((40, 40)), (200, 100, 50))


if __name__ == "__main__":
    unittest.main()
