"""`pipeline_agente.seguir_impugnaciones`, contra una página local que imita /claims de Studio.

Chromium real; Studio se simula respondiendo las rutas de studio.youtube.com. La forma de cada
fila es la vista en vivo el 2026-11-04: un `ytcr-video-content-list-row` con canción, artista,
"Derechos de autor", "Audio" y, en `#impact-container`, la línea de estado más "Ver detalles"
y "Tomar medidas". El seguimiento solo lee: cualquier botón pulsado llega al "servidor".
"""
import contextlib
import json
import os
import re
import shutil
import tempfile
import time
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
<style>ytcr-video-content-list, ytcr-video-content-list-row, ytcp-button { display: block; }</style>
</head><body>
<h1>Contenido protegido</h1>
<ytcr-video-content-list>__FILAS__</ytcr-video-content-list>
<div id="vacio">__VACIO__</div>
<script>
document.addEventListener("click", e => {
  const boton = e.target.closest("button");
  if (boton) fetch("/_pulsado", {method: "POST", body: boton.innerText});
}, true);
__DESPUES__
</script></body></html>"""

FILA = """<ytcr-video-content-list-row>
<div>{cancion}</div><div>{artista}</div><div>Derechos de autor</div><div>Audio</div>
<div id="impact-container" class="table-cell content-impact"><div>{estado}</div>
<div class="action-buttons"><ytcp-button class="details-button"><button>Ver detalles</button></ytcp-button>
<ytcp-button><button>Tomar medidas</button></ytcp-button></div></div>
</ytcr-video-content-list-row>"""

PAGINA_DE_ERROR = "<!doctype html><html><body><h1>Algo salió mal</h1></body></html>"

EN_REVISION = "Impugnación en proceso de revisión. Caduca dentro de {dias} días."
RECHAZADA = "El reclamante ha rechazado tu impugnación."


class StudioFalso:
    """Los reclamos de cada video, las páginas /claims que se abrieron y los botones pulsados."""

    def __init__(self, reclamos):
        self.reclamos = reclamos  # {video_id: [(canción, artista, estado)]}
        self.con_error = set()  # videos cuya /claims responde una página de error
        self.despues = {}  # {video_id: (ms, reclamos)}: la lista que se termina de dibujar después
        self.abiertos = []
        self.pulsados = []

    def atender(self, ruta):
        url = ruta.request.url
        if url.endswith("/_pulsado"):
            self.pulsados.append(ruta.request.post_data)
            return ruta.fulfill(status=200, body="{}", content_type="application/json")
        encontrado = re.search(r"/video/([^/]+)/claims", url)
        if not encontrado:
            return ruta.fulfill(status=200, body="<h1>Studio</h1>", content_type="text/html")
        video_id = encontrado.group(1)
        self.abiertos.append(video_id)
        if video_id in self.con_error:
            return ruta.fulfill(status=200, body=PAGINA_DE_ERROR, content_type="text/html; charset=utf-8")
        filas = self.reclamos.get(video_id, [])
        despues = ""
        if video_id in self.despues:
            ms, completas = self.despues[video_id]
            despues = (
                "setTimeout(() => { document.querySelector('ytcr-video-content-list').innerHTML = "
                f"{json.dumps(self.filas(completas))}; document.getElementById('vacio').textContent = ''; }}, {ms});"
            )
        html = PAGINA.replace("__FILAS__", self.filas(filas)).replace(
            "__VACIO__", "" if filas else "No se han encontrado reclamaciones en este vídeo"
        ).replace("__DESPUES__", despues)
        return ruta.fulfill(status=200, body=html, content_type="text/html; charset=utf-8")

    @staticmethod
    def filas(reclamos):
        return "".join(FILA.format(cancion=c, artista=a, estado=e) for c, a, e in reclamos)


class TestSeguirImpugnaciones(unittest.TestCase):
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
        self.logs = self.tmp / "logs"
        self.logs.mkdir()
        self.registro = self.tmp / "estado" / "impugnaciones.json"
        parches = [
            mock.patch.object(pa, "DIR_LOGS", self.logs),
            mock.patch.object(pa, "IMPUGNACIONES_JSON", self.registro),
            mock.patch.object(pa, "PENDIENTES_JSON", self.tmp / "estado" / "pendientes.json"),
            mock.patch.object(pa, "DIR_CAPTURAS", self.tmp / "capturas"),
            # Los tiempos de producción son los que Studio necesita; aquí la página es local.
            mock.patch.object(pa, "ESPERA_RESULTADO_RECLAMOS_S", 2),
            mock.patch.object(pa, "PAUSA_RELECTURA_RECLAMOS_S", 3),
        ]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def impugnado(self, video_id, impugnados, titulo):
        """La salida de un `impugnar` lanzado por el servicio, como la deja en data/agente/logs."""
        salida = self.logs / f"servicio_impugnar_20261104-0{len(list(self.logs.iterdir()))}0000_abc123.out"
        resultado = {"ok": bool(impugnados), "video_id": video_id, "titulo": titulo, "reclamos": 2,
                     "impugnados": impugnados, "fallos": []}
        salida.write_text("Lanzo impugnar\n" + json.dumps(resultado, ensure_ascii=False) + "\n", encoding="utf-8")

    def seguir(self, studio, abortar=()):
        """Una corrida de `seguir_impugnaciones`; la /claims de los videos de `abortar` no carga."""
        contexto = self.navegador.new_context()
        self.addCleanup(contexto.close)
        contexto.route("https://studio.youtube.com/**", studio.atender)
        for video_id in abortar:
            contexto.route(f"https://studio.youtube.com/**/{video_id}/claims", lambda ruta: ruta.abort())
        pagina = contexto.new_page()

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        return pa.seguir_impugnaciones(abrir_studio=abrir_studio)

    def test_avisa_un_cambio_de_estado_una_vez_y_no_la_cuenta_de_los_dias(self):
        # Defecto: un cambio de estado que no se avisa, o un aviso diario porque "Caduca dentro
        # de N días" cambia todos los días.
        self.impugnado("V1", 2, "Banda - Uno")
        self.impugnado("V2", 0, "Banda - Dos")  # la impugnación no salió: no hay nada que seguir
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29)),
                                     ("Canción B", "Artista", EN_REVISION.format(dias=29))]})
        primero = self.seguir(studio)
        studio.reclamos["V1"] = [("Canción A", "Artista", EN_REVISION.format(dias=28)),
                                 ("Canción B", "Artista", RECHAZADA)]
        segundo = self.seguir(studio)
        tercero = self.seguir(studio)
        self.assertEqual(primero["cambios"], [])
        self.assertEqual(segundo["cambios"], [{
            "video_id": "V1", "titulo": "Banda - Uno", "reclamo": "Canción B — Artista",
            "antes": EN_REVISION.format(dias=29), "ahora": RECHAZADA,
        }])
        self.assertEqual(tercero["cambios"], [])
        self.assertTrue(tercero["ok"], tercero)
        self.assertNotIn("V2", studio.abiertos)
        self.assertEqual(studio.pulsados, [])

    def test_tambien_sigue_lo_que_se_impugno_con_ventana(self):
        # Defecto: la corrida con ventana deja otro log (`servicio_impugnar_con_ventana_*`, con
        # varios videos) y su impugnación nunca se seguía.
        resultado = {"ok": False, "ventana": True, "sin_revisar": [], "videos": [
            {"ok": True, "video_id": "V1", "titulo": "Banda - Uno", "reclamos": 1, "impugnados": 1, "fallos": []},
            {"ok": False, "video_id": "V2", "titulo": "Banda - Dos", "reclamos": 1, "impugnados": 0, "fallos": []},
        ]}
        salida = self.logs / "servicio_impugnar_con_ventana_20261107-010000_abc123.out"
        salida.write_text(json.dumps(resultado, ensure_ascii=False) + "\n", encoding="utf-8")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        self.assertEqual(studio.abiertos, ["V1"])

    def test_un_reclamo_que_desaparece_se_avisa_y_el_video_sin_pendientes_deja_de_seguirse(self):
        # Defecto: un reclamo retirado (ya no está en la lista) no se avisaba, o el video se
        # seguía abriendo todos los días aunque no le quedara nada pendiente.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        studio.reclamos["V1"] = []
        segundo = self.seguir(studio)
        self.seguir(studio)
        self.assertEqual(
            [(c["reclamo"], c["ahora"]) for c in segundo["cambios"]], [("Canción A — Artista", None)]
        )
        self.assertEqual(studio.abiertos, ["V1", "V1"])

    def test_un_video_que_se_queda_sin_reclamos_sale_una_vez_en_liberados(self):
        # Defecto: un video liberado que no se vuelve a publicar, o que se relanza todos los días.
        # V2 ya no tenía reclamos en la primera lectura: se liberó antes de que se lo siguiera.
        self.impugnado("V1", 1, "Banda - Uno")
        self.impugnado("V2", 1, "Banda - Dos")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        primero = self.seguir(studio)
        studio.reclamos["V1"] = []
        segundo = self.seguir(studio)
        tercero = self.seguir(studio)
        self.assertEqual(primero["liberados"], [{"video_id": "V2", "titulo": "Banda - Dos"}])
        self.assertEqual(segundo["liberados"], [{"video_id": "V1", "titulo": "Banda - Uno"}])
        self.assertEqual((tercero["liberados"], tercero["rechazados"]), ([], []))
        self.assertEqual(primero["rechazados"] + segundo["rechazados"], [])

    def test_una_impugnacion_que_termina_con_reclamo_sale_una_vez_en_rechazados(self):
        # Defecto: un rechazo sin aviso, sin el estado que muestra Studio o sin captura; o un
        # video que todavía está en revisión dado por rechazado.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        primero = self.seguir(studio)
        studio.reclamos["V1"] = [("Canción A", "Artista", RECHAZADA)]
        segundo = self.seguir(studio)
        tercero = self.seguir(studio)
        self.assertEqual((primero["liberados"], primero["rechazados"]), ([], []))
        (rechazado,) = segundo["rechazados"]
        captura = rechazado.pop("captura")
        self.assertEqual(rechazado, {
            "video_id": "V1", "titulo": "Banda - Uno",
            "reclamos": [{"reclamo": "Canción A — Artista", "estado": RECHAZADA}],
        })
        self.assertTrue(Path(captura).is_file())
        self.assertEqual(segundo["liberados"], [])
        self.assertEqual(tercero["rechazados"], [])
        self.assertEqual(studio.pulsados, [])

    def test_una_corrida_que_se_corta_igual_devuelve_lo_que_ya_termino(self):
        # Defecto: un video ya marcado como terminado que no sale en el resultado de una corrida
        # cortada por un error de Studio, y como no se vuelve a mirar, no se publica nunca.
        self.impugnado("V1", 1, "Banda - Uno")
        self.impugnado("V2", 1, "Banda - Dos")
        studio = StudioFalso({"V2": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        contexto = self.navegador.new_context()
        self.addCleanup(contexto.close)
        contexto.route("https://studio.youtube.com/**", studio.atender)
        # La última ruta registrada atiende primero: /claims de V2 no carga y la corrida se corta.
        contexto.route("https://studio.youtube.com/**/V2/claims", lambda ruta: ruta.abort())
        pagina = contexto.new_page()

        @contextlib.contextmanager
        def abrir_studio():
            yield pagina

        resultado = pa.seguir_impugnaciones(abrir_studio=abrir_studio)
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["liberados"], [{"video_id": "V1", "titulo": "Banda - Uno"}])
        self.assertTrue(json.loads(self.registro.read_text(encoding="utf-8"))["V1"]["terminado"])

    def test_una_pagina_que_no_carga_es_un_fallo_y_no_da_nada_por_retirado(self):
        # Defecto: una /claims que no cargó se leía como "sin reclamos" y daba todo por retirado.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        studio.con_error.add("V1")
        segundo = self.seguir(studio)
        studio.con_error.clear()
        tercero = self.seguir(studio)
        self.assertFalse(segundo["ok"])
        self.assertEqual(segundo["cambios"], [])
        self.assertEqual([(f["video_id"], f["motivo"]) for f in segundo["fallos"]],
                         [("V1", "lista_de_reclamos_no_disponible")])
        self.assertEqual(tercero["cambios"], [])

    def test_un_video_terminado_vuelve_a_seguirse_si_despues_se_impugna_otro_de_sus_reclamos(self):
        # Defecto: `terminado` no se revisaba nunca más: un reclamo impugnado después de cerrar el
        # video quedaba sin seguir y su resultado (liberado o rechazado) no llegaba.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", RECHAZADA)]})
        self.seguir(studio)
        self.assertTrue(json.loads(self.registro.read_text(encoding="utf-8"))["V1"]["terminado"])
        self.impugnado("V1", 1, "Banda - Uno")
        # El log nuevo se escribió después de la lectura (impugnar tarda minutos).
        nuevo = max(self.logs.iterdir(), key=lambda archivo: archivo.name)
        os.utime(nuevo, (time.time() + 3600, time.time() + 3600))
        studio.reclamos["V1"] = [("Canción A", "Artista", RECHAZADA),
                                 ("Canción B", "Artista", EN_REVISION.format(dias=29))]
        segunda = self.seguir(studio)
        self.assertEqual(studio.abiertos, ["V1", "V1"])
        self.assertEqual(segunda["rechazados"], [])  # el reclamo nuevo sigue en revisión
        self.assertFalse(json.loads(self.registro.read_text(encoding="utf-8"))["V1"]["terminado"])

    def test_un_video_cuya_lectura_aborta_no_frena_el_seguimiento_de_los_demas(self):
        # Defecto: el video que aborta la corrida no actualizaba `revisado`: al día siguiente
        # volvía a ser el primero, abortaba otra vez y los demás nunca se seguían.
        self.impugnado("V1", 1, "Banda - Uno")
        self.impugnado("V2", 1, "Banda - Dos")
        studio = StudioFalso({v: [("Canción A", "Artista", EN_REVISION.format(dias=29))] for v in ("V1", "V2")})
        primera = self.seguir(studio, abortar=["V1"])
        segunda = self.seguir(studio, abortar=["V1"])
        self.assertEqual(primera["motivo"], "error_navegador")
        self.assertEqual(studio.abiertos, ["V2"])  # la segunda corrida llegó a V2
        self.assertEqual(segunda["motivo"], "error_navegador")

    def test_un_video_cuya_lista_no_se_lee_actualiza_su_revisado_y_pasa_al_final(self):
        # Defecto: una lectura sin resultado no actualizaba `revisado`: el video ilegible seguía
        # primero cada día y gastaba el tiempo de la corrida antes que los demás.
        antes = "2026-11-02T00:00:00"
        self.registro.parent.mkdir(parents=True)
        self.registro.write_text(json.dumps({"V1": {"titulo": "Banda - Uno", "revisado": antes}}), encoding="utf-8")
        studio = StudioFalso({})
        studio.con_error.add("V1")
        resultado = self.seguir(studio)
        self.assertEqual([f["video_id"] for f in resultado["fallos"]], ["V1"])
        guardado = json.loads(self.registro.read_text(encoding="utf-8"))["V1"]
        self.assertGreater(guardado["revisado"], antes)
        self.assertFalse(guardado.get("terminado"))

    def test_si_falla_la_captura_del_rechazado_el_video_no_queda_terminado_y_la_pregunta_llega_despues(self):
        # Defecto: el video se guardaba como terminado antes de la captura; si ésta lanzaba un
        # error que no es de Playwright, la corrida se caía con el video ya cerrado y su pregunta
        # de Eliminar no llegaba nunca.
        self.impugnado("V1", 1, "Banda - Uno")
        studio = StudioFalso({"V1": [("Canción A", "Artista", EN_REVISION.format(dias=29))]})
        self.seguir(studio)
        studio.reclamos["V1"] = [("Canción A", "Artista", RECHAZADA)]
        with mock.patch.object(pa, "capturar_pagina", side_effect=OSError("disco lleno")):
            fallida = self.seguir(studio)
        self.assertFalse(fallida["ok"])
        self.assertEqual(fallida["rechazados"], [])
        self.assertFalse(json.loads(self.registro.read_text(encoding="utf-8"))["V1"].get("terminado"))
        sana = self.seguir(studio)
        (rechazado,) = sana["rechazados"]
        self.assertEqual(rechazado["video_id"], "V1")
        self.assertTrue(Path(rechazado["captura"]).is_file())
        # El cambio de estado salió una sola vez: no se avisó con la corrida que se cayó.
        self.assertEqual([c["ahora"] for c in sana["cambios"]], [RECHAZADA])
        self.assertEqual(fallida["cambios"], [])

    def seguir_con_la_lista_a_medio_dibujar(self, filas_al_principio, ms_hasta_el_resto=1500):
        self.impugnado("V1", 2, "Banda - Uno")
        dos = [("Canción A", "Artista", EN_REVISION.format(dias=29)),
               ("Canción B", "Artista", EN_REVISION.format(dias=29))]
        studio = StudioFalso({"V1": dos})
        self.seguir(studio)
        studio.reclamos["V1"] = dos[:filas_al_principio]
        studio.despues["V1"] = (ms_hasta_el_resto, dos)  # el resto de la lista llega después
        resultado = self.seguir(studio)
        guardado = json.loads(self.registro.read_text(encoding="utf-8"))["V1"]
        return resultado, guardado

    def test_una_lista_que_llega_por_partes_no_da_reclamos_por_retirados(self):
        # Defecto: la lista leída antes de que llegara su segunda fila daba esa fila por retirada.
        resultado, guardado = self.seguir_con_la_lista_a_medio_dibujar(1)
        self.assertEqual((resultado["cambios"], resultado["fallos"]), ([], []))
        self.assertEqual(sorted(guardado["reclamos"]), ["Canción A — Artista", "Canción B — Artista"])

    def test_un_aviso_de_sin_reclamos_antes_de_la_lista_no_da_nada_por_retirado(self):
        # Defecto: el aviso de "sin reclamos" que Studio muestra un rato antes de la lista daba
        # todo por retirado y el video dejaba de seguirse.
        resultado, guardado = self.seguir_con_la_lista_a_medio_dibujar(0)
        self.assertEqual((resultado["cambios"], resultado["fallos"]), ([], []))
        self.assertEqual(sorted(guardado["reclamos"]), ["Canción A — Artista", "Canción B — Artista"])
        self.assertFalse(guardado["terminado"])

    def test_un_aviso_de_sin_reclamos_que_dura_mas_que_la_pausa_no_libera_el_video(self):
        # Defecto: con el aviso de "sin reclamos" en pantalla más que la pausa entre lecturas, la
        # segunda lectura lo veía igual, la lista vacía valía y el servicio publicaba solo un
        # video con reclamo. Aquí el aviso dura 6 s contra una pausa de 3 s.
        resultado, guardado = self.seguir_con_la_lista_a_medio_dibujar(0, ms_hasta_el_resto=6000)
        self.assertEqual(resultado["liberados"], [])
        self.assertEqual((resultado["cambios"], resultado["fallos"]), ([], []))
        self.assertEqual(sorted(guardado["reclamos"]), ["Canción A — Artista", "Canción B — Artista"])
        self.assertFalse(guardado["terminado"])


class TestCompararReclamos(unittest.TestCase):
    def test_pasar_de_dias_a_un_dia_no_es_un_cambio_pero_un_estado_distinto_si(self):
        # Defecto: "Caduca dentro de 1 día" contra "N días" contaba como cambio de estado y daba
        # un aviso de más al final del plazo.
        antes = {"Canción A": EN_REVISION.format(dias=2), "Canción B": EN_REVISION.format(dias=2)}
        reclamos = [
            {"reclamo": "Canción A", "estado": "Impugnación en proceso de revisión. Caduca dentro de 1 día."},
            {"reclamo": "Canción B", "estado": RECHAZADA},
        ]
        _, cambios = pa.comparar_reclamos(antes, reclamos)
        self.assertEqual([(c["reclamo"], c["ahora"]) for c in cambios], [("Canción B", RECHAZADA)])


if __name__ == "__main__":
    unittest.main()
