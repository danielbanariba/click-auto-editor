"""Clave y registro de álbumes ya publicados.

Los pares de nombres son reales: carpetas de Album/ y EP/ y títulos del canal
(sonda del 2026-11-04).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import registro_publicados as rp


class TestClave(unittest.TestCase):
    def test_la_carpeta_y_el_titulo_del_canal_dan_la_misma_clave(self):
        # Defecto: año, género, tipo, bandera, mayúsculas, acentos o el sufijo __N impiden
        # reconocer como publicado un álbum que ya está en el canal.
        pares = [
            ("Abysal - Theogony (2017)",
             "🇵🇱 Abysal - Theogony (2017) • [Technical Death Metal] ⟨FULL ALBUM⟩"),
            ("Twitch Of The Death Nerve - Beset By False Prophets (2021) [EP]",
             "Twitch Of The Death Nerve - Beset By False Prophets (Full Album)"),
            ("Intracranial Butchery - Keepin' It Grind (Reissue) (2025)__2",
             "INTRACRANIAL BUTCHERY - KEEPIN' IT GRIND (REISSUE) (2025) • [Slamming Brutal Death Metal] ⟨ALBUM⟩"),
            ("Crème Brûlée - Écorché (2020)", "🇫🇷 CREME BRULEE - Ecorche (2020) ⟨FULL EP⟩"),
            ("Escuela Grind - Early Recordings 2016-2018 (2019)",
             "Escuela Grind ‎- Early Recordings 2016-2018 (Full Album)"),
            ("Wetwork - New Start Human (2014)", "Wetwork  - New Start Human (Full Album)"),
        ]
        for carpeta, titulo in pares:
            with self.subTest(carpeta=carpeta):
                self.assertIsNotNone(rp.clave(carpeta))
                self.assertEqual(rp.clave(carpeta), rp.clave(titulo))

    def test_letras_no_latinas_cuentan_y_un_nombre_sin_album_no_da_clave(self):
        # Defecto: borrar lo que no es ASCII deja "atramustum|" y cualquier disco de la banda
        # coincidiría con el título viejo "Atra Mustum -  (Full Album)".
        self.assertNotEqual(rp.clave("Atra Mustum - Хаос (2012)"), rp.clave("Atra Mustum - Пустота (2015)"))
        for nombre in ("Atra Mustum -  (Full Album)", "Deleted video", "(Full Album)", "Banda Sola (2020)"):
            with self.subTest(nombre=nombre):
                self.assertIsNone(rp.clave(nombre))


class TestClaveBanda(unittest.TestCase):
    def test_la_carpeta_el_titulo_y_la_linea_de_la_lista_dan_la_misma_clave_de_banda(self):
        # Defecto: año, tipo, género, bandera, mayúsculas, acentos o el sufijo __N hacen que una
        # carpeta de una banda de la lista negra no coincida con la línea de la lista.
        esperada = rp.clave_banda("Prophecy")
        self.assertTrue(esperada)
        for nombre in (
            "PROPHECY - Devastated by Mankind (2026) [EP]",
            "🇩🇪 PROPHECY - Devastated By Mankind (2026) • [Doom Metal] ⟨FULL EP⟩",
            "Prophécy - Otro disco (2020)__2",
            "  prophecy  ",
        ):
            with self.subTest(nombre=nombre):
                self.assertEqual(rp.clave_banda(nombre), esperada)

    def test_la_clave_de_banda_es_la_parte_de_banda_de_la_clave_del_album(self):
        # Defecto: un segundo normalizador con otra forma (otro orden de pasos, otros separadores)
        # daría claves distintas a las del registro para el mismo nombre.
        for nombre in ("Twitch Of The Death Nerve - Beset By False Prophets (2021) [EP]",
                       "Escuela Grind ‎- Early Recordings 2016-2018 (Full Album)",
                       "Atra Mustum - Хаос (2012)"):
            with self.subTest(nombre=nombre):
                self.assertEqual(rp.clave_banda(nombre), rp.clave(nombre).split("|")[0])

    def test_una_banda_que_solo_comparte_el_comienzo_no_coincide(self):
        # Defecto: comparar por prefijo o por "contiene" aparta los álbumes de otras bandas.
        self.assertNotEqual(rp.clave_banda("Prophecy of Doom - Disco (2020)"), rp.clave_banda("Prophecy"))
        self.assertNotEqual(rp.clave_banda("Eternal - Tragedy (2020)"), rp.clave_banda("Eternal Tragedy"))

    def test_sin_nombre_no_hay_clave_de_banda(self):
        # Defecto: una clave vacía en la lista negra coincidiría con cualquier banda sin letras.
        for nombre in ("", "   ", None, "!!!", "(2020)"):
            with self.subTest(nombre=nombre):
                self.assertIsNone(rp.clave_banda(nombre))


class TestLeerBandas(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ruta = self.tmp / "lista_negra_bandas.txt"

    def test_lee_una_banda_por_linea_e_ignora_comentarios_y_lineas_en_blanco(self):
        # Defecto: el encabezado "# ..." o una línea en blanco entran como banda; una línea con
        # espacios o con fin de línea de Windows no coincide.
        self.ruta.write_text(
            "# Lista negra de bandas\n# otra nota\n\nProphecy\r\n  Absurdeity  \n\t\nEternal Tragedy\n",
            encoding="utf-8",
        )

        self.assertEqual(
            rp.leer_bandas(self.ruta),
            {rp.clave_banda("Prophecy"), rp.clave_banda("Absurdeity"), rp.clave_banda("Eternal Tragedy")},
        )

    def test_sin_archivo_no_hay_bandas(self):
        # Defecto: un archivo que todavía no existe frena `buscar` en lugar de no apartar nada.
        self.assertEqual(rp.leer_bandas(self.tmp / "no_existe.txt"), set())


class TestRegistro(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ruta = self.tmp / "ya_publicados.tsv"

    def test_un_titulo_censurado_coincide_solo_con_el_mismo_largo_y_las_letras_visibles(self):
        # Defecto: sin comodín, los 286 títulos censurados del canal nunca coinciden; con un
        # comodín flojo, coincide cualquier banda que empiece con la misma letra.
        rp.anotar(self.ruta, [
            ("🇪🇨 A*** Arcade - The Temple of Madness (2015) • [Goregrind] ⟨FULL ALBUM⟩", "canal", "vid00000001"),
        ])
        registro = rp.leer(self.ruta)
        fila = registro.buscar("Anal Arcade - The Temple Of Madness (2015)")
        self.assertEqual((fila["origen"], fila["video_id"]), ("canal", "vid00000001"))
        for otro in ("Anal Arcadia - The Temple Of Madness (2015)", "Oral Arcade - The Temple Of Madness (2015)",
                     "Anal Arcade - The Temple Of Sadness (2015)"):
            with self.subTest(otro=otro):
                self.assertIsNone(registro.buscar(otro))

    def test_anotar_agrega_al_final_y_saltea_lo_que_no_da_clave(self):
        # Defecto: reescribir el archivo pierde lo anotado antes, y un tabulador en el nombre
        # corre las columnas.
        self.assertEqual(rp.anotar(self.ruta, [("Banda - Uno (2020)", "eliminar", "")]), 1)
        agregadas = rp.anotar(self.ruta, [
            ("Banda\t - Dos (2021)", "subir", "vid00000002"),
            ("Deleted video", "canal", "vid00000003"),
        ])
        self.assertEqual(agregadas, 1)
        registro = rp.leer(self.ruta)
        self.assertEqual(registro.buscar("Banda - Uno (2020)")["origen"], "eliminar")
        self.assertEqual(registro.buscar("Banda - Dos (2021)")["video_id"], "vid00000002")
        self.assertEqual(registro.video_ids(), {"vid00000002"})

    def test_un_registro_cortado_o_con_separadores_raros_se_sigue_leyendo(self):
        # Defecto: una escritura cortada a mitad de una letra deja bytes que no son UTF-8 y leer
        # lanza, así que ningún render automático arranca; y un título con U+2028 se parte en
        # dos líneas al leer y no coincide nunca.
        rp.anotar(self.ruta, [("Banda - Uno Parte (2020)", "canal", "vid00000001")])
        with open(self.ruta, "ab") as archivo:
            archivo.write("banda|do\tBanda - Dó".encode("utf-8")[:-1])  # la "ó" sin su segundo byte

        registro = rp.leer(self.ruta)

        self.assertEqual(registro.buscar("Banda - Uno Parte (2020)")["video_id"], "vid00000001")

    def test_sin_archivo_no_hay_nada_publicado(self):
        self.assertIsNone(rp.leer(self.tmp / "no_existe.tsv").buscar("Banda - Uno (2020)"))


if __name__ == "__main__":
    unittest.main()
