import tempfile
import unittest
from pathlib import Path

from portadas import elegir_portada, reemplazar_portada


class TestElegirPortada(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.carpeta = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def crear(self, *nombres):
        for nombre in nombres:
            (self.carpeta / nombre).write_bytes(b"x")

    def test_sin_cover_la_portada_es_otra_imagen_que_no_sea_la_sombra(self):
        self.assertIsNone(elegir_portada(self.carpeta))
        self.crear("aaa_shadow.png")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "aaa_shadow.png")
        self.crear("Arte.JPG")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "Arte.JPG")

    def test_sin_cover_salta_las_propagandas_de_deathgrind_y_elige_la_primera_por_nombre(self):
        self.crear("DGC - PURE HATRED.jpg", "DGC Pure Hatred.jpeg", "DGC.jpg", "DGCx.jpg",
                   "DGC - PURE HATRED_shadow.png", "Zeta.jpg", "Marcio Dentro.jpg", "notas.txt")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "Marcio Dentro.jpg")

    def test_cover_gana_aunque_otra_imagen_con_cover_vaya_antes_por_nombre(self):
        self.crear("Album - Cover.jpg", "DGC.jpg", "cover.png")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "cover.png")

    def test_la_contraportada_y_las_miniaturas_no_son_la_portada(self):
        self.crear("AlbumArtSmall.jpg", "Back.jpg", "Booklet.jpg", "cd.jpg", "Folder.jpg")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "Folder.jpg")

    def test_cd_en_el_nombre_no_tapa_la_imagen_que_dice_front(self):
        self.crear("00-banda-disco-cd-flac-2016-back.jpg", "00-banda-disco-cd-flac-2016-front.jpg",
                   "Arte.jpg")
        self.assertEqual(elegir_portada(self.carpeta),
                         self.carpeta / "00-banda-disco-cd-flac-2016-front.jpg")

    def test_prefiere_la_imagen_que_se_llama_front_o_cover_aunque_otra_vaya_antes(self):
        self.crear("55.jpg", "Grumo - Back Cover.jpg", "Grumo - Front.jpg")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "Grumo - Front.jpg")

    def test_si_solo_hay_propagandas_no_hay_portada_y_el_render_la_saca_del_audio(self):
        self.crear("DGC - PURE HATRED.jpg", "DGC.jpg", "DGC_shadow.png")
        self.assertIsNone(elegir_portada(self.carpeta))


class TestReemplazarPortada(unittest.TestCase):
    """La portada censurada tiene que ser la que elige `elegir_portada`, y la original se guarda."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.carpeta = base / "Banda - Disco (2020)"
        self.guardar_en = base / "originales"
        self.carpeta.mkdir()
        self.censurada = base / "candidata.png"
        self.censurada.write_bytes(b"censurada")

    def tearDown(self):
        self._tmp.cleanup()

    def crear(self, nombre, contenido=b"original"):
        (self.carpeta / nombre).write_bytes(contenido)

    def test_la_censurada_gana_a_cualquier_cover_y_a_la_portada_que_elegia_el_render(self):
        # Defecto: dejar un cover.jpg junto al cover.png nuevo hace que el render siga eligiendo
        # el original sin censura (empatan en prioridad y gana el nombre que va antes).
        self.crear("cover.jpg", b"cover jpg")
        self.crear("cover.png", b"cover png")
        self.crear("Arte.jpg", b"arte")
        self.crear("01. pista.flac", b"audio")

        reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        elegida = elegir_portada(self.carpeta)
        self.assertEqual(elegida.read_bytes(), b"censurada")
        self.assertEqual(elegida.name, "cover.png")
        # Lo que no compite con la portada se queda donde estaba.
        self.assertEqual((self.carpeta / "01. pista.flac").read_bytes(), b"audio")
        self.assertEqual((self.carpeta / "Arte.jpg").read_bytes(), b"arte")

    def test_la_portada_original_se_guarda_entera_y_nunca_se_borra(self):
        # Defecto: reemplazar la imagen en su lugar pierde la original para siempre.
        self.crear("Front.jpg", b"portada original")
        self.crear("Back.jpg", b"contraportada")

        guardadas = reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        self.assertEqual([p.read_bytes() for p in guardadas], [b"portada original"])
        self.assertTrue(all(p.parent == self.guardar_en for p in guardadas))
        self.assertFalse((self.carpeta / "Front.jpg").exists())
        # La contraportada no era la portada: sigue donde estaba.
        self.assertEqual((self.carpeta / "Back.jpg").read_bytes(), b"contraportada")
        # Lo guardado está fuera de la carpeta: no puede elegirse como portada del álbum.
        self.assertEqual(elegir_portada(self.carpeta).name, "cover.png")

    def test_todo_cover_se_guarda_aunque_haya_varios(self):
        # Defecto: guardar solo el que elegía el render y borrar el otro cover.* lo perdía.
        self.crear("cover.jpg", b"cover jpg")
        self.crear("cover.png", b"cover png")

        guardadas = reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        self.assertEqual(sorted(p.read_bytes() for p in guardadas), [b"cover jpg", b"cover png"])

    def test_las_sombras_de_la_original_se_apartan_porque_el_render_las_reusa(self):
        # Defecto: `ensure_shadow_cover` reusa un cover_shadow.png ya existente: el video
        # llevaría la sombra de la portada sin censurar.
        self.crear("cover_shadow.png", b"sombra sin censura")
        self.crear("cover.jpg", b"original")

        reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        self.assertFalse((self.carpeta / "cover_shadow.png").exists())
        self.assertIn(b"sombra sin censura", [p.read_bytes() for p in self.guardar_en.iterdir()])

    def test_dos_reemplazos_del_mismo_nombre_no_se_pisan_en_lo_guardado(self):
        # Defecto: el segundo reemplazo (el mismo álbum devuelto con Deshacer) pisaba la original.
        self.crear("cover.jpg", b"primera")
        reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)
        self.crear("cover.jpg", b"segunda")
        reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        self.assertIn(b"primera", [p.read_bytes() for p in self.guardar_en.iterdir()])
        self.assertIn(b"segunda", [p.read_bytes() for p in self.guardar_en.iterdir()])

    def test_un_album_sin_ninguna_imagen_igual_queda_con_la_censurada_de_portada(self):
        self.crear("01. pista.flac", b"audio")

        guardadas = reemplazar_portada(self.carpeta, self.censurada, self.guardar_en)

        self.assertEqual(guardadas, [])
        self.assertEqual(elegir_portada(self.carpeta).read_bytes(), b"censurada")


if __name__ == "__main__":
    unittest.main()
