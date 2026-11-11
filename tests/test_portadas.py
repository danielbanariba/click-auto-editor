import tempfile
import unittest
from pathlib import Path

from portadas import elegir_portada


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
                   "DGC - PURE HATRED_shadow.png", "Cover_2023.jpg", "Cover_2018.jpg", "notas.txt")
        self.assertEqual(elegir_portada(self.carpeta), self.carpeta / "Cover_2018.jpg")

    def test_cover_gana_aunque_otra_imagen_vaya_antes_por_nombre(self):
        self.crear("Arte.jpg", "DGC.jpg", "cover.png")
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


if __name__ == "__main__":
    unittest.main()
