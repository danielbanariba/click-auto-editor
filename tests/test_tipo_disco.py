"""
"Full Album" o "Full EP" en el título: lo decide DeathGrind y, si no conoce
el disco, el sufijo [EP] que pone el scraper en el nombre de la carpeta.
"""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAY_FFMPEG = shutil.which("ffmpeg") is not None


def cargar_subida():
    spec = importlib.util.spec_from_file_location("subir_video_api_tipo", PROJECT_ROOT / "3. subir_video_API.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


subida = cargar_subida()


class TestTituloDeBusqueda(unittest.TestCase):
    def titulo(self, nombre, release=None):
        with mock.patch.object(subida, "get_repertorio_csv_path", return_value=Path("repertorio.csv")), mock.patch.object(
            subida, "buscar_release_en_csv", return_value=release
        ):
            return subida.build_preview_title(Path(nombre), None)

    def test_ep_que_deathgrind_no_conoce_sale_como_full_ep(self):
        self.assertEqual(
            self.titulo("13Rituals - Rituals Of Punishment (2009) [EP]"),
            "13Rituals - Rituals Of Punishment (Full EP)",
        )

    def test_el_tipo_de_deathgrind_gana_sobre_el_nombre_de_la_carpeta(self):
        self.assertEqual(
            self.titulo("Banda - Disco (2020) [EP]", release={"type": [1]}),
            "Banda - Disco (Full Album)",
        )

    def test_sin_sufijo_ni_deathgrind_sigue_siendo_full_album(self):
        self.assertEqual(self.titulo("Banda - Disco (2020)"), "Banda - Disco (Full Album)")


@unittest.skipUnless(HAY_FFMPEG, "requiere ffmpeg")
class TestTituloDeSubida(unittest.TestCase):
    def test_ep_que_deathgrind_no_conoce_se_sube_como_full_ep(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        carpeta = tmp / "13Rituals - Rituals Of Punishment (2009) [EP]"
        carpeta.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1",
             str(carpeta / "01-13rituals-abatement.mp3")],
            check=True,
        )
        with mock.patch.object(subida, "get_repertorio_csv_path", return_value=None):
            datos = subida.procesar_carpeta(carpeta, None, [], {}, None)
        self.assertIn("FULL EP", datos["title"].upper())
        self.assertNotIn("FULL ALBUM", datos["title"].upper())


if __name__ == "__main__":
    unittest.main()
