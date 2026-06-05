import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "16. fusionar_playlists_duplicadas.py"
SPEC = importlib.util.spec_from_file_location("fusionar", MODULE_PATH)
fusionar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusionar)


def _pl(pid, title, count=0):
    return {"id": pid, "title": title, "count": count}


class TestAgrupar(unittest.TestCase):
    def test_agrupa_por_codigo_pais(self):
        playlists = [
            _pl("a", "Germany"),
            _pl("b", "Alemania"),
            _pl("c", "Brazil"),
            _pl("d", "Some Band"),
        ]
        grupos = fusionar.agrupar_paises_duplicados(playlists)
        self.assertIn("DE", grupos)
        self.assertEqual(len(grupos["DE"]), 2)
        # Brazil solo aparece una vez -> no es duplicado
        self.assertNotIn("BR", grupos)
        # una banda no debe agruparse como pais
        self.assertNotIn(None, grupos)

    def test_ignora_no_paises(self):
        playlists = [_pl("a", "Cannibal Corpse"), _pl("b", "Death Metal")]
        self.assertEqual(fusionar.agrupar_paises_duplicados(playlists), {})


class TestElegirDestino(unittest.TestCase):
    def test_prefiere_nombre_canonico_existente(self):
        pls = [_pl("a", "Alemania", 5), _pl("b", "Germany", 2)]
        destino, nombre, extras, rename = fusionar.elegir_destino("DE", pls)
        self.assertEqual(nombre, "Germany")
        self.assertEqual(destino["id"], "b")  # el que ya se llama Germany
        self.assertFalse(rename)
        self.assertEqual([e["id"] for e in extras], ["a"])

    def test_sin_canonico_elige_mas_videos_y_renombra(self):
        pls = [_pl("a", "Alemania", 10), _pl("b", "Deutschland", 3)]
        destino, nombre, extras, rename = fusionar.elegir_destino("DE", pls)
        self.assertEqual(destino["id"], "a")  # mas videos
        self.assertTrue(rename)
        self.assertEqual(nombre, "Germany")

    def test_estimar_cuota_suma_inserts_y_deletes(self):
        grupos = {"DE": [_pl("a", "Germany", 0), _pl("b", "Alemania", 2)]}
        # mover 2 videos (2*50) + borrar 1 playlist (50) = 150
        self.assertEqual(fusionar.estimar_cuota(grupos), 150)


if __name__ == "__main__":
    unittest.main()
