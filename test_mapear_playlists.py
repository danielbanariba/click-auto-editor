import importlib.util
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "12. mapear_playlists.py"
SPEC = importlib.util.spec_from_file_location("mapear_playlists", MODULE_PATH)
mapear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapear)


class TestSplitCountriesCanonicaliza(unittest.TestCase):
    def test_variantes_de_idioma_colapsan(self):
        # "Republica Checa" y "Czech Republic" deben dar UN solo pais en ingles
        result = mapear.split_countries("Republica Checa, Czech Republic, Czechia")
        self.assertEqual(result, ["Czechia"])

    def test_ingles_y_espanol_mismo_pais(self):
        self.assertEqual(mapear.split_countries("Nueva Zelanda"), ["New Zealand"])
        self.assertEqual(mapear.split_countries("Alemania"), ["Germany"])

    def test_conserva_valor_desconocido(self):
        self.assertEqual(mapear.split_countries("Atlantis"), ["Atlantis"])

    def test_lista_de_paises_mixta(self):
        result = mapear.split_countries(["Japón", "Japan", "Brasil"])
        self.assertEqual(result, ["Japan", "Brazil"])


class TestLock(unittest.TestCase):
    def test_segunda_instancia_no_obtiene_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "test.lock"
            primero = mapear.adquirir_lock(lock_path)
            self.assertTrue(primero)

            # Proceso independiente (spawn, sin heredar descriptores del padre):
            # no debe poder tomar el mismo lock mientras este proceso lo tiene.
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=_intentar_lock, args=(str(lock_path), q))
            p.start()
            p.join(15)
            self.assertFalse(q.get(timeout=5))

    def test_lock_es_idempotente_en_mismo_proceso(self):
        # main() se re-llama al rotar credenciales: el mismo proceso debe poder
        # "tomar" el lock varias veces sin auto-bloquearse.
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "idem.lock"
            mapear._lock_handle = None
            try:
                self.assertTrue(mapear.adquirir_lock(lock_path))
                self.assertTrue(mapear.adquirir_lock(lock_path))
                self.assertTrue(mapear.adquirir_lock(lock_path))
            finally:
                mapear._lock_handle = None


def _intentar_lock(path, q):
    import importlib.util

    spec = importlib.util.spec_from_file_location("mp2", MODULE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    q.put(m.adquirir_lock(Path(path)))


if __name__ == "__main__":
    unittest.main()
