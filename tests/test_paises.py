import unittest

import paises


class TestCanonicalizarPais(unittest.TestCase):
    def test_resuelve_ingles_a_codigo(self):
        self.assertEqual(paises.canonicalizar_pais("Germany"), "DE")
        self.assertEqual(paises.canonicalizar_pais("New Zealand"), "NZ")

    def test_resuelve_espanol_a_codigo(self):
        self.assertEqual(paises.canonicalizar_pais("Alemania"), "DE")
        self.assertEqual(paises.canonicalizar_pais("Nueva Zelanda"), "NZ")
        self.assertEqual(paises.canonicalizar_pais("España"), "ES")

    def test_ignora_acentos_y_mayusculas(self):
        self.assertEqual(paises.canonicalizar_pais("HUNGRÍA"), "HU")
        self.assertEqual(paises.canonicalizar_pais("hungria"), "HU")
        self.assertEqual(paises.canonicalizar_pais("Japón"), "JP")

    def test_alias_y_gentilicios(self):
        self.assertEqual(paises.canonicalizar_pais("Russian"), "RU")
        self.assertEqual(paises.canonicalizar_pais("USA"), "US")
        self.assertEqual(paises.canonicalizar_pais("UK"), "GB")
        self.assertEqual(paises.canonicalizar_pais("Czech Republic"), "CZ")
        self.assertEqual(paises.canonicalizar_pais("Republica Checa"), "CZ")

    def test_no_usa_fuzzy_peligroso(self):
        # "UK" jamas debe resolver a Uganda (UG) via fuzzy
        self.assertNotEqual(paises.canonicalizar_pais("UK"), "UG")

    def test_codigo_iso_directo(self):
        self.assertEqual(paises.canonicalizar_pais("US"), "US")
        self.assertEqual(paises.canonicalizar_pais("de"), "DE")

    def test_valor_no_pais_devuelve_none(self):
        self.assertIsNone(paises.canonicalizar_pais("unknown"))
        self.assertIsNone(paises.canonicalizar_pais("Mars"))
        self.assertIsNone(paises.canonicalizar_pais(""))
        self.assertIsNone(paises.canonicalizar_pais(None))


class TestNombrePaisEn(unittest.TestCase):
    def test_devuelve_nombre_ingles_canonico(self):
        self.assertEqual(paises.nombre_pais_en("Alemania"), "Germany")
        self.assertEqual(paises.nombre_pais_en("Republica Checa"), "Czechia")
        self.assertEqual(paises.nombre_pais_en("Russian"), "Russia")
        self.assertEqual(paises.nombre_pais_en("Nueva Zelanda"), "New Zealand")

    def test_idempotente(self):
        once = paises.nombre_pais_en("España")
        self.assertEqual(paises.nombre_pais_en(once), once)

    def test_conserva_original_si_no_resuelve(self):
        self.assertEqual(paises.nombre_pais_en("Mars"), "Mars")
        self.assertEqual(paises.nombre_pais_en("unknown"), "unknown")

    def test_distintas_variantes_colapsan_al_mismo_nombre(self):
        variantes = ["Czech Republic", "Republica Checa", "Chequia", "Czechia", "CZ"]
        resultados = {paises.nombre_pais_en(v) for v in variantes}
        self.assertEqual(resultados, {"Czechia"})


if __name__ == "__main__":
    unittest.main()
