"""Lista negra de reclamantes: qué reclamantes de un video coinciden con ella."""
import unittest

import pipeline_agente as pa


class TestReclamantesEnListaNegra(unittest.TestCase):
    def test_coincide_sin_distinguir_mayusculas_y_como_parte_del_nombre(self):
        # Defecto: comparar el nombre completo y con mayúsculas dejaba pasar a UMG cuando
        # Studio lo muestra junto a otro sello o con "en nombre de".
        reclamantes = [
            "Believe Music",
            "Believe Music, UMG",
            "UMG | en nombre de Artists & Acts",
            "en nombre de Church Road Records",
        ]
        self.assertEqual(
            pa.reclamantes_en_lista_negra(reclamantes, ["umg", "church road records"]),
            [
                "Believe Music, UMG",
                "UMG | en nombre de Artists & Acts",
                "en nombre de Church Road Records",
            ],
        )

    def test_sin_coincidencias_o_sin_lista_no_marca_nada(self):
        self.assertEqual(pa.reclamantes_en_lista_negra(["Believe Music"], ["umg"]), [])
        self.assertEqual(pa.reclamantes_en_lista_negra(["Believe Music"], []), [])
        self.assertEqual(pa.reclamantes_en_lista_negra([], ["umg"]), [])

    def test_la_lista_negra_del_proyecto_se_carga_desde_su_ruta_por_defecto(self):
        # Defecto: una ruta mal armada devolvería una lista vacía y no avisaría nunca.
        self.assertTrue(pa.cargar_lista_negra())


if __name__ == "__main__":
    unittest.main()
