import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "13. filtrar_portadas.py"
SPEC = importlib.util.spec_from_file_location("filtrar_portadas", MODULE_PATH)
filtrar_portadas = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(filtrar_portadas)


class FiltrarPortadasTests(unittest.TestCase):
    def test_should_flag_cadaver_real_positive(self):
        thresholds = filtrar_portadas.CadaverThresholds(
            graphic_threshold=0.14,
            photo_threshold=0.58,
            combined_threshold=0.12,
        )

        flagged, score, _reason = filtrar_portadas.should_flag_cadaver_real(
            violence_graphic=0.44,
            violence=0.81,
            photo_score=0.77,
            thresholds=thresholds,
        )

        self.assertTrue(flagged)
        self.assertGreaterEqual(score, thresholds.combined_threshold)

    def test_should_flag_cadaver_real_rejects_illustration_like_case(self):
        thresholds = filtrar_portadas.CadaverThresholds(
            graphic_threshold=0.14,
            photo_threshold=0.58,
            combined_threshold=0.12,
        )

        flagged, score, _reason = filtrar_portadas.should_flag_cadaver_real(
            violence_graphic=0.51,
            violence=0.86,
            photo_score=0.31,
            thresholds=thresholds,
        )

        self.assertFalse(flagged)
        self.assertLess(score, 0.20)

    def test_find_cover_image_prefers_cover_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "zeta.jpg").write_bytes(b"fake")
            expected = tmp_path / "cover.jpg"
            expected.write_bytes(b"fake")

            found = filtrar_portadas.find_cover_image(tmp_path)

            self.assertEqual(found, expected)

    def test_parse_exts_normalizes_values(self):
        parsed = filtrar_portadas.parse_exts("PNG, .JPG, jpeg")
        self.assertEqual(parsed, {".png", ".jpg", ".jpeg"})

    def test_resolve_cadaver_backend_prefers_ollama(self):
        with patch.object(filtrar_portadas, "ollama_is_available", return_value=True):
            backend = filtrar_portadas.resolve_cadaver_backend(
                backend="auto",
                ollama_host="http://localhost:11434",
                timeout=30,
                api_key=None,
                api_key_env="OPENAI_API_KEY",
                ollama_api_key=None,
            )

        self.assertEqual(backend, "ollama")

    def test_scan_cover_path_skips_when_no_backend_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "cover.jpg"
            image_path.write_bytes(b"fake")

            with patch.object(
                filtrar_portadas, "ollama_is_available", return_value=False
            ), patch.object(
                filtrar_portadas, "load_openai_api_key", return_value=None
            ):
                result = filtrar_portadas.scan_cover_path(
                    image_path=image_path,
                    policy="cadaver-real",
                    backend="auto",
                    api_key_env="OPENAI_API_KEY",
                    timeout=1,
                )

        self.assertEqual(result["status"], "skip")
        self.assertFalse(result["flagged"])


if __name__ == "__main__":
    unittest.main()
