"""
Datos de DeathGrind que llegan a la descripción del video: la prueba previa y el país.

Ejecutar: pytest tests/test_datos_deathgrind.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "3. subir_video_API.py"
GENEROS = [(13, "Grindcore")]


@pytest.fixture
def uploader(monkeypatch):
    spec = importlib.util.spec_from_file_location("datos_deathgrind_uploader", SCRIPT_PATH)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["datos_deathgrind_uploader"] = modulo
    spec.loader.exec_module(modulo)
    monkeypatch.setattr(modulo.time, "sleep", lambda _segundos: None)
    return modulo


class Respuesta:
    def __init__(self, status_code, datos=None):
        self.status_code = status_code
        self.datos = datos

    def json(self):
        return self.datos


class SesionFalsa:
    """Responde según el final de la URL, como la API de DeathGrind."""

    def __init__(self, respuestas):
        self.respuestas = respuestas

    def get(self, url, params=None, timeout=None):
        for final, respuesta in self.respuestas.items():
            if url.endswith(final):
                return respuesta
        return Respuesta(404)


def test_sin_creditos_para_links_la_prueba_no_apaga_deathgrind(uploader):
    sesion = SesionFalsa({
        "/posts/filter": Respuesta(200, {"posts": [{"postId": 1, "bands": [{"bandId": 2}]}]}),
        "/bands/2/discography": Respuesta(200, {"posts": []}),
        "/posts/1/links": Respuesta(403, {"errorCode": "DOWNLOAD_CREDITS_EXHAUSTED"}),
    })

    assert uploader.deathgrind_smoke_test(sesion, GENEROS) is True


def test_sin_posts_la_prueba_sigue_apagando_deathgrind(uploader):
    sesion = SesionFalsa({"/posts/filter": Respuesta(500)})

    assert uploader.deathgrind_smoke_test(sesion, GENEROS) is False


def pais_con_bandera(uploader, datos_de_la_banda):
    pais = uploader.normalizar_pais(uploader.extraer_pais_desde_data(datos_de_la_banda))
    return uploader.resolve_country_meta(pais)


def test_el_pais_sale_de_countries_y_no_de_la_region(uploader):
    banda = {"countries": ["PL"], "location": "Wrocław, Lower Silesia"}

    meta = pais_con_bandera(uploader, banda)

    assert (meta["name"], meta["flag"]) == ("Poland", "🇵🇱")


def test_sin_countries_el_pais_sale_de_location(uploader):
    banda = {"location": "São Paulo, Brazil"}

    meta = pais_con_bandera(uploader, banda)

    assert (meta["name"], meta["flag"]) == ("Brazil", "🇧🇷")
