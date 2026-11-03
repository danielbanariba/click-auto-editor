"""
Datos de DeathGrind que llegan a la descripción del video: el país.

Ejecutar: pytest tests/test_datos_deathgrind.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "3. subir_video_API.py"


@pytest.fixture
def uploader(monkeypatch):
    spec = importlib.util.spec_from_file_location("datos_deathgrind_uploader", SCRIPT_PATH)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["datos_deathgrind_uploader"] = modulo
    spec.loader.exec_module(modulo)
    monkeypatch.setattr(modulo.time, "sleep", lambda _segundos: None)
    return modulo


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
