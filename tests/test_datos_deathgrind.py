"""
Datos de DeathGrind que llegan a la descripción del video: la prueba previa, los links y el país.

Ejecutar: pytest tests/test_datos_deathgrind.py
"""

import importlib.util
import shutil
import subprocess
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
        self.pedidas = []

    def get(self, url, params=None, timeout=None):
        self.pedidas.append(url)
        for final, respuesta in self.respuestas.items():
            if url.endswith(final):
                return respuesta
        return Respuesta(404)


def pidio_links_de_descarga(sesion):
    return [url for url in sesion.pedidas if url.endswith("/links")]


def test_la_prueba_no_pide_links_de_descarga(uploader):
    # Los links de descarga gastan los créditos de la cuenta, que el scraper necesita.
    sesion = SesionFalsa({
        "/posts/filter": Respuesta(200, {"posts": [{"postId": 1, "bands": [{"bandId": 2}]}]}),
        "/bands/2/discography": Respuesta(200, {"posts": []}),
        "/posts/1/links": Respuesta(403, {"errorCode": "DOWNLOAD_CREDITS_EXHAUSTED"}),
    })

    assert uploader.deathgrind_smoke_test(sesion, GENEROS) is True
    assert pidio_links_de_descarga(sesion) == []


def test_sin_posts_la_prueba_sigue_apagando_deathgrind(uploader):
    sesion = SesionFalsa({"/posts/filter": Respuesta(500)})

    assert uploader.deathgrind_smoke_test(sesion, GENEROS) is False


def album_con_audio(tmp_path):
    carpeta = tmp_path / "Banda - Disco (2020)"
    carpeta.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1",
         str(carpeta / "01-banda-tema.mp3")],
        check=True,
    )
    return carpeta


def subir_con_deathgrind(uploader, monkeypatch, carpeta):
    monkeypatch.setattr(uploader, "get_repertorio_csv_path", lambda: None)
    monkeypatch.setattr(uploader, "load_playlist_links_cache", lambda: {})
    repertorio = [{"band": "Banda", "album": "Disco", "postId": 1, "bandId": 2, "year": 2020}]
    sesion = SesionFalsa({
        "/bands/2": Respuesta(200, {"band": {
            "relatedLinks": [{"url": "https://banda.bandcamp.com", "name": "Bandcamp"}],
        }}),
        "/posts/1/links": Respuesta(200, {"links": [{"url": "https://mega.nz/#!disco"}]}),
    })
    datos = uploader.procesar_carpeta(carpeta, sesion, GENEROS, {}, repertorio)
    return datos, sesion


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requiere ffmpeg")
def test_la_subida_no_pide_links_de_descarga(uploader, monkeypatch, tmp_path):
    _datos, sesion = subir_con_deathgrind(uploader, monkeypatch, album_con_audio(tmp_path))

    assert pidio_links_de_descarga(sesion) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requiere ffmpeg")
def test_la_descripcion_sigue_con_el_bandcamp_de_la_banda(uploader, monkeypatch, tmp_path):
    datos, _sesion = subir_con_deathgrind(uploader, monkeypatch, album_con_audio(tmp_path))

    assert "https://banda.bandcamp.com" in datos["description"]


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
