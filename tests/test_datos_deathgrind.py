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


class SesionPorCursor:
    """Responde la página que pide el cursor, como /posts/search y /posts/filter.

    Pasado `limite` pedidos falla: un bucle sin guarda no cuelga la prueba, la pone en rojo.
    """

    def __init__(self, paginas, limite=10):
        self.paginas = paginas
        self.pedidas = []
        self.limite = limite

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.pedidas.append((url.rsplit("/api", 1)[-1], params))
        if len(self.pedidas) > self.limite:
            raise AssertionError(f"más de {self.limite} pedidos: el bucle no corta")
        for (final, cursor), datos in self.paginas.items():
            if url.endswith(final) and params.get("cursor") == cursor:
                return Respuesta(200, datos)
        return Respuesta(404)


def post_de(post_id, banda, album, **campos):
    return {"postId": post_id, "bands": [{"name": banda}], "album": album, **campos}


def test_la_busqueda_sigue_el_cursor_hasta_la_pagina_del_disco(uploader):
    sesion = SesionPorCursor({
        ("/posts/search", None): {"posts": [post_de(1, "Otra", "Disco")], "hasMore": True, "nextCursor": "c1"},
        ("/posts/search", "c1"): {"posts": [post_de(2, "Banda", "Disco")], "hasMore": False, "nextCursor": None},
    })

    encontrado = uploader.buscar_release_en_api(sesion, "Banda", "Disco", GENEROS)

    assert encontrado["postId"] == 2
    assert sesion.pedidas[1] == ("/posts/search", {"q": "Banda Disco", "cursor": "c1"})


def test_la_busqueda_por_genero_sigue_el_cursor(uploader):
    sesion = SesionPorCursor({
        ("/posts/search", None): {"posts": [], "hasMore": False, "nextCursor": None},
        ("/posts/filter", None): {"posts": [post_de(1, "Otra", "Disco")], "hasMore": True, "nextCursor": "c1"},
        ("/posts/filter", "c1"): {"posts": [post_de(2, "Banda", "Disco")], "hasMore": False, "nextCursor": None},
    })

    encontrado = uploader.buscar_release_en_api(sesion, "Banda", "Disco", GENEROS, allow_genre_fallback=True)

    assert encontrado["postId"] == 2
    assert sesion.pedidas[-1] == ("/posts/filter", {"genres": 13, "cursor": "c1"})


SIN_RESULTADOS = {"posts": [], "hasMore": False, "nextCursor": None}


def paginas_de(ruta, paginas):
    """Las páginas de `ruta` por cursor; /posts/filter solo se pide tras una búsqueda sin resultados."""
    previas = {("/posts/search", None): SIN_RESULTADOS} if ruta == "/posts/filter" else {}
    return previas | {(ruta, cursor): pagina for cursor, pagina in paginas.items()}


def pedidas_de(sesion, ruta):
    return [params for url, params in sesion.pedidas if url == ruta]


def buscar_con_solo_las_guardas_del_cursor(uploader, monkeypatch, sesion):
    # Sin caché ni tope de páginas, solo la guarda del cursor puede cortar el bucle.
    monkeypatch.setenv("DEATHGRIND_CACHE", "0")
    monkeypatch.setenv("DEATHGRIND_SEARCH_PAGES", "0")
    monkeypatch.setenv("DEATHGRIND_GENRE_PAGES", "0")
    return uploader.buscar_release_en_api(sesion, "Banda", "Disco", GENEROS, allow_genre_fallback=True)


@pytest.mark.parametrize("ruta", ["/posts/search", "/posts/filter"])
def test_la_busqueda_corta_si_el_servidor_repite_el_cursor(uploader, monkeypatch, ruta):
    # Si no cortara, pediría a DeathGrind sin fin: gasta cuota y termina en rate limit.
    trabada = {"posts": [post_de(1, "Otra", "Disco")], "hasMore": True, "nextCursor": "c1"}
    sesion = SesionPorCursor(paginas_de(ruta, {None: trabada, "c1": trabada}))

    encontrado = buscar_con_solo_las_guardas_del_cursor(uploader, monkeypatch, sesion)

    assert encontrado is None
    assert len(pedidas_de(sesion, ruta)) == 2


@pytest.mark.parametrize("ruta", ["/posts/search", "/posts/filter"])
def test_la_busqueda_corta_si_hasmore_viene_sin_cursor(uploader, monkeypatch, ruta):
    # Sin `nextCursor` no hay página siguiente que pedir: volver a pedir la primera la repetiría.
    sin_cursor = {"posts": [post_de(1, "Otra", "Disco")], "hasMore": True, "nextCursor": None}
    sesion = SesionPorCursor(paginas_de(ruta, {None: sin_cursor}))

    encontrado = buscar_con_solo_las_guardas_del_cursor(uploader, monkeypatch, sesion)

    assert encontrado is None
    assert len(pedidas_de(sesion, ruta)) == 1


def test_la_busqueda_para_en_el_tope_de_paginas_aunque_los_cursores_sean_nuevos(uploader, monkeypatch):
    monkeypatch.setenv("DEATHGRIND_CACHE", "0")
    monkeypatch.setenv("DEATHGRIND_SEARCH_PAGES", "2")
    otra = post_de(1, "Otra", "Disco")
    cursores = [None] + [f"c{n}" for n in range(1, 9)]
    sesion = SesionPorCursor({
        ("/posts/search", cursor): {"posts": [otra], "hasMore": True, "nextCursor": f"c{n}"}
        for n, cursor in enumerate(cursores, start=1)
    })

    encontrado = uploader.buscar_release_en_api(sesion, "Banda", "Disco", GENEROS)

    assert encontrado is None
    assert len(sesion.pedidas) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requiere ffmpeg")
def test_un_ep_de_deathgrind_se_sube_como_full_ep_aunque_la_carpeta_no_lo_diga(
    uploader, monkeypatch, tmp_path
):
    # La API da el tipo en `releaseTypes`, y el EP es el 9.
    monkeypatch.setattr(uploader, "get_repertorio_csv_path", lambda: None)
    monkeypatch.setattr(uploader, "load_playlist_links_cache", lambda: {})
    ep = post_de(1, "Banda", "Disco", releaseTypes=[9], genres=[13])
    monkeypatch.setattr(uploader, "esperar_release_en_api", lambda *_args: ep)

    datos = uploader.procesar_carpeta(album_con_audio(tmp_path), SesionFalsa({}), GENEROS, dict(GENEROS), None)

    assert "FULL EP" in datos["title"].upper()
    assert "FULL ALBUM" not in datos["title"].upper()
