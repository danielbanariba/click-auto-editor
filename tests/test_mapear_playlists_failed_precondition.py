"""Pruebas de 12. mapear_playlists.py contra el 400 failedPrecondition de YouTube.

Nunca llaman a la API: el cliente es un doble local que lanza HttpError con el
cuerpo real del journal (mapear_playlists.service, 2026-11-17).

Por defecto cargan el script del repo. MAPEAR_MODULE_PATH apunta a otra copia
(por ejemplo el respaldo previo al cambio, para ver el RED).
"""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httplib2
from googleapiclient.errors import HttpError

DEFAULT_MODULE = Path(__file__).resolve().parent.parent / "12. mapear_playlists.py"
MODULE_PATH = Path(os.environ.get("MAPEAR_MODULE_PATH", str(DEFAULT_MODULE)))
SPEC = importlib.util.spec_from_file_location("mapear_playlists_under_test", MODULE_PATH)
mapear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapear)

URI_PLAYLISTS = (
    "https://youtube.googleapis.com/youtube/v3/playlists?part=snippet%2Cstatus&alt=json"
)
URI_ITEMS = "https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet&alt=json"

def make_http_error(status, reason, message, uri, api_status="FAILED_PRECONDITION"):
    body = {
        "error": {
            "code": status,
            "message": message,
            "errors": [{"message": message, "domain": "global", "reason": reason}],
            "status": api_status,
        }
    }
    resp = httplib2.Response({"status": str(status)})
    return HttpError(resp, json.dumps(body).encode("utf-8"), uri=uri)


def failed_precondition(uri):
    return make_http_error(400, "failedPrecondition", "Precondition check failed.", uri)


class _Req:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class _Channels:
    def list(self, **kwargs):
        return _Req(
            lambda: {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
        )


class _Playlists:
    def __init__(self, yt):
        self.yt = yt

    def list(self, **kwargs):
        items = [
            {"id": f"PL_{t}", "snippet": {"title": t, "publishedAt": "2020-01-01T00:00:00Z"}}
            for t in self.yt.existing
        ]
        return _Req(lambda: {"items": items})

    def insert(self, part, body):
        self.yt.playlist_inserts.append(body["snippet"]["title"])

        def run():
            if self.yt.playlist_insert_error is None:
                return {"id": "PL_new_" + body["snippet"]["title"]}
            raise self.yt.playlist_insert_error

        return _Req(run)


class _PlaylistItems:
    def __init__(self, yt):
        self.yt = yt

    def list(self, **kwargs):
        items = [
            {
                "snippet": {
                    "title": v["title"],
                    "description": v["description"],
                    "resourceId": {"videoId": v["id"]},
                },
                "status": {"privacyStatus": "public"},
            }
            for v in self.yt.videos
        ]
        return _Req(lambda: {"items": items})

    def insert(self, part, body):
        pid = body["snippet"]["playlistId"]
        vid = body["snippet"]["resourceId"]["videoId"]
        self.yt.item_inserts.append((pid, vid))

        def run():
            err = self.yt.item_insert_errors.get(pid)
            if err is not None:
                raise err
            return {"id": "item"}

        return _Req(run)


class FakeYouTube:
    """Doble del cliente de la Data API. No hace ninguna llamada de red."""

    def __init__(self, videos, existing, playlist_insert_error=None, item_insert_errors=None):
        self.videos = videos
        self.existing = existing
        self.playlist_insert_error = playlist_insert_error
        self.item_insert_errors = item_insert_errors or {}
        self.playlist_inserts = []
        self.item_inserts = []

    def channels(self):
        return _Channels()

    def playlists(self):
        return _Playlists(self)

    def playlistItems(self):
        return _PlaylistItems(self)


def two_videos():
    return [
        {
            "id": "vidAAAAAAA1",
            "title": "Band A - Album One (2020) [Death Metal]",
            "description": "Genre: Death Metal\nCountry: United States\nYear: 2020",
        },
        {
            "id": "vidBBBBBBB2",
            "title": "Band B - Album Two (2020) [Death Metal]",
            "description": "Genre: Death Metal\nCountry: United States\nYear: 2020",
        },
    ]


class MainHarness(unittest.TestCase):
    """Corre main() con rutas temporales, sin lock real, sin sleeps y sin red."""

    def run_main(self, yt):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            patches = [
                mock.patch.object(mapear, "DEFAULT_CHECKPOINT", tmp / "ckpt.txt"),
                mock.patch.object(mapear, "DEFAULT_STATE", tmp / "state.json"),
                mock.patch.object(mapear, "DEFAULT_PLAYLIST_LINKS_CACHE", tmp / "links.json"),
                mock.patch.object(mapear, "adquirir_lock", lambda *a, **k: True),
                mock.patch.object(mapear.time, "sleep", lambda *_: None),
                mock.patch("sys.argv", ["mapear", "--pausa", "0"]),
                mock.patch.dict(
                    os.environ,
                    {"YOUTUBE_FAST_EXIT": "0", "YOUTUBE_SKIP_PENDING": "0"},
                ),
            ]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                with contextlib.redirect_stdout(out):
                    try:
                        mapear.main(youtube=yt)
                        error = None
                    except HttpError as exc:
                        error = exc
            ckpt = tmp / "ckpt.txt"
            ckpt_text = ckpt.read_text(encoding="utf-8") if ckpt.exists() else ""
        return out.getvalue(), error, ckpt_text


class TestCreacionDePlaylistRechazada(MainHarness):
    def test_failed_precondition_al_crear_no_tumba_y_pausa_la_creacion(self):
        yt = FakeYouTube(
            two_videos(),
            existing=["Death Metal"],
            playlist_insert_error=failed_precondition(URI_PLAYLISTS),
        )
        out, error, ckpt = self.run_main(yt)

        # No debe escapar ninguna excepcion (RED: HttpError 400 failedPrecondition).
        self.assertIsNone(error, f"main() dejo escapar: {error}")
        # Un solo intento de crear: el resto de la ejecucion no vuelve a pedir.
        self.assertEqual(len(yt.playlist_inserts), 1, yt.playlist_inserts)
        # Lo que si existe (Death Metal) sigue recibiendo los videos.
        self.assertEqual(
            [i for i in yt.item_inserts if i[0] == "PL_Death Metal"],
            [("PL_Death Metal", "vidAAAAAAA1"), ("PL_Death Metal", "vidBBBBBBB2")],
        )
        # Una sola linea clara sobre el rechazo.
        lineas = [l for l in out.splitlines() if "failedPrecondition" in l]
        self.assertEqual(len(lineas), 1, lineas)
        self.assertIn("Band A", lineas[0])
        self.assertIn("no se crean mas playlists", lineas[0].lower())
        # Ambos videos quedan pendientes para la proxima ejecucion.
        self.assertIn("vidAAAAAAA1\tpendiente", ckpt)
        self.assertIn("vidBBBBBBB2\tpendiente", ckpt)

    def test_el_video_cuya_playlist_fue_rechazada_queda_pendiente(self):
        # Defecto: marcar como hecho un video al que le falto su playlist; la
        # siguiente ejecucion ya no lo vuelve a intentar. Aqui la rechazada es la
        # unica que le falta, asi que ningun otro camino lo deja pendiente.
        yt = FakeYouTube(
            two_videos(),
            existing=["Death Metal", "Band B", "United States", "2020"],
            playlist_insert_error=failed_precondition(URI_PLAYLISTS),
        )
        out, error, ckpt = self.run_main(yt)
        self.assertIsNone(error, f"main() dejo escapar: {error}")
        self.assertIn("vidAAAAAAA1\tpendiente", ckpt)
        self.assertIn("vidBBBBBBB2", ckpt)
        self.assertNotIn("vidBBBBBBB2\tpendiente", ckpt)

    def test_max_playlist_exceeded_tambien_pausa_la_creacion(self):
        err = make_http_error(
            400,
            "maxPlaylistExceeded",
            "The playlist cannot be created because the channel already has the maximum number of playlists allowed.",
            URI_PLAYLISTS,
            api_status="INVALID_ARGUMENT",
        )
        yt = FakeYouTube(two_videos(), existing=["Death Metal"], playlist_insert_error=err)
        out, error, _ = self.run_main(yt)
        self.assertIsNone(error, f"main() dejo escapar: {error}")
        self.assertEqual(len(yt.playlist_inserts), 1, yt.playlist_inserts)

    def test_otros_errores_al_crear_no_se_ocultan(self):
        err = make_http_error(
            403, "playlistForbidden", "Forbidden", URI_PLAYLISTS, api_status="PERMISSION_DENIED"
        )
        yt = FakeYouTube(two_videos(), existing=["Death Metal"], playlist_insert_error=err)
        _, error, _ = self.run_main(yt)
        self.assertIsNotNone(error)
        self.assertEqual(error.resp.status, 403)

    def test_400_con_otra_razon_al_crear_no_se_oculta(self):
        err = make_http_error(
            400, "invalidPlaylistSnippet", "Bad snippet", URI_PLAYLISTS, api_status="INVALID_ARGUMENT"
        )
        yt = FakeYouTube(two_videos(), existing=["Death Metal"], playlist_insert_error=err)
        _, error, _ = self.run_main(yt)
        self.assertIsNotNone(error)


    def test_failed_precondition_al_recrear_una_playlist_borrada_no_tumba(self):
        # Una playlist del cache ya no existe (404 al agregar) y YouTube no deja
        # crearla de nuevo: refresh_playlist tambien pasa por playlists.insert.
        no_existe = make_http_error(
            404, "playlistNotFound", "Playlist not found", URI_ITEMS, api_status="NOT_FOUND"
        )
        yt = FakeYouTube(
            two_videos(),
            existing=["Death Metal", "Band A", "Band B", "United States", "2020"],
            playlist_insert_error=failed_precondition(URI_PLAYLISTS),
            item_insert_errors={"PL_Death Metal": no_existe},
        )
        out, error, ckpt = self.run_main(yt)
        self.assertIsNone(error, f"main() dejo escapar: {error}")
        # El segundo video ya no vuelve a pedir que se cree la playlist.
        self.assertEqual(len(yt.playlist_inserts), 1, yt.playlist_inserts)
        self.assertIn("vidAAAAAAA1\tpendiente", ckpt)
        self.assertIn("vidBBBBBBB2\tpendiente", ckpt)

class TestAgregarVideoRechazado(MainHarness):
    def test_failed_precondition_al_agregar_se_omite_y_sigue(self):
        yt = FakeYouTube(
            two_videos(),
            existing=["Death Metal", "Band A", "Band B", "United States", "2020"],
            item_insert_errors={"PL_Death Metal": failed_precondition(URI_ITEMS)},
        )
        out, error, _ = self.run_main(yt)
        self.assertIsNone(error, f"main() dejo escapar: {error}")
        # Cada video intenta Death Metal (falla) y sigue con las demas playlists.
        for vid, banda in (("vidAAAAAAA1", "PL_Band A"), ("vidBBBBBBB2", "PL_Band B")):
            for pl in (banda, "PL_United States", "PL_2020"):
                self.assertIn((pl, vid), yt.item_inserts)
        lineas = [l for l in out.splitlines() if "No se pudo agregar" in l]
        self.assertEqual(len(lineas), 2, lineas)
        for linea in lineas:
            self.assertIn("failedPrecondition", linea)
            self.assertNotIn("https://", linea)

    def test_otros_errores_al_agregar_no_se_ocultan_como_precondition(self):
        # Un 403 de otro tipo al agregar sigue por la ruta existente (se imprime,
        # no se confunde con failedPrecondition).
        err = make_http_error(
            403, "playlistItemsNotAccessible", "No access", URI_ITEMS, api_status="PERMISSION_DENIED"
        )
        yt = FakeYouTube(
            two_videos(),
            existing=["Death Metal", "Band A", "Band B", "United States", "2020"],
            item_insert_errors={"PL_Death Metal": err},
        )
        out, error, _ = self.run_main(yt)
        self.assertIsNone(error)
        lineas = [l for l in out.splitlines() if "No se pudo agregar" in l]
        self.assertEqual(len(lineas), 2, lineas)
        for linea in lineas:
            self.assertNotIn("failedPrecondition", linea)


if __name__ == "__main__":
    unittest.main()
