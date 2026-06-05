"""Test del orden de rotacion de credenciales (respeto del prefijo)."""

from pathlib import Path

from google.auth.exceptions import RefreshError

import subir_video.authenticate as auth

_UP1 = (
    Path("credentials/client_secrets_upload_1.json"),
    Path("credentials/token_upload_1.json"),
)
_UP2 = (
    Path("credentials/client_secrets_upload_2.json"),
    Path("credentials/token_upload_2.json"),
)
_PL = [
    (
        Path(f"credentials/client_secrets_playlists_{i}.json"),
        Path(f"credentials/token_playlists_{i}.json"),
    )
    for i in range(1, 9)
]
_UP3 = (
    Path("credentials/client_secrets_upload_3.json"),
    Path("credentials/token_upload_3.json"),
)


class _FakeYouTube:
    """Cliente youtube minimo: soporta channels().list(...).execute()."""

    def channels(self):
        return self

    def list(self, **kwargs):
        return self

    def execute(self):
        return {"items": [{"id": "UCxxxxxxxx"}]}


def _instalar_fakes(upload_pool=None, dead=None, cliente_fake=False):
    """Instala fakes de auth.

    'dead' = nombres de client_secrets cuyo token esta revocado:
    authenticate_with_credentials(..., interactive=False) lanza RefreshError,
    igual que un invalid_grant real. Con interactive=True el fake igual
    devuelve cliente (simula que el browser reautentica)."""
    pool = list(upload_pool) if upload_pool is not None else [_UP1, _UP2]
    dead_names = set(dead or [])

    def _fake_get(prefix=None):
        if prefix == "upload":
            return list(pool)
        # Orden real al globear: playlists_* van antes que upload_* alfabeticamente.
        return _PL + list(pool)

    def _fake_auth(secrets, token, interactive=True):
        name = Path(secrets).name
        if name in dead_names and not interactive:
            raise RefreshError(f"invalid_grant: token revocado {name}")
        return _FakeYouTube() if cliente_fake else f"YT::{name}"

    auth.get_credential_sets = _fake_get
    auth.authenticate_with_credentials = _fake_auth
    auth.reset_exhausted_credentials()


def test_rotacion_upload_marca_upload_no_playlists():
    """Al rotar con prefix='upload', debe marcar upload_1 y pasar a upload_2,
    nunca tocar las credenciales de playlists."""
    _instalar_fakes()

    cliente = auth.authenticate_next(prefix="upload")

    assert cliente == "YT::client_secrets_upload_2.json", cliente
    assert str(_UP1[0]) in auth._exhausted_credentials
    assert str(_PL[0][0]) not in auth._exhausted_credentials


def test_rotacion_aisla_pools_no_derrama():
    """Tras agotar upload_1 y upload_2 NO debe derramar a playlists: devuelve None.
    Apunta CREDENTIALS_DIR al dir real (con playlists) para detectar si derrama."""
    _instalar_fakes()
    auth.CREDENTIALS_DIR = Path("credentials")

    auth.authenticate_next(prefix="upload")  # upload_1 -> upload_2
    siguiente = auth.authenticate_next(prefix="upload")  # upload_2 agotado

    assert siguiente is None, f"derramo a otro pool: {siguiente}"
    assert str(_UP1[0]) in auth._exhausted_credentials
    assert str(_UP2[0]) in auth._exhausted_credentials


def test_authenticate_next_saltea_credencial_muerta():
    """authenticate_next debe saltar un token revocado (RefreshError) sin abrir
    browser y pasar a la siguiente credencial viva, marcando la muerta."""
    _instalar_fakes(upload_pool=[_UP1, _UP2, _UP3], dead={"client_secrets_upload_2.json"})

    cliente = auth.authenticate_next(prefix="upload")  # agota up1, salta up2, usa up3

    assert cliente == "YT::client_secrets_upload_3.json", cliente
    assert str(_UP1[0]) in auth._exhausted_credentials  # current agotada
    assert str(_UP2[0]) in auth._exhausted_credentials  # muerta marcada (skip)


def test_authenticate_next_todas_muertas_devuelve_none():
    """Si tras agotar la actual todas las restantes estan muertas, devuelve None
    en vez de colgarse en el browser."""
    _instalar_fakes(
        upload_pool=[_UP1, _UP2],
        dead={"client_secrets_upload_2.json"},
    )

    siguiente = auth.authenticate_next(prefix="upload")  # agota up1, up2 muerta

    assert siguiente is None, f"deberia ser None, fue: {siguiente}"


def test_authenticate_prefiere_credencial_viva_sin_browser():
    """authenticate debe saltar el token muerto y devolver la primera viva sin
    caer al flujo interactivo (browser)."""
    _instalar_fakes(upload_pool=[_UP1, _UP2], dead={"client_secrets_upload_1.json"})

    cliente = auth.authenticate(prefix="upload")

    assert cliente == "YT::client_secrets_upload_2.json", cliente


def test_probar_credenciales_reporta_muertas_sin_colgar():
    """probar_credenciales_disponibles debe reportar las revocadas en la lista
    'muertas' (4to elemento) sin abrir browser ni crashear."""
    _instalar_fakes(
        upload_pool=[_UP1, _UP2],
        dead={"client_secrets_upload_1.json"},
        cliente_fake=True,
    )

    youtube, sanas, agotadas, muertas = auth.probar_credenciales_disponibles(
        prefix="upload"
    )

    assert youtube is not None
    assert _UP2[0] in sanas
    assert _UP1[0] in muertas
    assert _UP1[0] not in sanas


def main():
    import sys

    pruebas = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"  OK  {prueba.__name__}")
        except AssertionError as exc:
            fallos += 1
            print(f"FALLA {prueba.__name__}: {exc}")
    print("=" * 50)
    if fallos:
        print(f"{fallos}/{len(pruebas)} pruebas FALLARON")
        sys.exit(1)
    print(f"{len(pruebas)} pruebas OK")


if __name__ == "__main__":
    main()
