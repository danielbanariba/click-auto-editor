"""Test del orden de rotacion de credenciales (respeto del prefijo)."""

from pathlib import Path

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
# Orden real al globear: playlists_* van antes que upload_* alfabeticamente.
_ALL = _PL + [_UP1, _UP2]


def _fake_get_credential_sets(prefix=None):
    if prefix == "upload":
        return [_UP1, _UP2]
    return _ALL


def _instalar_fakes():
    auth.get_credential_sets = _fake_get_credential_sets
    auth.authenticate_with_credentials = lambda s, t: f"YT::{Path(s).name}"
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
