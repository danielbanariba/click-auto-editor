"""Tests de clasificacion de errores de cuota de la YouTube Data API."""

import json

from googleapiclient.errors import HttpError
from httplib2 import Response

from subir_video.quota_errors import is_quota_error, is_upload_limit_error


def _http_error(status, reasons, message=""):
    """Fabrica un HttpError realista de la YouTube API."""
    errors = [{"message": message, "domain": "global", "reason": r} for r in reasons]
    content = json.dumps(
        {"error": {"code": status, "message": message, "errors": errors}}
    ).encode("utf-8")
    return HttpError(Response({"status": str(status)}), content)


# --- El bug real: 429 'rateLimitExceeded' con "Video Uploads per day" ---


def test_429_video_uploads_per_day_es_cuota():
    exc = _http_error(
        429,
        ["rateLimitExceeded"],
        "Quota exceeded for quota metric 'Video Uploads' and limit "
        "'Video Uploads per day' of service 'youtube.googleapis.com' "
        "for consumer 'project_number:866095533807'.",
    )
    assert is_quota_error(exc) is True


# --- Formatos clasicos de cuota agotada ---


def test_403_quota_exceeded_es_cuota():
    exc = _http_error(403, ["quotaExceeded"], "The request cannot be completed.")
    assert is_quota_error(exc) is True


def test_403_daily_limit_exceeded_es_cuota():
    exc = _http_error(403, ["dailyLimitExceeded"], "Daily Limit Exceeded")
    assert is_quota_error(exc) is True


# --- Rate limit transitorio (rafaga): NO es cuota diaria, no se rota ---


def test_429_rate_limit_transitorio_no_es_cuota():
    exc = _http_error(
        429,
        ["rateLimitExceeded"],
        "Rate of requests for user exceed configured project quota.",
    )
    assert is_quota_error(exc) is False


# --- Limite de subidas del CANAL (400 uploadLimitExceeded): rotar no ayuda ---


def test_400_upload_limit_no_es_cuota_es_limite_canal():
    exc = _http_error(
        400,
        ["uploadLimitExceeded"],
        "The user has exceeded the number of videos they may upload.",
    )
    assert is_quota_error(exc) is False
    assert is_upload_limit_error(exc) is True


# --- Errores ajenos a cuota ---


def test_500_no_es_cuota_ni_limite():
    exc = _http_error(500, ["backendError"], "Internal error")
    assert is_quota_error(exc) is False
    assert is_upload_limit_error(exc) is False


def main():
    """Runner standalone (sin pytest): python -m subir_video.test_quota_errors"""
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
