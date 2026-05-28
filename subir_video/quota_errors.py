"""Clasificacion de errores de la YouTube Data API.

Modulo liviano y puro (solo depende de json y de HttpError) para poder
testearlo sin levantar todo el pipeline de subida. Decide si un HttpError
corresponde a:

  - cuota / limite diario agotado  -> hay que ROTAR a otra credencial
  - limite de subidas del CANAL    -> rotar NO ayuda, hay que esperar ~24h
"""

import json

from googleapiclient.errors import HttpError


def _get_error_reasons(exc: HttpError) -> list[str]:
    """Extrae las razones (campo 'reason') de un HttpError."""
    try:
        content = (
            exc.content.decode("utf-8")
            if isinstance(exc.content, bytes)
            else exc.content
        )
        data = json.loads(content)
        errors = data.get("error", {}).get("errors", [])
        return [e.get("reason", "") for e in errors]
    except (json.JSONDecodeError, AttributeError):
        return []


def _get_error_message(exc: HttpError) -> str:
    """Extrae el mensaje principal (campo 'message') de un HttpError."""
    try:
        content = (
            exc.content.decode("utf-8")
            if isinstance(exc.content, bytes)
            else exc.content
        )
        data = json.loads(content)
        return str(data.get("error", {}).get("message", ""))
    except (json.JSONDecodeError, AttributeError):
        return ""


# Reasons que SIEMPRE indican cuota / limite diario agotado.
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}


def is_quota_error(exc: HttpError) -> bool:
    """True si el error es por cuota / limite diario agotado (hay que rotar credencial).

    Cubre dos formatos de la API:
      - Clasico (403): reason 'quotaExceeded' / 'dailyLimitExceeded'.
      - Cloud (429):   reason 'rateLimitExceeded' con mensaje
                       "Quota exceeded for quota metric ... per day"
                       (ej: limite 'Video Uploads per day').

    Un rate-limit transitorio (rafaga de requests) trae reason
    'rateLimitExceeded' pero SIN 'per day' en el mensaje; en ese caso esto
    devuelve False para reintentar en la misma credencial en vez de quemarla.
    """
    if exc.resp is None:
        return False
    if exc.resp.status not in (403, 429):
        return False
    reasons = _get_error_reasons(exc)
    if any(r in _QUOTA_REASONS for r in reasons):
        return True
    message = _get_error_message(exc).lower()
    return "quota exceeded for quota metric" in message or "per day" in message


def is_upload_limit_error(exc: HttpError) -> bool:
    """True si el error es el limite de subidas del CANAL (rotar NO ayuda)."""
    if exc.resp is None:
        return False
    if exc.resp.status != 400:
        return False
    reasons = _get_error_reasons(exc)
    return "uploadLimitExceeded" in reasons
