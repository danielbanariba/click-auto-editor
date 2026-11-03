"""
Login a DeathGrind del render y del uploader, contra un DeathGrind falso local.

La página de login trae dos tokens distintos: uno en la cookie `csrfToken` y otro
en el HTML (`window.__USER_DATA__`). El login solo acepta el del HTML; con el de
la cookie responde 419 y la descripción del video sale sin géneros, país ni sello.

Ejecutar: pytest tests/test_login_deathgrind.py
"""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = {
    "render": PROJECT_ROOT / "2. ffmpeg_render.py",
    "uploader": PROJECT_ROOT / "3. subir_video_API.py",
}
TOKEN_COOKIE = "token-de-la-cookie"
TOKEN_PAGINA = "token-de-la-pagina"


def cargar_script(nombre):
    alias = f"login_deathgrind_{nombre}"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS[nombre])
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[alias] = modulo
    spec.loader.exec_module(modulo)
    return modulo


class DeathGrindFalso(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/auth/sign-in":
            self.send_error(404)
            return
        datos = {"userId": 0, "lang": "en-US"}
        if self.server.token_en_pagina:
            datos["csrfToken"] = TOKEN_PAGINA
        cuerpo = f"<script>window.__USER_DATA__={json.dumps(datos)};</script>".encode()
        self.send_response(200)
        self.send_header("Set-Cookie", f"csrfToken={TOKEN_COOKIE}; Path=/; HttpOnly")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        recibido = self.headers.get("x-csrf-token")
        self.server.tokens_recibidos.append(recibido)
        aceptado = self.path == "/api/auth/login" and recibido == self.server.token_valido
        cuerpo = b'{"status":"Accepted"}' if aceptado else b'{"error":""}'
        self.send_response(202 if aceptado else 419)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


@pytest.fixture
def deathgrind():
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), DeathGrindFalso)
    servidor.token_en_pagina = True
    servidor.token_valido = TOKEN_PAGINA
    servidor.tokens_recibidos = []
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    yield servidor
    servidor.shutdown()
    servidor.server_close()


@pytest.fixture(params=sorted(SCRIPTS))
def login(request, deathgrind, monkeypatch):
    modulo = cargar_script(request.param)
    base = f"http://127.0.0.1:{deathgrind.server_address[1]}"
    monkeypatch.setattr(modulo, "BASE_URL", base)
    monkeypatch.setattr(modulo, "API_URL", f"{base}/api")
    monkeypatch.setattr(modulo, "DEATHGRIND_SESSION", None, raising=False)
    monkeypatch.setenv("DEATHGRIND_EMAIL", "usuario@example.com")
    monkeypatch.setenv("DEATHGRIND_PASSWORD", "clave")
    return modulo.crear_sesion_autenticada


def test_el_login_manda_el_token_de_la_pagina(login, deathgrind):
    sesion = login()

    assert sesion is not None
    assert deathgrind.tokens_recibidos == [TOKEN_PAGINA]


def test_sin_token_en_la_pagina_usa_el_de_la_cookie(login, deathgrind):
    deathgrind.token_en_pagina = False
    deathgrind.token_valido = TOKEN_COOKIE

    sesion = login()

    assert sesion is not None
    assert deathgrind.tokens_recibidos == [TOKEN_COOKIE]
