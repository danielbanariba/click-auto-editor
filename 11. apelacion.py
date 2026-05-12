import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from browser_profiles import (
    describe_profile_source,
    discover_signed_in_profile,
    load_google_cookies_for_playwright,
    prepare_runtime_profile,
)
from config import PLAYWRIGHT_PROFILE_DIR, PLAYWRIGHT_SELECTORS_DIR
from studio_claims import (
    claim_dialog_is_open,
    claim_item_key,
    dismiss_claim_dialog,
    ensure_content_list,
    load_checkpoint,
    open_claim_modal_with_recovery,
    reset_content_scroll,
    save_queue,
    scan_claim_rows,
    update_checkpoint,
)

mensaje = (
    "Hola nuevamente! Lastima que rechazaron mi solicitud, la razon por la cual creo que deberian "
    "de retirar el reclamo es que afecta directamente mi canal, y el alcance que puede llegar los "
    "videos hacia mas gente, y de esa manera poder llegar a mas personas, y que puedan disfrutar "
    "de la musica que ustedes crean, espero que puedan reconsiderar su decision, saludos desde Honduras!"
)

firma = "Daniel Alejandro Barrientos Anariba"

nombre_y_apellido = "Daniel Alejandro Barrientos Anariba"
dirreccion_postal = "Honduras, Francisco Morazan, Comayaguela, Districto Central, Residencial la Ca#ada, Bloque BH, Casa 6312"
correo_electronico = "banaribad@gmail.com"
ciudad = "Tegucigalpa"
pais = "Honduras"
departamento = "Francisco Morazan"
codigo_postal = "12101"

DEFAULT_URL = os.environ.get("YOUTUBE_STUDIO_URL", "https://studio.youtube.com/")
DEFAULT_CHANNEL = os.environ.get("PLAYWRIGHT_CHANNEL", "chrome")
DEFAULT_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")
SELECTORS_PATH = PLAYWRIGHT_SELECTORS_DIR / "apelacion.json"
ACTIONS_PATH = PLAYWRIGHT_SELECTORS_DIR / "apelacion_acciones.json"
DEFAULT_QUEUE_PATH = Path(__file__).resolve().parent / "data" / "apelacion_claims_queue.json"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "apelacion_claims_checkpoint.txt"
DEFAULT_RUNTIME_PROFILE_DIR = Path(__file__).resolve().parent / ".runtime_browser_profiles" / "apelacion"

P_CONTINUAR = re.compile(r"(continuar|siguiente|next|continue)", re.IGNORECASE)
P_ENVIAR = re.compile(r"(^enviar$|^submit$|^send$|^envio$)", re.IGNORECASE)
P_APELAR = re.compile(r"(apelar|appeal|apelaci[oó]n)", re.IGNORECASE)
P_ENTIENDO_RIESGOS = re.compile(r"(entiendo.*riesgos|i understand.*risk)", re.IGNORECASE)
P_SELECCIONAR = re.compile(
    r"(seleccionar.*canci[oó]n|select.*song|ver detalles|detalles de la reclamaci[oó]n|reclamaci[oó]n de derechos de autor|copyright claim)",
    re.IGNORECASE,
)
P_NOMBRE = re.compile(r"(nombre.*apellido|nombre completo|full name)", re.IGNORECASE)
P_DIRECCION = re.compile(r"(direcci[oó]n postal|direcci[oó]n|postal address|address)", re.IGNORECASE)
P_CORREO = re.compile(
    r"(correo electr[oó]nic[oa]|direcci[oó]n de correo|email|e-mail|@|email address)",
    re.IGNORECASE,
)
P_CIUDAD = re.compile(r"(ciudad|city|localidad)", re.IGNORECASE)
P_PAIS = re.compile(
    r"(pa[ií]s/?regi[oó]n|pa[ií]s\s*o\s*regi[oó]n|country/?region|pa[ií]s|country|territorio|naci[oó]n)",
    re.IGNORECASE,
)
P_DEPARTAMENTO = re.compile(r"(departamento|estado|provincia|state|region)", re.IGNORECASE)
P_CODIGO_POSTAL = re.compile(r"(c[oó]digo postal|postal code|zip)", re.IGNORECASE)
P_MENSAJE = re.compile(r"(mensaje|motivo|explica|describe|statement|reason)", re.IGNORECASE)
P_FIRMA = re.compile(r"(firma|signature|nombre completo|nombre y apellido|full name)", re.IGNORECASE)
P_CERRAR = re.compile(r"(cerrar|finalizar|listo|done|close)", re.IGNORECASE)

SELECTOR_STEPS = [
    ("seleccionar_cancion", "Selecciona el video o la reclamacion para ver detalles."),
    ("apelar", "Haz clic en el boton Apelar."),
    ("apelar_confirmar", "Confirma la apelacion si aparece un paso extra."),
    ("continuar", "Haz clic en Continuar / Siguiente."),
    ("entiendo_riesgos", "Marca Entiendo los riesgos (checkbox)."),
    ("nombre", "Haz clic en el campo Nombre y apellido."),
    ("direccion", "Haz clic en el campo Direccion postal."),
    ("correo", "Haz clic en el campo Correo electronico."),
    ("ciudad", "Haz clic en el campo Ciudad."),
    ("pais", "Haz clic en el campo Pais."),
    ("departamento", "Haz clic en el campo Departamento/Estado."),
    ("codigo_postal", "Haz clic en el campo Codigo postal."),
    ("mensaje", "Haz clic en el campo de mensaje de apelacion."),
    ("firma", "Haz clic en el campo de firma."),
    ("cerrar", "Haz clic en Cerrar / Finalizar."),
]


_MESES_ES = [
    "ene(?:ro)?", "feb(?:rero)?", "mar(?:zo)?", "abr(?:il)?",
    "may(?:o)?", "jun(?:io)?", "jul(?:io)?", "ago(?:sto)?",
    "sep(?:t(?:iembre)?)?|set(?:iembre)?", "oct(?:ubre)?",
    "nov(?:iembre)?", "dic(?:iembre)?",
]
_MESES_EN = [
    "jan(?:uary)?", "feb(?:ruary)?", "mar(?:ch)?", "apr(?:il)?",
    "may", "jun(?:e)?", "jul(?:y)?", "aug(?:ust)?",
    "sep(?:t(?:ember)?)?", "oct(?:ober)?",
    "nov(?:ember)?", "dec(?:ember)?",
]


def build_fecha_pattern(fecha_iso):
    """Construye un regex que matchea variantes de una fecha ISO (YYYY-MM-DD).
    Cubre formatos es/en/numericos que YouTube Studio puede mostrar.
    Devuelve None si la fecha es vacia o invalida."""
    if not fecha_iso:
        return None
    fecha_iso = fecha_iso.strip()
    if not fecha_iso:
        return None
    try:
        from datetime import datetime
        dt = datetime.strptime(fecha_iso, "%Y-%m-%d")
    except ValueError:
        print(f"AVISO: fecha invalida ({fecha_iso!r}), se ignora. Usa formato YYYY-MM-DD.")
        return None
    d, m, y = dt.day, dt.month, dt.year
    mes_es = _MESES_ES[m - 1]
    mes_en = _MESES_EN[m - 1]
    variants = [
        rf"\b{d}\s+(?:{mes_es})\.?\s+{y}\b",
        rf"\b(?:{mes_en})\.?\s+{d},?\s+{y}\b",
        rf"\b{d:02d}/{m:02d}/{y}\b",
        rf"\b{d}/{m}/{y}\b",
        rf"\b{y}-{m:02d}-{d:02d}\b",
    ]
    return re.compile("|".join(variants), re.IGNORECASE)


def build_hora_pattern(hora_hhmm):
    """Construye un regex que matchea variantes de una hora HH:MM (24h).
    Cubre formato 24h ('23:45') y 12h con AM/PM ('11:45 PM', '11:45 p.m.', '11:45p.m.').
    Devuelve None si la hora es vacia o invalida."""
    if not hora_hhmm:
        return None
    hora_hhmm = hora_hhmm.strip()
    if not hora_hhmm:
        return None
    try:
        hh_str, mm_str = hora_hhmm.split(":")
        hh = int(hh_str)
        mm = int(mm_str)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
    except (ValueError, AttributeError):
        print(f"AVISO: hora invalida ({hora_hhmm!r}), se ignora. Usa formato HH:MM (24h).")
        return None

    if hh == 0:
        hh12, ampm = 12, "am"
    elif hh < 12:
        hh12, ampm = hh, "am"
    elif hh == 12:
        hh12, ampm = 12, "pm"
    else:
        hh12, ampm = hh - 12, "pm"

    variants = [
        rf"\b{hh:02d}:{mm:02d}\b",
        rf"\b{hh}:{mm:02d}\b",
        rf"\b{hh12}:{mm:02d}\s*{ampm}\b",
        rf"\b{hh12}:{mm:02d}\s*{ampm[0]}\.{ampm[1]}\.?",
    ]
    return re.compile("|".join(variants), re.IGNORECASE)


def video_fecha_matches(item, fecha_pat, hora_pat=None):
    """True si el item (fila escaneada) contiene la fecha (y opcionalmente la hora)
    en alguno de sus campos textuales."""
    if fecha_pat is None or not isinstance(item, dict):
        return False
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in ("date_text", "row_text", "title")
    )
    if not fecha_pat.search(blob):
        return False
    if hora_pat is not None and not hora_pat.search(blob):
        return False
    return True


def load_selectors(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if isinstance(data, dict) and "steps" in data:
        return data.get("steps", {})
    return data if isinstance(data, dict) else {}


def save_selectors(path, selectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "steps": selectors}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def save_actions(path, actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "actions": actions}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def summarize_selector(info):
    if not info:
        return "selector vacio"
    keys = []
    for label, key in (
        ("data-testid", "data_testid"),
        ("aria", "aria_label"),
        ("title", "title"),
        ("name", "name"),
        ("text", "text"),
        ("role", "role"),
        ("id", "id"),
    ):
        value = info.get(key)
        if value:
            keys.append(f"{label}={str(value)[:40]}")
    return ", ".join(keys) if keys else "selector capturado"


def ensure_selector_probe(page):
    page.evaluate(
        """
        () => {
          if (window.__codexSelectorProbeInstalled) {
            return;
          }
          window.__codexLastSelector = null;
          function escapeAttr(value) {
            return String(value).replace(/\\\\/g, '\\\\\\\\').replace(/"/g, '\\\\\"');
          }
          function cssPath(el) {
            if (!el || !el.tagName) return null;
            const parts = [];
            let node = el;
            while (node && node.nodeType === 1 && node !== document.body) {
              let part = node.tagName.toLowerCase();
              const id = node.getAttribute('id');
              if (id) {
                part += `#${escapeAttr(id)}`;
                parts.unshift(part);
                break;
              }
              const parent = node.parentElement;
              if (!parent) {
                parts.unshift(part);
                break;
              }
              const siblings = Array.from(parent.children).filter((s) => s.tagName === node.tagName);
              if (siblings.length > 1) {
                const index = siblings.indexOf(node) + 1;
                part += `:nth-of-type(${index})`;
              }
              parts.unshift(part);
              node = parent;
            }
            return parts.join(' > ');
          }
          function buildInfo(el) {
            if (!el) return null;
            const info = {};
            const attrs = {
              'data_testid': el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa') || el.getAttribute('data-e2e') || el.getAttribute('data-automation'),
              'aria_label': el.getAttribute('aria-label'),
              'title': el.getAttribute('title'),
              'name': el.getAttribute('name'),
              'placeholder': el.getAttribute('placeholder'),
              'id': el.getAttribute('id'),
              'type': el.getAttribute('type'),
            };
            Object.keys(attrs).forEach((key) => {
              if (attrs[key]) info[key] = attrs[key];
            });
            const text = (el.innerText || '').trim();
            if (text) {
              info.text = text.slice(0, 200);
            }
            const tag = el.tagName.toLowerCase();
            info.tag = tag;
            const roleAttr = el.getAttribute('role');
            if (roleAttr) {
              info.role = roleAttr;
            } else if (tag === 'button') {
              info.role = 'button';
            } else if (tag === 'a') {
              info.role = 'link';
            } else if (tag === 'textarea') {
              info.role = 'textbox';
            } else if (tag === 'input') {
              const type = (el.getAttribute('type') || '').toLowerCase();
              if (type === 'checkbox') {
                info.role = 'checkbox';
              } else {
                info.role = 'textbox';
              }
            }
            info.css = cssPath(el);
            return info;
          }
          document.addEventListener(
            'click',
            (event) => {
              const path = event.composedPath ? event.composedPath() : [event.target];
              let candidate = null;
              for (const node of path) {
                if (node && node.nodeType === 1) {
                  if (node.matches('button, a, input, textarea, [role]')) {
                    candidate = node;
                    break;
                  }
                }
              }
              if (!candidate) return;
              window.__codexLastSelector = buildInfo(candidate);
            },
            true
          );
          window.__codexSelectorProbeInstalled = true;
        }
        """
    )


def ensure_action_recorder(page):
    page.evaluate(
        """
        () => {
          if (window.__codexActionRecorderInstalled) {
            return;
          }
          window.__codexActionLog = [];
          function escapeAttr(value) {
            return String(value).replace(/\\\\/g, '\\\\\\\\').replace(/"/g, '\\\\\"');
          }
          function cssPath(el) {
            if (!el || !el.tagName) return null;
            const parts = [];
            let node = el;
            while (node && node.nodeType === 1 && node !== document.body) {
              let part = node.tagName.toLowerCase();
              const id = node.getAttribute('id');
              if (id) {
                part += `#${escapeAttr(id)}`;
                parts.unshift(part);
                break;
              }
              const parent = node.parentElement;
              if (!parent) {
                parts.unshift(part);
                break;
              }
              const siblings = Array.from(parent.children).filter((s) => s.tagName === node.tagName);
              if (siblings.length > 1) {
                const index = siblings.indexOf(node) + 1;
                part += `:nth-of-type(${index})`;
              }
              parts.unshift(part);
              node = parent;
            }
            return parts.join(' > ');
          }
          function buildInfo(el) {
            if (!el) return null;
            const info = {};
            const attrs = {
              'data_testid': el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa') || el.getAttribute('data-e2e') || el.getAttribute('data-automation'),
              'aria_label': el.getAttribute('aria-label'),
              'title': el.getAttribute('title'),
              'name': el.getAttribute('name'),
              'placeholder': el.getAttribute('placeholder'),
              'id': el.getAttribute('id'),
              'type': el.getAttribute('type'),
            };
            Object.keys(attrs).forEach((key) => {
              if (attrs[key]) info[key] = attrs[key];
            });
            const text = (el.innerText || '').trim();
            if (text) {
              info.text = text.slice(0, 200);
            }
            const tag = el.tagName.toLowerCase();
            info.tag = tag;
            const roleAttr = el.getAttribute('role');
            if (roleAttr) {
              info.role = roleAttr;
            } else if (tag === 'button') {
              info.role = 'button';
            } else if (tag === 'a') {
              info.role = 'link';
            } else if (tag === 'textarea') {
              info.role = 'textbox';
            } else if (tag === 'input') {
              const type = (el.getAttribute('type') || '').toLowerCase();
              if (type === 'checkbox') {
                info.role = 'checkbox';
              } else {
                info.role = 'textbox';
              }
            }
            info.css = cssPath(el);
            return info;
          }
          function record(type, el, value) {
            const info = buildInfo(el);
            if (!info) return;
            window.__codexActionLog.push({
              type,
              value: value ?? null,
              info,
              time: Date.now(),
            });
          }
          function pickCandidate(path) {
            const selector = 'button, a, input, textarea, select, [role], [aria-label], [data-testid]';
            for (const node of path) {
              if (node && node.nodeType === 1) {
                if (node.matches(selector)) {
                  return node;
                }
              }
            }
            return null;
          }
          document.addEventListener(
            'click',
            (event) => {
              const path = event.composedPath ? event.composedPath() : [event.target];
              const candidate = pickCandidate(path) || event.target;
              if (!candidate) return;
              record('click', candidate, null);
            },
            true
          );
          document.addEventListener(
            'change',
            (event) => {
              const el = event.target;
              if (!el || !el.tagName) return;
              const tag = el.tagName.toLowerCase();
              if (tag === 'input' || tag === 'textarea' || tag === 'select') {
                record('fill', el, el.value);
              }
            },
            true
          );
          document.addEventListener(
            'blur',
            (event) => {
              const el = event.target;
              if (!el || !el.isContentEditable) return;
              const value = (el.innerText || el.textContent || '').trim();
              if (value) {
                record('fill', el, value);
              }
            },
            true
          );
          window.__codexActionRecorderInstalled = true;
        }
        """
    )


def capture_selector(page, descripcion):
    ensure_selector_probe(page)
    page.evaluate("() => { window.__codexLastSelector = null; }")
    print(f"\n{descripcion}")
    print("Haz clic en el elemento y luego vuelve aqui.")
    respuesta = input("Presiona Enter para guardar, o escribe 's' para omitir: ").strip().lower()
    if respuesta in ("fin", "salir", "exit", "q"):
        return "__FIN__"
    if respuesta in ("s", "skip", "omitir"):
        return None
    info = page.evaluate("() => window.__codexLastSelector")
    if not info:
        print("No se detecto ningun clic. Omite este paso o intenta de nuevo.")
    return info


def learn_selectors(page, selectors_path, steps):
    selectors = load_selectors(selectors_path)
    print("\nModo aprendizaje: sigue los pasos y haz clic en cada elemento.")
    print("Si quieres saltar un paso, escribe 's'. Si quieres terminar, escribe 'fin'.")
    for key, descripcion in steps:
        info = capture_selector(page, descripcion)
        if info == "__FIN__":
            break
        if info:
            selectors[key] = info
            print(f"Selector guardado para {key}: {summarize_selector(info)}")
    save_selectors(selectors_path, selectors)
    print(f"\nSelectores guardados en: {selectors_path}")


def record_actions(page, actions_path):
    ensure_action_recorder(page)
    input("Presiona Enter para iniciar la grabacion de acciones...")
    page.evaluate("() => { window.__codexActionLog = []; }")
    input("Realiza todo el flujo manualmente. Presiona Enter aqui para finalizar la grabacion...")
    time.sleep(0.5)
    actions = page.evaluate("() => window.__codexActionLog")
    save_actions(actions_path, actions or [])
    print(f"\nAcciones guardadas en: {actions_path}")


def css_attr_selector(attr, value):
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'[{attr}="{escaped}"]'


def locator_from_selector_info(root, info):
    if not info:
        return None
    data_testid = info.get("data_testid")
    if data_testid:
        return root.locator(css_attr_selector("data-testid", data_testid))
    element_id = info.get("id")
    if element_id:
        return root.locator(css_attr_selector("id", element_id))
    aria_label = info.get("aria_label")
    role = info.get("role")
    text = info.get("text")
    if role and (aria_label or text):
        return root.get_by_role(role, name=aria_label or text)
    if aria_label:
        return root.get_by_label(aria_label)
    placeholder = info.get("placeholder")
    if placeholder:
        return root.get_by_placeholder(placeholder)
    name_attr = info.get("name")
    if name_attr:
        return root.locator(css_attr_selector("name", name_attr))
    if text:
        return root.get_by_text(text)
    css = info.get("css")
    if css:
        return root.locator(css)
    return None


def click_by_selector_info(root, info, timeout_ms=3000, optional=False):
    locator = locator_from_selector_info(root, info)
    if locator is None:
        return False
    try:
        target = locator.first
        target.wait_for(state="visible", timeout=timeout_ms)
        target.scroll_into_view_if_needed()
        target.click()
        return True
    except Exception:
        if not optional:
            print("No se pudo usar el selector guardado.")
        return False


def fill_by_selector_info(root, info, value, timeout_ms=3000, press_enter=False):
    locator = locator_from_selector_info(root, info)
    if locator is None:
        return False
    try:
        target = locator.first
        target.wait_for(state="visible", timeout=timeout_ms)
        target.scroll_into_view_if_needed()
        target.fill(value)
        if press_enter:
            target.press("Enter")
        return True
    except Exception:
        return False


def click_with_selector(root, info, patterns, roles=("button", "menuitem", "link", "radio", "option"), timeout_ms=3000, optional=False, descripcion=None):
    if info and click_by_selector_info(root, info, timeout_ms=timeout_ms, optional=True):
        return True
    return click_action(root, patterns, roles=roles, timeout_ms=timeout_ms, optional=optional, descripcion=descripcion)


def fill_with_selector(root, info, patterns, value, timeout_ms=3000, press_enter=False):
    if info and fill_by_selector_info(root, info, value, timeout_ms=timeout_ms, press_enter=press_enter):
        return True
    return fill_by_label(root, patterns, value, timeout_ms=timeout_ms, press_enter=press_enter)


def get_active_root(page):
    try:
        dialogs = page.get_by_role("dialog")
        if dialogs.count() > 0:
            dialog = dialogs.last
            if dialog.is_visible():
                return dialog
    except Exception:
        pass
    return page


def click_action(root, patterns, roles=("button", "menuitem", "link", "radio", "option"), timeout_ms=3000, optional=False, descripcion=None):
    for pattern in patterns:
        for role in roles:
            try:
                locator = root.get_by_role(role, name=pattern)
                target = locator.first
                target.wait_for(state="visible", timeout=timeout_ms)
                target.scroll_into_view_if_needed()
                target.click()
                return True
            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue
        try:
            locator = root.get_by_text(pattern)
            target = locator.first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.click()
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue

    if not optional:
        texto = descripcion or " / ".join([pat.pattern for pat in patterns])
        print(f"No se encontro el elemento: {texto}")
    return False


def click_repeatedly(root, patterns, max_clicks=2, timeout_ms=2000, selector_info=None):
    clicks = 0
    for _ in range(max_clicks):
        if selector_info and click_by_selector_info(root, selector_info, timeout_ms=timeout_ms, optional=True):
            clicks += 1
            time.sleep(0.8)
            continue
        if not click_action(root, patterns, timeout_ms=timeout_ms, optional=True):
            break
        clicks += 1
        time.sleep(0.8)
    return clicks


def fill_by_label(root, patterns, value, timeout_ms=3000, press_enter=False):
    for pattern in patterns:
        try:
            locator = root.get_by_label(pattern)
            target = locator.first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.fill(value)
            if press_enter:
                target.press("Enter")
            return True
        except PlaywrightTimeoutError:
            pass
        except Exception:
            pass

        try:
            locator = root.get_by_placeholder(pattern)
            target = locator.first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.fill(value)
            if press_enter:
                target.press("Enter")
            return True
        except PlaywrightTimeoutError:
            pass
        except Exception:
            pass

        for role in ("textbox", "combobox"):
            try:
                locator = root.get_by_role(role, name=pattern)
                target = locator.first
                target.wait_for(state="visible", timeout=timeout_ms)
                target.scroll_into_view_if_needed()
                target.fill(value)
                if press_enter:
                    target.press("Enter")
                return True
            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue

    return False


def fill_textarea_fallback(root, value):
    locator = root.locator("textarea:visible")
    try:
        if locator.count() > 0:
            target = locator.first
            target.scroll_into_view_if_needed()
            target.fill(value)
            return True
    except Exception:
        pass
    return False


def fill_last_textbox_fallback(root, value):
    locator = root.locator("input:visible")
    try:
        if locator.count() > 0:
            target = locator.last
            target.scroll_into_view_if_needed()
            target.fill(value)
            return True
    except Exception:
        pass
    return False


def check_all_visible_checkboxes(root):
    checked = 0
    try:
        checkboxes = root.get_by_role("checkbox")
        for i in range(checkboxes.count()):
            checkbox = checkboxes.nth(i)
            try:
                if not checkbox.is_visible():
                    continue
                checkbox.scroll_into_view_if_needed()
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click()
                checked += 1
            except Exception:
                continue
    except Exception:
        pass
    return checked


def fill_required_field(root, selector_info, patterns, value, nombre, press_enter=False, page=None):
    if fill_with_selector(root, selector_info, patterns, value, press_enter=press_enter):
        return True
    if page is not None:
        try:
            preview = ""
            try:
                preview = (get_active_root(page).inner_text() or "").strip().replace("\n", " | ")[:2000]
            except Exception:
                pass
            patterns_str = " | ".join(p.pattern for p in patterns)
            print(f"     [DEBUG] campo '{nombre}' no encontrado. patterns: {patterns_str}")
            print(f"     [DEBUG] preview modal: {preview!r}")
        except Exception:
            pass
    print(f"No se encontro el campo: {nombre}")
    return False


def _click_nth_take_action_in_page(page, claim_index):
    """Click el i-th boton 'Tomar medidas' visible Y HABILITADO en la pagina /copyright.
    Devuelve {'ok': bool, 'total': int}."""
    return page.evaluate(
        """
        (index) => {
          const all = Array.from(document.querySelectorAll(
            'button, ytcp-button-shape, [role="button"], tp-yt-paper-button'
          ));
          const ctas = all.filter(el => {
            const t = (el.innerText || el.textContent || '').trim().toLowerCase();
            return t === 'tomar medidas' || t === 'take action';
          });
          const isEnabled = (el) => {
            if (el.disabled === true) return false;
            const aria = (el.getAttribute && el.getAttribute('aria-disabled') || '').toLowerCase();
            if (aria === 'true') return false;
            let cur = el;
            for (let i = 0; i < 5 && cur; i++) {
              const cls = String(cur.className || '');
              if (/disabled/.test(cls)) return false;
              if (cur.getAttribute) {
                if (cur.getAttribute('disabled') !== null) return false;
                if ((cur.getAttribute('aria-disabled') || '').toLowerCase() === 'true') return false;
              }
              cur = cur.parentElement;
            }
            return true;
          };
          const usable = ctas.filter(el => {
            const r = el.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) return false;
            const s = window.getComputedStyle(el);
            if (s.display === 'none' || s.visibility === 'hidden') return false;
            if (parseFloat(s.opacity || '1') < 0.4) return false;
            if (s.pointerEvents === 'none') return false;
            return isEnabled(el);
          });
          if (index >= usable.length) return {ok: false, total: usable.length};
          const el = usable[index];
          el.scrollIntoView({block: 'center'});
          el.click();
          return {ok: true, total: usable.length};
        }
        """,
        claim_index,
    )


def _scroll_copyright_page_to_bottom(page, max_iters=15, pause=0.4):
    """Scrollea la página /copyright hasta el fondo para forzar lazy render de todos los reclamos."""
    last_count = -1
    for i in range(max_iters):
        try:
            page.evaluate(
                """
                () => {
                  const scrollables = Array.from(document.querySelectorAll('*')).filter(el => {
                    const s = window.getComputedStyle(el);
                    return (s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 10;
                  });
                  for (const el of scrollables) {
                    el.scrollTop = el.scrollHeight;
                  }
                  window.scrollTo(0, document.documentElement.scrollHeight);
                }
                """
            )
        except Exception:
            pass
        time.sleep(pause)
        try:
            count = int(page.evaluate(
                """
                () => {
                  const all = Array.from(document.querySelectorAll('button, ytcp-button-shape, [role="button"], tp-yt-paper-button'));
                  return all.filter(el => {
                    const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                    return t === 'tomar medidas' || t === 'take action';
                  }).length;
                }
                """
            ))
        except Exception:
            count = 0
        if count == last_count and count > 0:
            break
        last_count = count
    try:
        page.evaluate("() => window.scrollTo(0, 0)")
    except Exception:
        pass


def _scan_apelable_take_actions(page, do_click=False):
    """Escanea CTAs 'Tomar medidas' y clasifica por status. Sube niveles desde
    el btn hasta encontrar un container cuyo text matchee algun pattern de status.
    Si do_click=True, clickea el PRIMER apelable. Devuelve dict con conteos + preview de otros."""
    return page.evaluate(
        """
        (doClick) => {
          const all = Array.from(document.querySelectorAll('button, ytcp-button-shape, [role="button"], tp-yt-paper-button'));
          const ctas = all.filter(el => {
            const t = (el.innerText || el.textContent || '').trim().toLowerCase();
            return t === 'tomar medidas' || t === 'take action';
          });
          const apelablesPat = /(impugnaci[oó]n|disputa)\\s*(rechaza|denega|denied|rejected|declined)/i;
          const yaApeladaPat = /(apelaci[oó]n|appeal)\\s+(en\\s+proceso|en\\s+revisi|under\\s+review|caduca|expires|in\\s+review)/i;
          const apelables = [];
          let yaApeladas = 0;
          let otros = 0;
          const otrosPreview = [];
          for (const btn of ctas) {
            const r = btn.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) continue;
            if (btn.disabled === true) continue;
            let status = 'unknown';
            let bestText = '';
            let cur = btn;
            for (let i = 0; i < 15 && cur.parentElement; i++) {
              cur = cur.parentElement;
              const t = (cur.innerText || '').toLowerCase();
              if (t.length > bestText.length) bestText = t;
              if (apelablesPat.test(t)) { status = 'apelable'; break; }
              if (yaApeladaPat.test(t)) { status = 'ya_apelada'; break; }
            }
            if (status === 'apelable') apelables.push(btn);
            else if (status === 'ya_apelada') yaApeladas += 1;
            else {
              otros += 1;
              if (otrosPreview.length < 3) {
                otrosPreview.push((bestText || '').slice(0, 260).replace(/\\s+/g, ' '));
              }
            }
          }
          if (doClick && apelables.length > 0) {
            const target = apelables[0];
            target.scrollIntoView({block: 'center'});
            target.click();
            return {ok: true, apelables_total: apelables.length, ya_apeladas: yaApeladas, otros: otros, otros_preview: otrosPreview};
          }
          return {ok: apelables.length > 0, apelables_total: apelables.length, ya_apeladas: yaApeladas, otros: otros, otros_preview: otrosPreview};
        }
        """,
        do_click,
    )


def _click_next_apelable_take_action(page):
    return _scan_apelable_take_actions(page, do_click=True)


def _count_take_action_buttons(page):
    """Cuenta cuántos botones 'Tomar medidas' habilitados hay en la pagina /copyright.
    Scrollea primero para asegurar lazy render completo."""
    _scroll_copyright_page_to_bottom(page)
    try:
        return int(page.evaluate(
            """
            () => {
              const all = Array.from(document.querySelectorAll(
                'button, ytcp-button-shape, [role="button"], tp-yt-paper-button'
              ));
              return all.filter(el => {
                const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                if (t !== 'tomar medidas' && t !== 'take action') return false;
                const r = el.getBoundingClientRect();
                if (r.width <= 0 || r.height <= 0) return false;
                if (el.disabled === true) return false;
                return true;
              }).length;
            }
            """
        ))
    except Exception:
        return 0


def _force_trusted_checkboxes(page, max_passes=3, ignore_aria=True):
    """Marca checkboxes del dialog activo con page.mouse.click(x,y) en coordenadas reales.
    Es el unico approach trusted server-side para tp-yt-paper-checkbox / ytcp-checkbox.
    Fallback: focus() + Space. Ultimo fallback: box.click() locator."""
    clicks = 0
    for pass_idx in range(max_passes):
        try:
            root = get_active_root(page)
            boxes = root.locator("input[type='checkbox'], tp-yt-paper-checkbox, ytcp-checkbox, [role='checkbox']")
            n = boxes.count()
        except Exception:
            return clicks
        unchecked_in_pass = 0
        for i in range(n):
            try:
                box = boxes.nth(i)
                if not box.is_visible():
                    continue
                is_chk = False
                if not (ignore_aria and pass_idx == 0):
                    try:
                        inner_input = box.locator("input[type='checkbox']").first
                        if inner_input.count() > 0:
                            is_chk = inner_input.is_checked()
                        else:
                            is_chk = box.is_checked()
                    except Exception:
                        try:
                            is_chk = box.is_checked()
                        except Exception:
                            aria = (box.get_attribute("aria-checked") or "").lower()
                            is_chk = aria == "true"
                if is_chk:
                    continue
                unchecked_in_pass += 1
                try:
                    box.scroll_into_view_if_needed()
                except Exception:
                    pass
                time.sleep(0.1)
                clicked = False
                try:
                    bbox = box.bounding_box()
                    if bbox and bbox.get("width", 0) > 4 and bbox.get("height", 0) > 4:
                        cx = bbox["x"] + bbox["width"] / 2
                        cy = bbox["y"] + bbox["height"] / 2
                        page.mouse.move(cx, cy)
                        time.sleep(0.05)
                        page.mouse.click(cx, cy, delay=60)
                        clicked = True
                        clicks += 1
                except Exception:
                    pass
                if not clicked:
                    try:
                        box.focus()
                        time.sleep(0.05)
                        page.keyboard.press("Space")
                        clicked = True
                        clicks += 1
                    except Exception:
                        pass
                if not clicked:
                    try:
                        box.click(timeout=1500)
                        clicks += 1
                    except Exception:
                        pass
                time.sleep(0.2)
            except Exception:
                continue
        if unchecked_in_pass == 0:
            break
        time.sleep(0.3)
    return clicks


def click_enviar_strict(page):
    """Busca un boton 'Enviar'/'Submit'/'Send' EXACTO en <button> nativo enabled y lo
    clickea con mouse.click(x,y) trusted via coords. Devuelve True si clickeo."""
    info = None
    try:
        info = page.evaluate(
            """
            () => {
              const buttons = Array.from(document.querySelectorAll('button'));
              for (const el of buttons) {
                const t = (el.innerText || '').trim().toLowerCase();
                if (!/^enviar$|^submit$|^send$/.test(t)) continue;
                if ((el.tagName || '').toLowerCase() !== 'button') continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) continue;
                const dis = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
                if (dis) continue;
                try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
              }
              return null;
            }
            """
        )
    except Exception:
        info = None
    if not info:
        return False
    try:
        cx = float(info.get("x", 0))
        cy = float(info.get("y", 0))
        if cx > 0 and cy > 0:
            page.mouse.move(cx, cy)
            time.sleep(0.1)
            page.mouse.click(cx, cy, delay=80)
            return True
    except Exception:
        pass
    return False


def select_pais_dropdown(page, pais_value, delay_s=0.5):
    """Maneja el dropdown 'Pais' del modal de apelacion.
    El componente real es <ytcp-dropdown-trigger> con iron-a11y-keys keys="enter":
    necesita FOCUS + ENTER para abrirse (mouse click sobre el host no dispara).
    Luego busca la opcion 'Honduras' en el listbox y la clickea."""
    opened = False
    try:
        focused = page.evaluate(
            """
            () => {
              const isVisible = (n) => {
                const r = n.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) return false;
                const s = window.getComputedStyle(n);
                return s.display !== 'none' && s.visibility !== 'hidden';
              };
              const labelRe = /^pa[ií]s$|^country$/i;
              const labelNodes = Array.from(document.querySelectorAll('label, span, div, p'))
                .filter(n => labelRe.test((n.innerText || '').trim()) && isVisible(n));
              for (const lbl of labelNodes) {
                let cur = lbl;
                for (let i = 0; i < 6 && cur; i++) {
                  const trig = cur.querySelector && cur.querySelector('ytcp-dropdown-trigger, ytcp-text-dropdown-trigger, ytcp-form-select, [role="combobox"]');
                  if (trig && isVisible(trig)) {
                    try { trig.scrollIntoView({block: 'center'}); } catch (e) {}
                    try { trig.focus(); } catch (e) {}
                    return { ok: true, src: 'trigger-focused', tag: (trig.tagName || '').toLowerCase() };
                  }
                  cur = cur.parentElement;
                }
              }
              return { ok: false };
            }
            """
        )
    except Exception:
        focused = {"ok": False}

    if focused.get("ok"):
        print(f"     [pais] focus en {focused.get('tag')} OK. Enviando Enter...")
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass
        time.sleep(max(1.2, delay_s * 1.5))

        try:
            listbox_text = page.evaluate(
                """
                () => {
                  const isVisible = (n) => {
                    const r = n.getBoundingClientRect();
                    if (r.width <= 0 || r.height <= 0) return false;
                    const s = window.getComputedStyle(n);
                    return s.display !== 'none' && s.visibility !== 'hidden';
                  };
                  const lbs = Array.from(document.querySelectorAll('[role="listbox"], tp-yt-paper-listbox, ytcp-text-menu, ytcp-paper-listbox'))
                    .filter(isVisible);
                  if (!lbs.length) return null;
                  return lbs.map(lb => (lb.innerText || '').slice(0, 400)).join(' | ');
                }
                """
            )
            if listbox_text:
                preview = listbox_text[:200].replace("\n", " | ")
                print(f"     [pais debug] listbox abierto via Enter. preview: {preview!r}")
                opened = True
            else:
                print("     [pais debug] Enter no abrio el listbox. Probando Space + ArrowDown...")
        except Exception:
            pass

    if not opened:
        try:
            page.keyboard.press("Space")
            time.sleep(0.4)
            page.keyboard.press("ArrowDown")
            time.sleep(0.6)
            listbox_text = page.evaluate(
                """
                () => {
                  const isVisible = (n) => {
                    const r = n.getBoundingClientRect();
                    return r.width > 0 && r.height > 0 && window.getComputedStyle(n).display !== 'none';
                  };
                  const lbs = Array.from(document.querySelectorAll('[role="listbox"], tp-yt-paper-listbox, ytcp-text-menu, ytcp-paper-listbox'))
                    .filter(isVisible);
                  return lbs.length ? (lbs[0].innerText || '').slice(0, 400) : null;
                }
                """
            )
            if listbox_text:
                print(f"     [pais debug] listbox abierto via Space/ArrowDown.")
                opened = True
        except Exception:
            pass

    if not opened:
        print("     [pais] no logre abrir el dropdown con focus+Enter ni Space/Down.")
        return False

    pat = re.compile(rf"^{re.escape(pais_value)}$", re.IGNORECASE)

    def _try_click_option():
        for strategy in ("role_option", "role_menuitem", "text_exact"):
            try:
                if strategy == "role_option":
                    loc = page.get_by_role("option", name=pat)
                elif strategy == "role_menuitem":
                    loc = page.get_by_role("menuitem", name=pat)
                else:
                    loc = page.get_by_text(pat)
                count = loc.count()
            except Exception:
                count = 0
            for i in range(min(count, 12)):
                try:
                    cand = loc.nth(i)
                    try:
                        cand.scroll_into_view_if_needed(timeout=1200)
                    except Exception:
                        pass
                    time.sleep(0.15)
                    if not cand.is_visible():
                        continue
                    bbox = cand.bounding_box()
                    if not bbox or bbox.get("width", 0) <= 4:
                        continue
                    cx2 = bbox["x"] + bbox["width"] / 2
                    cy2 = bbox["y"] + bbox["height"] / 2
                    page.mouse.move(cx2, cy2)
                    time.sleep(0.08)
                    page.mouse.click(cx2, cy2, delay=70)
                    time.sleep(0.3)
                    return True
                except Exception:
                    continue
        try:
            coords_opt = page.evaluate(
                """
                (target) => {
                  const isVisible = (n) => {
                    const r = n.getBoundingClientRect();
                    if (r.width <= 0 || r.height <= 0) return false;
                    const s = window.getComputedStyle(n);
                    return s.display !== 'none' && s.visibility !== 'hidden';
                  };
                  const wanted = target.toLowerCase();
                  const candidates = Array.from(document.querySelectorAll(
                    '[role="option"], [role="menuitem"], [role="listitem"], paper-item, tp-yt-paper-item, ytcp-text-menu-item, li, div, span'
                  ));
                  for (const el of candidates) {
                    const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (t !== wanted) continue;
                    if (!isVisible(el)) continue;
                    try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                    const r = el.getBoundingClientRect();
                    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
                  }
                  return null;
                }
                """,
                pais_value,
            )
            if coords_opt:
                cx3 = float(coords_opt["x"])
                cy3 = float(coords_opt["y"])
                page.mouse.move(cx3, cy3)
                time.sleep(0.08)
                page.mouse.click(cx3, cy3, delay=70)
                time.sleep(0.3)
                return True
        except Exception:
            pass
        return False

    if _try_click_option():
        return True

    for scroll_pass in range(6):
        try:
            page.evaluate(
                """
                () => {
                  const listboxes = Array.from(document.querySelectorAll(
                    '[role="listbox"], tp-yt-paper-listbox, ytcp-text-dropdown, ytcp-form-select'
                  )).filter(n => {
                    const r = n.getBoundingClientRect();
                    return r.width > 0 && r.height > 0;
                  });
                  for (const lb of listboxes) {
                    lb.scrollBy(0, Math.max(200, Math.floor((lb.clientHeight || 300) * 0.7)));
                  }
                }
                """
            )
        except Exception:
            pass
        time.sleep(0.35)
        if _try_click_option():
            return True

    try:
        page.keyboard.press("H")
        time.sleep(0.3)
        if _try_click_option():
            return True
        for _ in range(8):
            page.keyboard.press("ArrowDown")
            time.sleep(0.1)
        page.keyboard.press("Enter")
        time.sleep(0.4)
    except Exception:
        pass

    print(f"     [pais] no pude clickear la opcion '{pais_value}'.")
    return False


def click_continuar_strict(page):
    """Busca el boton 'Continuar'/'Continue' EXACTO (no parcial) en un <button> nativo
    habilitado y lo clickea con mouse.click(x,y) trusted via coords. Evita clickear
    botones que matcheen 'enviar' (que con P_CONTINUAR antiguo activaban el sidebar
    de feedback). Devuelve True si clickeo, False si no encontro."""
    info = None
    try:
        info = page.evaluate(
            """
            () => {
              const buttons = Array.from(document.querySelectorAll('button'));
              for (const el of buttons) {
                const t = (el.innerText || '').trim().toLowerCase();
                if (!/^continuar$|^continue$|^siguiente$|^next$/.test(t)) continue;
                if ((el.tagName || '').toLowerCase() !== 'button') continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) continue;
                const dis = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
                if (dis) continue;
                try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
              }
              return null;
            }
            """
        )
    except Exception:
        info = None
    if not info:
        return False
    try:
        cx = float(info.get("x", 0))
        cy = float(info.get("y", 0))
        if cx > 0 and cy > 0:
            page.mouse.move(cx, cy)
            time.sleep(0.1)
            page.mouse.click(cx, cy, delay=70)
            return True
    except Exception:
        pass
    return False


def dismiss_feedback_sidebar(page):
    """Cierra el sidebar 'Enviar comentarios' / 'Send feedback' si esta abierto.
    YouTube Studio puede abrirlo por error si algun click previo matcheo 'enviar'."""
    try:
        closed = page.evaluate(
            """
            () => {
              const headings = Array.from(document.querySelectorAll('h1, h2, h3, [role="heading"]'));
              let panel = null;
              for (const h of headings) {
                const t = (h.innerText || '').trim().toLowerCase();
                if (/^enviar comentarios|^send feedback/.test(t)) {
                  panel = h.closest('[role="dialog"], [role="complementary"], aside, section, div');
                  if (panel) break;
                }
              }
              if (!panel) return false;
              const closes = panel.querySelectorAll('button[aria-label*="errar" i], button[aria-label*="close" i], [role="button"][aria-label*="errar" i]');
              for (const btn of closes) {
                try {
                  const r = btn.getBoundingClientRect();
                  if (r.width <= 0) continue;
                  btn.click();
                  return true;
                } catch (e) {}
              }
              return false;
            }
            """
        )
        if closed:
            print("     [info] sidebar 'Enviar comentarios' cerrado.")
        return bool(closed)
    except Exception:
        return False


def detectar_modal_cancelar_apelacion(page):
    """Detecta si el modal actual es 'Cancelar apelacion' (aparece cuando el
    reclamo YA fue apelado y está en revisión). Si lo es, cierra con 'Cerrar'
    para NO cancelar la apelacion previa. Devuelve True si detecto y cerro."""
    try:
        text = (get_active_root(page).inner_text() or "").strip().lower()
    except Exception:
        text = ""
    if not text:
        return False
    is_cancel_modal = (
        "cancelar apelaci" in text
        or "retirar la apelaci" in text
        or "withdraw appeal" in text
        or "withdraw the appeal" in text
        or ("no podr[aá]s volver a apelar" in text)
        or "apelaci[oó]n en proceso de revisi[oó]n" in text
    )
    is_cancel_modal = bool(
        re.search(
            r"cancelar apelaci|retirar la apelaci|withdraw appeal|withdraw the appeal|no podr[aá]s volver a apelar|apelaci[oó]n en proceso de revisi",
            text,
        )
    )
    if not is_cancel_modal:
        return False

    print("     [WARN] modal 'Cancelar apelacion' detectado. SKIP (no cancelar apelacion previa).")
    try:
        coords = page.evaluate(
            """
            () => {
              const buttons = Array.from(document.querySelectorAll('button'));
              for (const el of buttons) {
                const t = (el.innerText || '').trim().toLowerCase();
                if (!/^cerrar$|^close$|^cancelar$|^cancel$/.test(t)) continue;
                if ((el.tagName || '').toLowerCase() !== 'button') continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) continue;
                const dis = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
                if (dis) continue;
                try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
              }
              return null;
            }
            """
        )
    except Exception:
        coords = None
    if coords:
        try:
            cx = float(coords.get("x", 0))
            cy = float(coords.get("y", 0))
            page.mouse.move(cx, cy)
            time.sleep(0.1)
            page.mouse.click(cx, cy, delay=70)
            print("     [WARN] click 'Cerrar' del modal Cancelar apelacion OK.")
            time.sleep(0.5)
        except Exception:
            pass
    return True


def apelar_una_reclamacion(page, delay_s, espera_envio_s, selectors):
    root = page
    selectors = selectors or {}

    if detectar_modal_cancelar_apelacion(page):
        return None

    print("Buscando la accion de apelacion...")

    if not click_with_selector(
        root,
        selectors.get("apelar"),
        [P_APELAR],
        roles=("button", "menuitem", "link"),
        optional=True,
        descripcion="Apelar",
    ):
        click_with_selector(
            root,
            selectors.get("seleccionar_cancion"),
            [P_SELECCIONAR],
            roles=("button", "link", "row"),
            optional=True,
            descripcion="Seleccionar cancion",
        )
        time.sleep(delay_s)
        if not click_with_selector(
            root,
            selectors.get("apelar"),
            [P_APELAR],
            roles=("button", "menuitem", "link"),
            optional=False,
            descripcion="Apelar",
        ):
            return False

    time.sleep(delay_s)
    root = get_active_root(page)

    click_with_selector(
        root,
        selectors.get("apelar_confirmar"),
        [P_APELAR],
        roles=("button", "menuitem", "link"),
        optional=True,
        descripcion="Apelar (confirmacion)",
    )
    time.sleep(delay_s)
    dismiss_feedback_sidebar(page)
    if not click_continuar_strict(page):
        print("Advertencia: no se encontro 'Continuar' (paso post-Apelar). Se intenta seguir.")

    click_with_selector(
        root,
        selectors.get("entiendo_riesgos"),
        [P_ENTIENDO_RIESGOS],
        roles=("checkbox", "button", "radio"),
        optional=True,
        descripcion="Entiendo los riesgos",
    )
    time.sleep(delay_s)
    dismiss_feedback_sidebar(page)
    if not click_continuar_strict(page):
        print("Advertencia: no se encontro 'Continuar' (paso descripcion general). Se intenta seguir.")
    time.sleep(delay_s)

    root = get_active_root(page)

    fields_ok = 0
    fields_fail = 0
    for label, sel_key, pat, valor, press in [
        ("Nombre y apellido", "nombre", P_NOMBRE, nombre_y_apellido, False),
        ("Direccion postal", "direccion", P_DIRECCION, dirreccion_postal, False),
        ("Ciudad", "ciudad", P_CIUDAD, ciudad, False),
        ("Departamento", "departamento", P_DEPARTAMENTO, departamento, False),
        ("Codigo postal", "codigo_postal", P_CODIGO_POSTAL, codigo_postal, False),
    ]:
        if fill_required_field(root, selectors.get(sel_key), [pat], valor, label, press_enter=press, page=page):
            fields_ok += 1
        else:
            fields_fail += 1

    if select_pais_dropdown(page, pais, delay_s=delay_s):
        print(f"     [pais] seleccion '{pais}' OK.")
        fields_ok += 1
    else:
        fields_fail += 1
        print("     [pais] no se pudo seleccionar pais.")

    print(f"     [INFO] campos llenados: {fields_ok}/{fields_ok + fields_fail}. Continuando con el flow.")
    if fields_ok == 0:
        print("Ningun campo se pudo llenar. Abortando este reclamo.")
        return False

    time.sleep(delay_s)
    dismiss_feedback_sidebar(page)
    if not click_continuar_strict(page):
        print("Advertencia: no se encontro 'Continuar' (paso info contacto). Se intenta seguir.")

    root = get_active_root(page)
    if not (fill_with_selector(root, selectors.get("mensaje"), [P_MENSAJE], mensaje) or fill_textarea_fallback(root, mensaje)):
        print("No se encontro el campo de mensaje.")
        return False

    time.sleep(delay_s)
    clicks_check = _force_trusted_checkboxes(page, max_passes=3, ignore_aria=True)
    print(f"     [argumentos] checkboxes clickeados (mouse coords): {clicks_check}")
    time.sleep(0.3)
    clicks_check2 = _force_trusted_checkboxes(page, max_passes=2, ignore_aria=False)
    if clicks_check2 > 0:
        print(f"     [argumentos] checkboxes retry: {clicks_check2}")

    root = get_active_root(page)
    if not (fill_with_selector(root, selectors.get("firma"), [P_FIRMA], firma) or fill_last_textbox_fallback(root, firma)):
        print("No se encontro el campo de firma.")
        return False

    time.sleep(delay_s)
    dismiss_feedback_sidebar(page)
    sent_via = None
    if click_enviar_strict(page):
        sent_via = "enviar"
        print("     [argumentos] click 'Enviar' trusted OK.")
    elif click_continuar_strict(page):
        sent_via = "continuar"
        print("     [argumentos] click 'Continuar' OK (fallback).")
    else:
        print("Advertencia: no se encontro 'Enviar'/'Continuar' (paso final).")
        return False
    time.sleep(espera_envio_s)

    modal_still_open = False
    try:
        modal_still_open = bool(
            page.evaluate(
                """
                () => {
                  const dialogs = Array.from(document.querySelectorAll('[role="dialog"], tp-yt-paper-dialog, ytcp-dialog'));
                  for (const d of dialogs) {
                    const r = d.getBoundingClientRect();
                    if (r.width <= 0 || r.height <= 0) continue;
                    const t = (d.innerText || '').toLowerCase();
                    if (/apelar la decisi[oó]n|appeal the decision|apelar reclamaci[oó]n|appeal claim|apelaci[oó]n.*impugnaci[oó]n/.test(t)) {
                      return true;
                    }
                  }
                  return false;
                }
                """
            )
        )
    except Exception:
        modal_still_open = False

    if modal_still_open:
        print(f"     [post-envio] modal de apelacion sigue abierto tras click '{sent_via}'. Envio NO confirmado.")
        click_with_selector(
            root,
            selectors.get("cerrar"),
            [P_CERRAR],
            roles=("button", "link"),
            optional=True,
            descripcion="Cerrar",
        )
        return False

    print(f"     [post-envio] modal de apelacion cerrado. Envio confirmado.")
    click_with_selector(
        root,
        selectors.get("cerrar"),
        [P_CERRAR],
        roles=("button", "link"),
        optional=True,
        descripcion="Cerrar",
    )
    return True


def parse_filtros_reclamante(raw):
    """Parsea '--filtrar-reclamante' (CSV) a lista normalizada. Vacio -> []."""
    if not raw:
        return []
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _locator_finds_filtros_visible(page_or_root, filtros, max_checks=8):
    """Usa Playwright get_by_text con regex que atraviesa shadow DOM.
    Devuelve True si encuentra al menos un elemento visible con alguno de los filtros."""
    if not filtros:
        return True
    pattern = "|".join(re.escape(f) for f in filtros)
    try:
        loc = page_or_root.get_by_text(re.compile(pattern, re.IGNORECASE))
        count = loc.count()
    except Exception:
        return False
    for i in range(min(count, max_checks)):
        try:
            if loc.nth(i).is_visible():
                return True
        except Exception:
            continue
    return False


def claimant_matches_filter(page, filtros):
    """Devuelve True si el modal/pagina contiene algun nombre del filtro.
    Usa get_by_text (atraviesa shadow DOM) + fallback a inner_text/innerText."""
    if not filtros:
        return True
    if _locator_finds_filtros_visible(page, filtros):
        return True
    try:
        root = get_active_root(page)
        if _locator_finds_filtros_visible(root, filtros):
            return True
        text = (root.inner_text() or "").lower()
    except Exception:
        text = ""
    if not text:
        try:
            text = (page.evaluate("() => (document.body && document.body.innerText) || ''") or "").lower()
        except Exception:
            return False
    return any(f in text for f in filtros)


def page_has_claimant(page, video_id, filtros, delay_s=1.0):
    """Pre-filtro a nivel de pagina /video/X/copyright. Carga la pagina y verifica si
    aparece algun nombre del filtro. Util para skip rapido sin abrir modal."""
    if not filtros:
        return True
    url = f"https://studio.youtube.com/video/{video_id}/copyright"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
    except Exception as exc:
        print(f"     [WARN] no se pudo cargar /copyright: {exc}")
        return True
    try:
        page.wait_for_load_state("networkidle", timeout=12000)
    except Exception:
        pass
    time.sleep(max(2.0, min(delay_s * 1.5, 4.0)))

    if _locator_finds_filtros_visible(page, filtros):
        return True

    try:
        text = (page.evaluate("() => (document.body && document.body.innerText) || ''") or "").lower()
    except Exception:
        text = ""
    if any(f in text for f in filtros):
        return True

    preview = text[:300].replace("\n", " | ")
    print(f"     [DEBUG] /copyright body preview (sin shadow DOM): {preview!r}")
    return False


def procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path):
    if not ensure_content_list(page):
        print("No se pudo abrir la tabla de Contenido en YouTube Studio.")
        return False

    limite_escaneo = args.max_scan or args.max
    fecha_pat = getattr(args, "_solo_fecha_pat", None)
    hora_pat = getattr(args, "_solo_hora_pat", None)
    is_valid = (lambda it: video_fecha_matches(it, fecha_pat, hora_pat)) if fecha_pat else None
    target_valid = args.max if args.max else limite_escaneo

    print("Escaneando videos con restriccion 'Derechos de autor'...")
    items = scan_claim_rows(
        page,
        delay_s=max(0.8, min(args.delay, 1.5)),
        max_items=limite_escaneo,
        is_valid=is_valid,
        target_valid=target_valid,
    )
    save_queue(queue_path, items)

    if not items:
        print("No se detectaron videos con reclamos en la lista actual.")
        return True

    if fecha_pat is not None:
        antes = len(items)
        items_filtrados = [it for it in items if video_fecha_matches(it, fecha_pat, hora_pat)]
        criterio = f"fecha {args.solo_fecha}"
        if hora_pat is not None:
            criterio += f" + hora {args.solo_hora}"
        print(f"Filtrado por {criterio}: {len(items_filtrados)}/{antes} videos matchean.")
        items = items_filtrados

    if not items:
        print("Ningun video matchea el filtro de fecha. Nada que hacer.")
        return True

    processed = load_checkpoint(checkpoint_path)
    pendientes = [item for item in items if processed.get(claim_item_key(item)) != "ok"]
    print(f"Videos detectados (filtrados): {len(items)} | Pendientes: {len(pendientes)}")

    filtros = parse_filtros_reclamante(getattr(args, "filtrar_reclamante", ""))
    if filtros:
        print(f"FILTRO ACTIVO: solo se apelaran reclamos que matcheen {filtros}")

    if args.solo_detectar:
        print(f"Cola guardada en: {queue_path}")
        return True

    total = 0
    saltados_filtro = 0
    errores_consecutivos = 0
    reset_content_scroll(page)

    for index, item in enumerate(pendientes, start=1):
        if args.max and total >= args.max:
            break

        key = claim_item_key(item)
        titulo = item.get("title") or key
        video_id = (item.get("video_id") or "").strip()
        print(f"\n[{index}/{len(pendientes)}] Abriendo reclamo: {titulo}")

        if filtros and video_id:
            if not page_has_claimant(page, video_id, filtros, delay_s=args.delay):
                print(f"  [SKIP] /copyright NO contiene ninguno de {filtros}. Saltando video.")
                item["status"] = "skip_filtro_reclamante"
                save_queue(queue_path, items)
                update_checkpoint(checkpoint_path, processed, key, "skip_filtro_reclamante")
                saltados_filtro += 1
                continue

        if not video_id:
            print("[SKIP] item sin video_id.")
            continue

        copyright_url = f"https://studio.youtube.com/video/{video_id}/copyright"
        try:
            page.goto(copyright_url, wait_until="domcontentloaded", timeout=25000)
        except Exception as exc:
            print(f"[ERROR] no se pudo cargar /copyright: {exc}")
            errores_consecutivos += 1
            if errores_consecutivos >= 3:
                return False
            continue
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        time.sleep(max(1.0, min(args.delay, 2.0)))

        _scroll_copyright_page_to_bottom(page)
        try:
            estado_inicial = _scan_apelable_take_actions(page, do_click=False) or {}
        except Exception:
            estado_inicial = {}
        total_apelables = int(estado_inicial.get("apelables_total", 0))
        total_ya_apeladas = int(estado_inicial.get("ya_apeladas", 0))
        total_otros = int(estado_inicial.get("otros", 0))
        print(f"  Status inicial: apelables={total_apelables} | ya_apeladas={total_ya_apeladas} | otros={total_otros}")
        otros_preview = estado_inicial.get("otros_preview") or []
        for i, prev in enumerate(otros_preview, start=1):
            print(f"     [debug otros #{i}] {prev[:160]!r}")

        if total_apelables == 0:
            print("  [SKIP] no hay reclamos apelables (con 'Impugnación rechazada').")
            item["status"] = "sin_accion"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "sin_accion")
            continue

        apelaciones_video = 0
        saltados_claim = 0
        intentos_sin_progreso = 0
        max_intentos = total_apelables + 5

        for vuelta_claim in range(max_intentos):
            if vuelta_claim > 0:
                try:
                    page.goto(copyright_url, wait_until="domcontentloaded", timeout=25000)
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                time.sleep(max(1.0, min(args.delay, 2.0)))
                _scroll_copyright_page_to_bottom(page)

            result = _click_next_apelable_take_action(page)
            if not result.get("ok"):
                print(f"  No quedan reclamos apelables (ya_apeladas={result.get('ya_apeladas')}, otros={result.get('otros')}). Fin del video.")
                break

            apelables_restantes = int(result.get("apelables_total", 0))
            print(f"\n  [apelable {apelaciones_video+1}] click 'Tomar medidas' OK. Apelables restantes: {apelables_restantes}")
            time.sleep(max(0.8, min(args.delay, 1.5)))

            if filtros and not claimant_matches_filter(page, filtros):
                print(f"  [SKIP] modal NO contiene {filtros}, saltando.")
                saltados_claim += 1
                dismiss_claim_dialog(page, delay_s=args.delay)
                intentos_sin_progreso += 1
                if intentos_sin_progreso >= 3:
                    break
                continue

            ok = apelar_una_reclamacion(page, args.delay, args.espera_envio, selectors)
            if ok is None:
                print(f"  [WARN] modal Cancelar inesperado en row marcada como apelable. SKIP.")
                saltados_claim += 1
                dismiss_claim_dialog(page, delay_s=args.delay)
                intentos_sin_progreso += 1
                if intentos_sin_progreso >= 3:
                    break
            elif ok:
                apelaciones_video += 1
                intentos_sin_progreso = 0
                print(f"  [OK] apelacion {apelaciones_video} enviada.")
            else:
                print(f"  [FALLO] no se pudo apelar.")
                dismiss_claim_dialog(page, delay_s=args.delay)
                intentos_sin_progreso += 1
                if intentos_sin_progreso >= 3:
                    print("  Demasiados fallos consecutivos. Cortando este video.")
                    break

            time.sleep(args.espera_entre)

        print(f"  [SUMMARY video] apelables_inicial={total_apelables} | apeladas={apelaciones_video} | skip_filtro_o_cancelar={saltados_claim} | ya_apeladas_previo={total_ya_apeladas}")

        if apelaciones_video == 0:
            print(f"El video no tenia apelaciones procesables (saltados ya apelados: {saltados_claim}).")
            item["status"] = "sin_accion"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "sin_accion")
            continue

        total += 1
        item["status"] = "ok"
        item["apelaciones_procesadas"] = apelaciones_video
        save_queue(queue_path, items)
        update_checkpoint(checkpoint_path, processed, key, "ok")
        print(f"Apelaciones completadas para el video: {apelaciones_video}")
        time.sleep(args.espera_entre)

    print(f"Proceso automatico finalizado. Videos procesados: {total}")
    if filtros:
        print(f"Videos saltados por filtro de reclamante {filtros}: {saltados_filtro}")
    print(f"Cola actualizada en: {queue_path}")
    return True


def normalize_channel(channel):
    if not channel:
        return None
    normalized = channel.strip().lower()
    if normalized in ("none", "chromium", "bundled"):
        return None
    return normalized


def executable_candidates(channel):
    if not channel:
        return []
    if channel == "chrome":
        return [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ]
    if channel in ("msedge", "edge"):
        return [
            "microsoft-edge",
            "microsoft-edge-stable",
            "msedge",
        ]
    if channel == "chromium":
        return ["chromium", "chromium-browser"]
    return []


def find_browser_executable(channel):
    for candidate in executable_candidates(channel):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_chrome_user_data_dir():
    home = Path.home()
    if sys.platform.startswith("linux"):
        candidates = [
            home / ".config" / "google-chrome",
            home / ".config" / "chromium",
        ]
    elif sys.platform == "darwin":
        candidates = [
            home / "Library" / "Application Support" / "Google" / "Chrome",
            home / "Library" / "Application Support" / "Chromium",
        ]
    else:
        local_appdata = os.environ.get("LOCALAPPDATA")
        candidates = []
        if local_appdata:
            candidates.extend(
                [
                    Path(local_appdata) / "Google" / "Chrome" / "User Data",
                    Path(local_appdata) / "Chromium" / "User Data",
                ]
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def profile_name_from_args(args):
    profile_name = (args.profile or os.environ.get("PLAYWRIGHT_PROFILE_NAME") or "Default").strip()
    return profile_name or "Default"


def resolve_user_data_dir(args):
    profile_name = profile_name_from_args(args)
    source = None
    allow_create = False

    if args.user_data_dir:
        source = describe_profile_source(Path(args.user_data_dir), profile_name=profile_name, name="manual")
        allow_create = True

    env_dir = os.environ.get("PLAYWRIGHT_USER_DATA_DIR")
    if source is None and env_dir:
        source = describe_profile_source(Path(env_dir), profile_name=profile_name, name="env")
        allow_create = True

    env_profile_dir = os.environ.get("PLAYWRIGHT_PROFILE_DIR")
    if source is None and env_profile_dir:
        source = describe_profile_source(Path(env_profile_dir), profile_name=profile_name, name="env")
        allow_create = True

    if source is None and args.usar_perfil_chrome:
        chrome_dir = find_chrome_user_data_dir()
        if not chrome_dir:
            return None, False, None
        source = describe_profile_source(chrome_dir, profile_name=profile_name, name="chrome-local")
        allow_create = False

    if source is None:
        candidates = [describe_profile_source(PLAYWRIGHT_PROFILE_DIR, profile_name=profile_name, name="repo")]
        candidates.append(discover_signed_in_profile(profile_name=profile_name))
        candidates = [item for item in candidates if item is not None]
        if not candidates:
            return None, False, None
        source = max(candidates, key=lambda item: item.get("cookie_count", 0))
        allow_create = source.get("name") == "repo"
        source["detected"] = source.get("name") != "repo"

    launch_dir = source["user_data_dir"]

    try:
        is_repo_profile = launch_dir.resolve() == PLAYWRIGHT_PROFILE_DIR.resolve()
    except Exception:
        is_repo_profile = launch_dir == PLAYWRIGHT_PROFILE_DIR

    if not is_repo_profile and launch_dir.exists():
        runtime_dir = DEFAULT_RUNTIME_PROFILE_DIR / f"{source['name']}_{profile_name.lower().replace(' ', '_')}"
        launch_dir = prepare_runtime_profile(launch_dir, profile_name, runtime_dir)
        source["cloned_from"] = str(source["user_data_dir"])
        source["runtime_user_data_dir"] = launch_dir
        allow_create = False

    return launch_dir, allow_create, source


def resolve_launch_config(args, profile_source=None):
    executable_path = args.executable_path or DEFAULT_EXECUTABLE_PATH
    if executable_path:
        return None, executable_path

    if profile_source and profile_source.get("executable_path"):
        return None, profile_source["executable_path"]

    channel = normalize_channel(args.channel)
    if channel:
        detected = find_browser_executable(channel)
        if detected:
            return None, detected
    return channel, None


def launch_context(playwright, user_data_dir, headless, channel, profile_name, executable_path):
    launch_args = [
        "--start-maximized",
        "--disable-blink-features=AutomationControlled",
    ]
    if profile_name:
        launch_args.append(f"--profile-directory={profile_name}")

    kwargs = dict(
        user_data_dir=str(user_data_dir),
        headless=headless,
        args=launch_args,
        viewport=None,
    )
    if executable_path:
        kwargs["executable_path"] = executable_path
    elif channel:
        kwargs["channel"] = channel
    return playwright.chromium.launch_persistent_context(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Apelar reclamaciones en YouTube Studio usando Playwright.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL inicial de YouTube Studio.")
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin interfaz grafica.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help="Canal del navegador (chrome, msedge). Usa 'none' para Chromium integrado.")
    parser.add_argument("--executable-path", default=None, help="Ruta del ejecutable del navegador (Chromium/Chrome).")
    parser.add_argument("--user-data-dir", default=None, help="Ruta del perfil del navegador para reutilizar sesion.")
    parser.add_argument("--profile", default=os.environ.get("PLAYWRIGHT_PROFILE_NAME"), help="Nombre del perfil dentro del navegador (Default, Profile 1).")
    parser.add_argument("--usar-perfil-chrome", action="store_true", help="Usar el perfil local de Chrome para reutilizar sesion.")
    parser.add_argument("--aprender", action="store_true", help="Grabar selectores manualmente para esta pantalla.")
    parser.add_argument("--selectores", default=None, help="Ruta del archivo JSON de selectores.")
    parser.add_argument("--grabar", action="store_true", help="Grabar todas las acciones para esta pantalla.")
    parser.add_argument("--acciones", default=None, help="Ruta del archivo JSON de acciones grabadas.")
    parser.add_argument("--auto-detect", action="store_true", help="Escanear la tabla de Contenido y abrir automaticamente los videos con reclamos.")
    parser.add_argument("--cola", default=str(DEFAULT_QUEUE_PATH), help="Ruta del JSON con la cola detectada.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Ruta del checkpoint de videos ya procesados.")
    parser.add_argument("--solo-detectar", action="store_true", help="Solo detectar reclamos y guardar la cola sin enviar formularios.")
    parser.add_argument("--max-scan", type=int, default=0, help="Cantidad maxima de videos a detectar en el escaneo automatico (0 = sin limite).")
    parser.add_argument("--reintentos-modal", type=int, default=3, help="Reintentos para abrir el modal de derechos de autor por video.")
    parser.add_argument("--max", type=int, default=0, help="Cantidad de apelaciones a procesar (0 = infinito).")
    parser.add_argument("--delay", type=float, default=1.0, help="Segundos de espera corta entre pasos.")
    parser.add_argument("--espera-envio", type=float, default=6.0, help="Segundos de espera despues de enviar.")
    parser.add_argument("--no-esperar", action="store_true", help="No esperar confirmacion manual al inicio.")
    parser.add_argument("--espera-entre", type=float, default=2.0, help="Segundos de espera entre apelaciones.")
    parser.add_argument(
        "--filtrar-reclamante",
        default="",
        help="Lista CSV de nombres de reclamantes a apelar. Si se especifica, solo apela reclamos cuyo "
             "texto del modal contenga alguno de estos nombres (case-insensitive). "
             "Ejemplo: --filtrar-reclamante 'CD Baby,CD Baby CO'. Vacio = sin filtro.",
    )
    parser.add_argument(
        "--solo-fecha",
        default="",
        help="Fecha marcador en formato YYYY-MM-DD. Si se especifica, SOLO se procesan videos "
             "cuya fila contenga esta fecha (variantes es/en/numericas). Ej: 2028-04-30 "
             "(los videos ya impugnados con fecha marcador para apelar). Vacio = sin filtro.",
    )
    parser.add_argument(
        "--solo-hora",
        default="",
        help="Hora exacta en formato HH:MM (24h). Si se especifica junto a --solo-fecha, "
             "SOLO se procesan videos cuya fila contenga esa hora (variantes 24h y 12h AM/PM). "
             "Ej: 23:45. Vacio = no filtra por hora.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Si se activa, re-escanea y reintenta tras cada vuelta hasta que no queden pendientes "
             "o se alcance --max-loops. Util para ciclo infinito.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=20,
        help="Cantidad maxima de vueltas del loop (default 20).",
    )
    parser.add_argument(
        "--espera-vuelta",
        type=float,
        default=3.0,
        help="Segundos a esperar entre vueltas del loop (default 3.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args._solo_fecha_pat = build_fecha_pattern(getattr(args, "solo_fecha", ""))
    args._solo_hora_pat = build_hora_pattern(getattr(args, "solo_hora", ""))
    if args._solo_fecha_pat is not None:
        msg = f"Filtro activo: SOLO se procesan videos con fecha {args.solo_fecha}"
        if args._solo_hora_pat is not None:
            msg += f" Y hora {args.solo_hora}"
        msg += " (variantes es/en/numericas)."
        print(msg)
    elif args._solo_hora_pat is not None:
        print(f"AVISO: --solo-hora {args.solo_hora} se ignora porque --solo-fecha esta vacio.")
    profile_name = profile_name_from_args(args)
    user_data_dir, allow_create, profile_source = resolve_user_data_dir(args)
    if user_data_dir is None:
        print("No se encontro un perfil local de Chrome. Usa --user-data-dir para indicar la ruta.")
        return
    if not user_data_dir.exists():
        if allow_create:
            user_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"No existe la ruta del perfil: {user_data_dir}")
            return

    channel, executable_path = resolve_launch_config(args, profile_source=profile_source)
    if profile_source and profile_source.get("detected"):
        print(
            f"Perfil detectado automaticamente: {profile_source['name']} "
            f"({profile_source.get('cookie_count', 0)} cookies Google/YouTube)."
        )
    if profile_source and profile_source.get("cloned_from"):
        print(f"Usando copia temporal del perfil: {profile_source['cloned_from']}")
    if args.usar_perfil_chrome:
        print(f"Usando perfil de Chrome: {user_data_dir}")
        print("Cierra Chrome antes de continuar para evitar bloqueo del perfil.")
    if executable_path:
        print(f"Usando ejecutable del navegador: {executable_path}")

    selectors_path = Path(args.selectores) if args.selectores else SELECTORS_PATH
    actions_path = Path(args.acciones) if args.acciones else ACTIONS_PATH
    queue_path = Path(args.cola) if args.cola else DEFAULT_QUEUE_PATH
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_PATH

    with sync_playwright() as p:
        context = launch_context(p, user_data_dir, args.headless, channel, profile_name, executable_path)
        try:
            cookies = load_google_cookies_for_playwright(profile_source, profile_name=profile_name)
            if cookies:
                context.add_cookies(cookies)
                print(f"Cookies de sesion cargadas: {len(cookies)}")

            page = context.new_page()
            page.set_default_timeout(15000)
            page.goto(args.url, wait_until="domcontentloaded")

            if not args.no_esperar:
                if args.auto_detect:
                    input("Inicia sesion en YouTube Studio. El script abrira Contenido y detectara reclamos automaticamente. Presiona Enter para continuar...")
                else:
                    input("Inicia sesion y entra a la pantalla de reclamaciones. Presiona Enter para continuar...")

            if args.grabar:
                record_actions(page, actions_path)
                return

            if args.aprender:
                learn_selectors(page, selectors_path, SELECTOR_STEPS)
                return

            selectors = load_selectors(selectors_path)

            if args.auto_detect:
                if not getattr(args, "loop", False):
                    procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path)
                    return
                max_loops = max(1, int(getattr(args, "max_loops", 20)))
                espera_vuelta = float(getattr(args, "espera_vuelta", 3.0))
                for vuelta in range(1, max_loops + 1):
                    print(f"\n========== LOOP {vuelta}/{max_loops} ==========")
                    procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path)
                    try:
                        with open(queue_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                        pendientes = [
                            it for it in items
                            if it.get("status") not in ("ok", "skip_filtro_reclamante", "sin_accion")
                        ]
                        print(f"Tras vuelta {vuelta}: pendientes = {len(pendientes)} / total items = {len(items)}")
                        if not pendientes:
                            print("No quedan apelaciones pendientes. Terminando loop.")
                            break
                    except Exception as exc:
                        print(f"[WARN] no pude evaluar pendientes ({exc}). Continuando loop.")
                    if vuelta < max_loops:
                        print(f"Esperando {espera_vuelta:.1f}s antes de la siguiente vuelta...")
                        time.sleep(espera_vuelta)
                return

            total = 0
            while True:
                if args.max and total >= args.max:
                    break

                ok = apelar_una_reclamacion(page, args.delay, args.espera_envio, selectors)
                if not ok:
                    print("No se pudo completar la apelacion. Revisa la pantalla y los selectores.")
                    break

                total += 1
                print(f"Apelaciones completadas: {total}")
                time.sleep(args.espera_entre)
        except KeyboardInterrupt:
            print("\nProceso detenido por el usuario.")
        finally:
            context.close()


if __name__ == "__main__":
    main()
