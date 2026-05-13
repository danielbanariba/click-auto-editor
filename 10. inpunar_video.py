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
    dismiss_blocking_overlays,
    dismiss_claim_dialog,
    ensure_content_list,
    load_checkpoint,
    open_claim_modal_with_recovery,
    reset_content_scroll,
    save_queue,
    scan_claim_rows,
    update_checkpoint,
)

mensaje = """Hello,

I have permission/license from the rights holder to use the claimed content in this video and publish it on my YouTube channel.

This upload is authorized. If needed, I can provide additional proof or details of that authorization.

Best regards from Honduras,"""
# mensaje = "Hola, creo y reafirmo firmemente que se trata de un error, ya que solo es un ritmo normal de bateria, no es una pista de audio protegida por derechos de autor, por favor, revisenlo y eliminen la reclamación de derechos de autor. Gracias."
# mensaje = "Hola miembros de arsenal! solo queria pedirles permiso para publicar su album, ya me conocen, soy Daniel Banariba :) el que anda con ustedes a todos lados y que les anda grabando los conciertos."

# Mensaje si se trata de un error evidente de la reclamacion
# mensaje = "Hola, creo y reafirmo firmemente que se trata de un error, ya que no se trata de ninguna cancion, y si se escuchan bien todo es parte de la cancion y solo capta esa parte que no tiene nada que ver con la cancion con la que se esta reclamando, no es una pista de audio protegida por derechos de autor, por favor, revisenlo y eliminen la reclamación de derechos de autor. Gracias."

firma = "Daniel Alejandro Barrientos Anariba"

DEFAULT_URL = os.environ.get("YOUTUBE_STUDIO_URL", "https://studio.youtube.com/")
DEFAULT_CHANNEL = os.environ.get("PLAYWRIGHT_CHANNEL", "chrome")
DEFAULT_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")
DEFAULT_USER_DATA_DIR = os.environ.get("PLAYWRIGHT_USER_DATA_DIR") or str(Path.home() / ".config" / "chromium-pw")
DEFAULT_PROFILE_NAME = os.environ.get("PLAYWRIGHT_PROFILE_NAME", "Default")
SELECTORS_PATH = PLAYWRIGHT_SELECTORS_DIR / "impugnar.json"
ACTIONS_PATH = PLAYWRIGHT_SELECTORS_DIR / "impugnar_acciones.json"
DEFAULT_QUEUE_PATH = Path(__file__).resolve().parent / "data" / "impugnar_claims_queue.json"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "impugnar_claims_checkpoint.txt"
DEFAULT_RUNTIME_PROFILE_DIR = Path(__file__).resolve().parent / ".runtime_browser_profiles" / "impugnar"

P_CONTINUAR = re.compile(r"(continuar|siguiente|next|continue)", re.IGNORECASE)
P_ENVIAR = re.compile(r"(enviar|send|submit)", re.IGNORECASE)
P_PRIMARY_SEND = re.compile(r"^\s*(enviar|send|submit)\s*$", re.IGNORECASE)
P_PRIMARY_CONTINUE = re.compile(r"^\s*(continuar|continue|siguiente|next)\s*$", re.IGNORECASE)
P_FEEDBACK_CONTROL = re.compile(r"(comentario|feedback|captura|screenshot|google)", re.IGNORECASE)
P_IMPUGNAR = re.compile(r"(impugnar|disputar|dispute|take action|tomar medidas)", re.IGNORECASE)
P_IMPUGNAR_CONFIRMAR = re.compile(r"(continuar con.*impugn|confirmar.*impugn|seguir.*impugn|dispute|dispute claim|impugnar)", re.IGNORECASE)
P_DISPUTE_MENU_OPTION = re.compile(r"(impugnar|dispute)", re.IGNORECASE)
P_ACTION_DIALOG_TITLE = re.compile(r"(seleccionar acci[oó]n|select action)", re.IGNORECASE)
P_SELECCIONAR = re.compile(
    r"(seleccionar.*canci[oó]n|select.*song|ver detalles|see details|detalles de la reclamaci[oó]n|reclamaci[oó]n de derechos de autor|copyright claim)",
    re.IGNORECASE,
)
P_VER_DETALLES = re.compile(r"(ver detalles|see details|details)", re.IGNORECASE)
P_LICENCIA = re.compile(r"(licencia|license|permiso|permission)", re.IGNORECASE)
P_ACEPTAR_TERMINOS = re.compile(r"(acepto.*terminos|accept.*terms|agree.*terms|aceptar.*terminos)", re.IGNORECASE)
P_INFO_LICENCIA = re.compile(
    r"(informaci[oó]n.*licencia|license information|detalles.*licencia|describe.*licencia)",
    re.IGNORECASE,
)
P_FIRMA = re.compile(
    r"(firma|signature|nombre completo|nombre y apellido|nombre y apellidos|full name|full legal name)",
    re.IGNORECASE,
)
P_CERRAR = re.compile(r"(cerrar|finalizar|listo|done|close)", re.IGNORECASE)
P_RATIONALE = re.compile(r"(rationale|details|reason|motivo|justificaci[oó]n|razonamiento|detalles)", re.IGNORECASE)
P_REASON = re.compile(r"(reason|motivo|raz[oó]n)", re.IGNORECASE)
P_DETAILS = re.compile(r"(details|detalles)", re.IGNORECASE)
P_TOMAR_MEDIDAS = re.compile(r"(tomar medidas|take action)", re.IGNORECASE)
P_PROGRAMADO = re.compile(r"\b(programado|scheduled)\b", re.IGNORECASE)

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


def build_omitir_fecha_pattern(fecha_iso):
    """Construye un regex que matchea variantes de una o varias fechas ISO (YYYY-MM-DD).

    Acepta una sola fecha o lista CSV: 'YYYY-MM-DD' o 'YYYY-MM-DD,YYYY-MM-DD,...'.
    Cubre formatos en es / en / numeric que YouTube Studio puede mostrar.
    Devuelve None si todas las fechas son vacias o invalidas.
    """
    if not fecha_iso:
        return None
    fecha_iso = fecha_iso.strip()
    if not fecha_iso:
        return None
    fechas = [f.strip() for f in fecha_iso.split(",") if f.strip()]
    if not fechas:
        return None
    from datetime import datetime
    all_variants = []
    for fecha in fechas:
        try:
            dt = datetime.strptime(fecha, "%Y-%m-%d")
        except ValueError:
            print(f"AVISO: --omitir-fecha invalida ({fecha!r}), se ignora. Usa formato YYYY-MM-DD.")
            continue
        d, m, y = dt.day, dt.month, dt.year
        mes_es = _MESES_ES[m - 1]
        mes_en = _MESES_EN[m - 1]
        all_variants.extend([
            rf"\b{d}\s+(?:{mes_es})\.?\s+{y}\b",
            rf"\b(?:{mes_en})\.?\s+{d},?\s+{y}\b",
            rf"\b{d:02d}/{m:02d}/{y}\b",
            rf"\b{d}/{m}/{y}\b",
            rf"\b{y}-{m:02d}-{d:02d}\b",
        ])
    if not all_variants:
        return None
    return re.compile("|".join(all_variants), re.IGNORECASE)


def is_video_marcado_omitir(item, fecha_pat):
    if fecha_pat is None or not isinstance(item, dict):
        return False
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in ("date_text", "row_text", "title")
    )
    return bool(fecha_pat.search(blob))
P_VERIFY_IDENTITY = re.compile(
    r"(verifica que eres t[uú]|verify (it'?s|that) you|confirm.*identity|"
    r"necesitamos confirmar que eres t[uú]|extra security|nivel extra de seguridad)",
    re.IGNORECASE,
)

_VERIFY_IDENTITY_WAIT_S = 0.0  # configurado desde main() via --esperar-verificacion-s


def set_verify_identity_wait(seconds):
    global _VERIFY_IDENTITY_WAIT_S
    _VERIFY_IDENTITY_WAIT_S = max(0.0, float(seconds or 0.0))
P_READ_CONFIRM = re.compile(
    r"(please read.*check the box|read the text above|check the box to continue|marque.*casilla|marca.*casilla|lee.*texto|leer.*arriba)",
    re.IGNORECASE,
)
P_CONFIRM_PERMISSION = re.compile(
    r"(i have permission to use the content|permission to use the content|copyright owner|tengo permiso|permiso para usar|estoy autorizado|me ha dado permiso para usar el contenido|titular de derechos de autor.*permiso)",
    re.IGNORECASE,
)
P_DIALOG_CLAIM_ROOT = re.compile(
    r"(impugnar reclamaci[oó]n|derechos de autor|copyright|tomar medidas|take action|detalles sobre los derechos de autor)",
    re.IGNORECASE,
)
P_ACTION_SELECTOR_DIALOG = re.compile(
    r"(seleccionar acci[oó]n|select action|borrar canci[oó]n|reemplazar canci[oó]n|recortar segmento|impugnar impugna una reclamaci[oó]n)",
    re.IGNORECASE,
)
P_CLAIMS_LIST_DIALOG = re.compile(
    r"(detalles sobre los derechos de autor del video|copyright details|contenido usado)",
    re.IGNORECASE,
)
P_DISPUTE_FLOW_HINTS = [
    "descripción general",
    "descripcion general",
    "motivo",
    "argumentos",
    "firma (obligatorio)",
    "selecciona el motivo principal",
    "select the primary reason",
    "i have permission",
]
P_SUMMARY_MODAL_HINTS = re.compile(r"(¿qu[eé] ha ocurrido\\?|what happened\\?)", re.IGNORECASE)
P_UI_BLOCKER_DIALOG = re.compile(
    r"(enviar comentarios a google|send feedback|haz clic aqu[ií] para a[nñ]adir una captura de pantalla|click here to add a screenshot)",
    re.IGNORECASE,
)

SELECTOR_STEPS = [
    ("seleccionar_cancion", "Selecciona el video o la reclamacion para ver detalles."),
    ("impugnar", "Haz clic en el boton Impugnar / Disputar."),
    ("impugnar_confirmar", "Confirma la impugnacion si aparece un paso extra."),
    ("continuar", "Haz clic en Continuar / Siguiente."),
    ("licencia", "Selecciona la opcion de Licencia o Permiso."),
    ("confirmar_lectura", "Marca la casilla 'I have permission' / lectura."),
    ("aceptar_terminos", "Marca Acepto los terminos (checkbox)."),
    ("info_licencia", "Haz clic en el campo de informacion de tu licencia."),
    ("firma", "Haz clic en el campo de firma."),
    ("cerrar", "Haz clic en Cerrar / Finalizar."),
]


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
    if not info:
        return False
    locator = locator_from_selector_info(root, info)
    if locator is not None:
        try:
            target = locator.first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            try:
                target.click(timeout=min(timeout_ms, 1200))
            except Exception:
                target.click(force=True, timeout=min(timeout_ms, 1200))
            return True
        except Exception:
            pass
    css = info.get("css")
    if css:
        try:
            target = root.locator(css).first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            try:
                target.click(timeout=min(timeout_ms, 1200))
            except Exception:
                target.click(force=True, timeout=min(timeout_ms, 1200))
            return True
        except Exception:
            pass
    if not optional:
        print("No se pudo usar el selector guardado.")
    return False


def fill_by_selector_info(root, info, value, timeout_ms=3000, press_enter=False):
    if not info:
        return False
    locator = locator_from_selector_info(root, info)
    if locator is not None:
        try:
            target = locator.first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.fill(value)
            if press_enter:
                target.press("Enter")
            return True
        except Exception:
            pass
    css = info.get("css")
    if css:
        try:
            target = root.locator(css).first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.fill(value)
            if press_enter:
                target.press("Enter")
            return True
        except Exception:
            pass
    return False


def click_with_selector(root, info, patterns, roles=("button", "menuitem", "link", "radio", "option"), timeout_ms=3000, optional=False, descripcion=None):
    if info and click_by_selector_info(root, info, timeout_ms=timeout_ms, optional=True):
        return True
    return click_action(root, patterns, roles=roles, timeout_ms=timeout_ms, optional=optional, descripcion=descripcion)


def fill_with_selector(root, info, patterns, value, timeout_ms=3000, press_enter=False):
    if info and fill_by_selector_info(root, info, value, timeout_ms=timeout_ms, press_enter=press_enter):
        return True
    return fill_by_label(root, patterns, value, timeout_ms=timeout_ms, press_enter=press_enter)


def click_license_fallback(root):
    try:
        radios = root.locator("tp-yt-paper-radio-button, ytcp-radio-button, [role='radio']")
        count = radios.count()
        for i in range(count):
            item = radios.nth(i)
            try:
                text = item.inner_text().strip().lower()
            except Exception:
                continue
            if (
                "license" in text
                or "licencia" in text
                or "permiso" in text
                or "permission" in text
                or "i have permission" in text
                or "permission to use" in text
            ):
                try:
                    item.scroll_into_view_if_needed()
                    item.click()
                    return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


def wait_for_license_step(root, timeout_ms=10000):
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        try:
            radios = root.locator("tp-yt-paper-radio-button, ytcp-radio-button, [role='radio']")
            count = radios.count()
            for i in range(count):
                radio = radios.nth(i)
                try:
                    if not radio.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    text = radio.inner_text().strip().lower()
                except Exception:
                    text = ""
                if "license" in text or "licencia" in text or "permiso" in text or "permission" in text:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def get_active_root(page):
    try:
        dialogs = page.get_by_role("dialog")
        if dialogs.count() > 0:
            best = None
            best_score = -1
            best_area = -1
            for index in range(dialogs.count() - 1, -1, -1):
                dialog = dialogs.nth(index)
                try:
                    if not dialog.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    text = dialog.inner_text().strip()
                except Exception:
                    text = ""
                if text and P_UI_BLOCKER_DIALOG.search(text):
                    continue
                if text and P_SUMMARY_MODAL_HINTS.search(text):
                    continue
                score = 0
                if text:
                    if P_ACTION_SELECTOR_DIALOG.search(text):
                        score += 6
                    if any(hint in text.lower() for hint in P_DISPUTE_FLOW_HINTS):
                        score += 4
                    if P_DIALOG_CLAIM_ROOT.search(text):
                        score += 2
                try:
                    box = dialog.bounding_box()
                    area = (box or {}).get("width", 0) * (box or {}).get("height", 0)
                except Exception:
                    area = 0
                if score > best_score or (score == best_score and area > best_area):
                    best = dialog
                    best_score = score
                    best_area = area
            if best is not None:
                return best
    except Exception:
        pass
    return page


def get_claims_modal_root(page):
    try:
        dialogs = page.get_by_role("dialog")
        if dialogs.count() == 0:
            return None
        best = None
        best_area = -1
        for index in range(dialogs.count() - 1, -1, -1):
            dialog = dialogs.nth(index)
            try:
                if not dialog.is_visible():
                    continue
            except Exception:
                continue
            try:
                text = dialog.inner_text().strip()
            except Exception:
                text = ""
            if not text:
                continue
            if P_UI_BLOCKER_DIALOG.search(text):
                continue
            lower_text = text.lower()
            if not P_CLAIMS_LIST_DIALOG.search(text):
                continue
            try:
                visible_take_actions = int(
                    dialog.evaluate(
                        """
                        (node) => {
                          const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                          const isVisible = (el) => {
                            if (!el || !el.getBoundingClientRect) return false;
                            const rect = el.getBoundingClientRect();
                            if (rect.width <= 0 || rect.height <= 0) return false;
                            const style = window.getComputedStyle(el);
                            return style.display !== 'none' && style.visibility !== 'hidden';
                          };
                          const controls = Array.from(
                            node.querySelectorAll('button, a, [role=\"button\"], [role=\"link\"], tp-yt-paper-button, ytcp-button-shape')
                          );
                          let count = 0;
                          for (const control of controls) {
                            if (!isVisible(control)) continue;
                            const label = clean(
                              control.innerText ||
                              control.textContent ||
                              control.getAttribute('aria-label') ||
                              control.getAttribute('title') ||
                              ''
                            );
                            if (!label) continue;
                            if (/(tomar medidas|take action)/i.test(label)) {
                              count += 1;
                            }
                          }
                          return count;
                        }
                        """
                    )
                )
            except Exception:
                visible_take_actions = 0
            if visible_take_actions <= 0:
                continue
            try:
                box = dialog.bounding_box()
                area = (box or {}).get("width", 0) * (box or {}).get("height", 0)
            except Exception:
                area = 0
            if area > best_area:
                best = dialog
                best_area = area
        return best
    except Exception:
        return None


def is_claims_list_active(page):
    try:
        root = get_active_root(page)
        text = root.inner_text().strip().lower()
    except Exception:
        return False
    if not text:
        return False
    if not P_CLAIMS_LIST_DIALOG.search(text):
        return False
    if "tomar medidas" in text or "take action" in text:
        return True
    return False


def is_dispute_form_active(page):
    try:
        root = get_active_root(page)
        text = root.inner_text().strip().lower()
    except Exception:
        return False
    if not text:
        return False
    if "impugnar reclamación" in text or "impugnar reclamacion" in text or "dispute" in text:
        if (
            "revisa las siguientes declaraciones" in text
            or "firma (obligatorio)" in text
            or "incluye la información de tu licencia" in text
            or "include your license information" in text
            or "argumentos" in text
        ):
            return True
    score = sum(1 for hint in P_DISPUTE_FLOW_HINTS if hint in text)
    return score >= 2


def click_action(root, patterns, roles=("button", "menuitem", "link", "radio", "option"), timeout_ms=3000, optional=False, descripcion=None):
    for pattern in patterns:
        for role in roles:
            try:
                locator = root.get_by_role(role, name=pattern)
                target = locator.first
                target.wait_for(state="visible", timeout=timeout_ms)
                target.scroll_into_view_if_needed()
                try:
                    target.click(timeout=min(timeout_ms, 1200))
                except Exception:
                    target.click(force=True, timeout=min(timeout_ms, 1200))
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
            try:
                target.click(timeout=min(timeout_ms, 1200))
            except Exception:
                target.click(force=True, timeout=min(timeout_ms, 1200))
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue

    if not optional:
        texto = descripcion or " / ".join([pat.pattern for pat in patterns])
        print(f"No se encontro el elemento: {texto}")
    return False


def click_dialog_primary_button(root, patterns):
    try:
        controls = root.locator("button, [role='button'], tp-yt-paper-button, ytcp-button-shape")
    except Exception:
        return False

    best = None
    best_score = None
    try:
        count = controls.count()
    except Exception:
        return False

    for i in range(count):
        control = controls.nth(i)
        try:
            if not control.is_visible():
                continue
        except Exception:
            continue
        try:
            if hasattr(control, "is_enabled") and not control.is_enabled():
                continue
        except Exception:
            pass
        try:
            label = (
                control.inner_text().strip()
                or (control.get_attribute("aria-label") or "").strip()
                or (control.get_attribute("title") or "").strip()
            )
        except Exception:
            label = ""
        if not label:
            continue
        if P_FEEDBACK_CONTROL.search(label):
            continue
        try:
            host_text = control.evaluate(
                """
                (node) => {
                  const host = node.closest && node.closest('[role="dialog"], tp-yt-paper-dialog, ytcp-dialog, section, div');
                  return String((host && (host.innerText || host.textContent)) || '');
                }
                """
            )
        except Exception:
            host_text = ""
        if host_text and (P_UI_BLOCKER_DIALOG.search(host_text) or P_SUMMARY_MODAL_HINTS.search(host_text)):
            continue
        if not any(pattern.search(label) for pattern in patterns):
            continue
        try:
            box = control.bounding_box() or {}
            score = [box.get("y", 0), box.get("x", 0)]
        except Exception:
            score = [0, 0]
        if host_text and (
            "impugnar reclamación" in host_text.lower()
            or "impugnar reclamacion" in host_text.lower()
            or "firma (obligatorio)" in host_text.lower()
            or "include your license information" in host_text.lower()
            or "incluye la información de tu licencia" in host_text.lower()
            or "revisa las siguientes declaraciones" in host_text.lower()
        ):
            score[0] += 10000
        score = tuple(score)
        if best is None or score > best_score:
            best = control
            best_score = score

    if best is None:
        return False
    try:
        best.scroll_into_view_if_needed()
    except Exception:
        pass
    try:
        best.click(timeout=1500)
        return True
    except Exception:
        try:
            best.click(force=True, timeout=1500)
            return True
        except Exception:
            return False


def dismiss_summary_modal(page):
    try:
        closed = page.evaluate(
            """
            () => {
              const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const isVisible = (node) => {
                if (!node || !node.getBoundingClientRect) return false;
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(node);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const dialogs = Array.from(
                document.querySelectorAll('tp-yt-paper-dialog, ytcp-dialog, [role="dialog"], div, section')
              ).filter((node) => isVisible(node));

              let closedAny = false;
              for (const dialog of dialogs) {
                const text = clean(dialog.innerText || dialog.textContent || '');
                if (!text) continue;
                if (!/(¿?qu[eé] ha ocurrido\\??|what happened\\??|enviar comentarios a google|send feedback)/i.test(text)) {
                  continue;
                }
                const buttons = Array.from(
                  dialog.querySelectorAll('button, [role="button"], tp-yt-paper-icon-button, ytcp-button-shape, ytcp-icon-button')
                );
                for (const btn of buttons) {
                  if (!isVisible(btn)) continue;
                  const label = clean(
                    btn.innerText ||
                    btn.textContent ||
                    btn.getAttribute('aria-label') ||
                    btn.getAttribute('title') ||
                    ''
                  );
                  if (!label) continue;
                  if (/(cerrar|close|dismiss|cancelar|x)/i.test(label)) {
                    try {
                      btn.click();
                      closedAny = true;
                      break;
                    } catch (e) {}
                  }
                }
                if (!closedAny) {
                  try {
                    const closeIcon = dialog.querySelector('ytcp-icon-button, tp-yt-paper-icon-button, [aria-label="Close"], [aria-label="Cerrar"]');
                    if (closeIcon) {
                      closeIcon.click();
                      closedAny = true;
                    }
                  } catch (e) {}
                }
              }
              return closedAny;
            }
            """
        )
        if closed:
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            return True
    except Exception:
        pass
    return False


_SUPPRESS_UI_BLOCKERS = False


def set_suppress_ui_blockers(value):
    global _SUPPRESS_UI_BLOCKERS
    _SUPPRESS_UI_BLOCKERS = bool(value)


def clear_ui_blockers(page, delay_s=0.25, rounds=2):
    if _SUPPRESS_UI_BLOCKERS:
        return False
    closed_any = False
    for _ in range(max(1, rounds)):
        closed_round = False
        try:
            if dismiss_blocking_overlays(
                page,
                delay_s=max(0.08, min(delay_s, 0.35)),
                max_rounds=1,
            ):
                closed_round = True
                closed_any = True
        except Exception:
            pass
        if dismiss_summary_modal(page):
            closed_round = True
            closed_any = True
        if not closed_round:
            break
        time.sleep(max(0.06, min(delay_s, 0.25)))
    return closed_any


def count_take_action_buttons(page, require_modal=True):
    clear_ui_blockers(page, delay_s=0.15, rounds=1)
    root = get_claims_modal_root(page)
    if root is None and require_modal:
        return 0
    root = root or get_active_root(page)
    total = 0
    try:
        candidates = root.locator(
            "button:has-text('Tomar medidas'), "
            "button:has-text('Take action'), "
            "tp-yt-paper-button:has-text('Tomar medidas'), "
            "tp-yt-paper-button:has-text('Take action'), "
            "[role='button']:has-text('Tomar medidas'), "
            "[role='button']:has-text('Take action'), "
            "a:has-text('Tomar medidas'), "
            "a:has-text('Take action')"
        )
        total += candidates.count()
    except Exception:
        pass
    try:
        total += root.get_by_role("button", name=P_TOMAR_MEDIDAS).count()
    except Exception:
        pass
    try:
        total += root.get_by_role("link", name=P_TOMAR_MEDIDAS).count()
    except Exception:
        pass
    try:
        total += root.get_by_text(P_TOMAR_MEDIDAS).count()
    except Exception:
        pass
    if total > 0:
        return total

    try:
        return root.evaluate(
            """
            (node) => {
              const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const target = node || document;
              const roots = [];
              const stack = [target];
              while (stack.length) {
                const current = stack.pop();
                if (!current) continue;
                roots.push(current);
                const all = current.querySelectorAll ? Array.from(current.querySelectorAll('*')) : [];
                for (const el of all) {
                  if (el && el.shadowRoot) stack.push(el.shadowRoot);
                }
              }
              const candidates = [];
              for (const scope of roots) {
                const found = scope.querySelectorAll
                  ? scope.querySelectorAll('button, a, tp-yt-paper-button, [role="button"], [role="link"]')
                  : [];
                for (const item of found) candidates.push(item);
              }
              let count = 0;
              for (const candidate of candidates) {
                const text = clean(
                  candidate.innerText ||
                  candidate.textContent ||
                  candidate.getAttribute('aria-label') ||
                  candidate.getAttribute('title') ||
                  ''
                );
                if (!text) continue;
                if (!/(tomar medidas|take action)/i.test(text)) continue;
                if (candidate.disabled || candidate.getAttribute('aria-disabled') === 'true') continue;
                count += 1;
              }
              return count;
            }
            """
        )
    except Exception:
        return 0


def debug_modal_controls(page, limit=16):
    root = get_claims_modal_root(page) or get_active_root(page)
    try:
        text = (root.inner_text() or "").strip().replace("\n", " ")
        if text:
            print(f"Contexto modal (resumen): {text[:220]}")
    except Exception:
        pass
    try:
        controls = root.evaluate(
            """
            (node) => {
              const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim();
              const isVisible = (el) => {
                if (!el || !el.getBoundingClientRect) return false;
                const rect = el.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const target = node || document;
              const items = Array.from(
                target.querySelectorAll('button, a, [role="button"], [role="link"], tp-yt-paper-button, ytcp-button-shape')
              );
              const out = [];
              for (const item of items) {
                if (!isVisible(item)) continue;
                const label = clean(
                  item.innerText ||
                  item.textContent ||
                  item.getAttribute('aria-label') ||
                  item.getAttribute('title') ||
                  ''
                );
                if (!label) continue;
                const disabled =
                  item.disabled === true ||
                  String(item.getAttribute('aria-disabled') || '').toLowerCase() === 'true';
                out.push({
                  label: label.slice(0, 120),
                  disabled,
                  tag: String(item.tagName || '').toLowerCase(),
                });
              }
              return out.slice(0, 40);
            }
            """
        )
        if isinstance(controls, list) and controls:
            print("Controles visibles detectados en modal:")
            for control in controls[: max(1, limit)]:
                if not isinstance(control, dict):
                    continue
                label = str(control.get("label") or "").strip()
                if not label:
                    continue
                disabled = bool(control.get("disabled"))
                tag = str(control.get("tag") or "")
                state = "disabled" if disabled else "enabled"
                print(f" - [{tag}] {label} ({state})")
    except Exception:
        pass


def is_video_programado(item):
    if not isinstance(item, dict):
        return False
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in ("row_text", "restriction_text", "title")
    )
    return bool(P_PROGRAMADO.search(blob))


def dump_dispute_evidence(page, item, reason, base_dir):
    if base_dir is None:
        return None
    try:
        out_dir = Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"No se pudo crear el dir de evidencia ({base_dir}): {exc}")
        return None

    video_id = (item.get("video_id") if isinstance(item, dict) else None) or "unknown"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"{stamp}_{video_id}_{reason}"

    info = {
        "reason": reason,
        "captured_at": stamp,
        "url": None,
        "modal_open": False,
        "modal_text": "",
        "controls": [],
        "item": item if isinstance(item, dict) else {"raw": str(item)},
    }
    try:
        info["url"] = page.url
    except Exception:
        pass

    try:
        info["modal_open"] = bool(claim_dialog_is_open(page))
    except Exception:
        pass

    try:
        root = get_claims_modal_root(page) or get_active_root(page)
        try:
            info["modal_text"] = (root.inner_text() or "").strip()[:4000]
        except Exception:
            pass
        try:
            html = root.evaluate("(node) => (node && node.outerHTML) || document.documentElement.outerHTML")
            if html:
                base.with_suffix(".html").write_text(html, encoding="utf-8")
        except Exception:
            pass
        try:
            controls = root.evaluate(
                """
                (node) => {
                  const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim();
                  const isVisible = (el) => {
                    if (!el || !el.getBoundingClientRect) return false;
                    const r = el.getBoundingClientRect();
                    if (r.width <= 0 || r.height <= 0) return false;
                    const s = window.getComputedStyle(el);
                    return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
                  };
                  const target = node || document;
                  const items = Array.from(
                    target.querySelectorAll('button, a, [role="button"], [role="link"], tp-yt-paper-button, ytcp-button-shape')
                  );
                  const out = [];
                  for (const it of items) {
                    if (!isVisible(it)) continue;
                    const label = clean(
                      it.innerText || it.textContent ||
                      it.getAttribute('aria-label') || it.getAttribute('title') || ''
                    );
                    if (!label) continue;
                    out.push({
                      label: label.slice(0, 160),
                      tag: String(it.tagName || '').toLowerCase(),
                      disabled: it.disabled === true ||
                        String(it.getAttribute('aria-disabled') || '').toLowerCase() === 'true',
                    });
                  }
                  return out.slice(0, 60);
                }
                """
            )
            if isinstance(controls, list):
                info["controls"] = controls
        except Exception:
            pass
    except Exception:
        pass

    try:
        page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception as exc:
        info["screenshot_error"] = str(exc)

    try:
        base.with_suffix(".json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"No se pudo escribir JSON de evidencia: {exc}")
        return None

    print(f"Evidencia guardada: {base}.[json|html|png] ({reason})")
    return base


def click_take_action_fallback(page, max_rounds=8, delay_s=0.6):
    for _ in range(max_rounds):
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        root = get_claims_modal_root(page) or get_active_root(page)
        try:
            texts = root.locator("text=/tomar medidas|take action/i")
            tcount = texts.count()
            for i in range(min(tcount, 40)):
                target = texts.nth(i)
                try:
                    if not target.is_visible():
                        continue
                except Exception:
                    continue
                clicked = target.evaluate(
                    """
                    (el) => {
                      const isClickable = (node) => {
                        if (!node || !node.matches) return false;
                        return node.matches(
                          'button, a, [role="button"], [role="link"], tp-yt-paper-button, ytcp-button-shape'
                        );
                      };
                      const goUp = (node) => {
                        if (!node) return null;
                        if (node.parentElement) return node.parentElement;
                        const root = node.getRootNode ? node.getRootNode() : null;
                        return root && root.host ? root.host : null;
                      };
                      let cur = el;
                      while (cur) {
                        if (isClickable(cur)) {
                          try { cur.scrollIntoView({ block: 'center' }); } catch (e) {}
                          try { cur.click(); return true; } catch (e) {}
                        }
                        cur = goUp(cur);
                      }
                      return false;
                    }
                    """
                )
                if clicked:
                    return True
        except Exception:
            pass

        try:
            selectors = [
                "button:has-text('Tomar medidas')",
                "button:has-text('Take action')",
                "a:has-text('Tomar medidas')",
                "a:has-text('Take action')",
                "[role='button']:has-text('Tomar medidas')",
                "[role='button']:has-text('Take action')",
                "tp-yt-paper-button:has-text('Tomar medidas')",
                "tp-yt-paper-button:has-text('Take action')",
                "ytcp-button-shape:has-text('Tomar medidas')",
                "ytcp-button-shape:has-text('Take action')",
            ]
            for selector in selectors:
                locator = root.locator(selector)
                count = locator.count()
                if count <= 0:
                    continue
                for i in range(min(count, 30)):
                    target = locator.nth(i)
                    try:
                        if not target.is_visible():
                            continue
                    except Exception:
                        continue
                    try:
                        target.scroll_into_view_if_needed()
                    except Exception:
                        pass
                    try:
                        target.click(timeout=1200)
                        return True
                    except Exception:
                        try:
                            target.click(force=True, timeout=1200)
                            return True
                        except Exception:
                            continue
        except Exception:
            pass

        try:
            candidates = root.locator(
                "button:has-text('Tomar medidas'), "
                "button:has-text('Take action'), "
                "tp-yt-paper-button:has-text('Tomar medidas'), "
                "tp-yt-paper-button:has-text('Take action'), "
                "[role='button']:has-text('Tomar medidas'), "
                "[role='button']:has-text('Take action'), "
                "a:has-text('Tomar medidas'), "
                "a:has-text('Take action')"
            )
            count = candidates.count()
            for i in range(count):
                target = candidates.nth(i)
                try:
                    if not target.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    target.scroll_into_view_if_needed()
                except Exception:
                    pass
                try:
                    target.click()
                    return True
                except Exception:
                    try:
                        target.click(force=True)
                        return True
                    except Exception:
                        continue
        except Exception:
            pass

        if click_action(
            root,
            [P_TOMAR_MEDIDAS],
            roles=("button", "menuitem", "link"),
            timeout_ms=1000,
            optional=True,
        ):
            return True
        try:
            clicked = root.evaluate(
                """
                (node) => {
                  const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const target = node || document;
                  const roots = [];
                  const stack = [target];
                  while (stack.length) {
                    const current = stack.pop();
                    if (!current) continue;
                    roots.push(current);
                    const all = current.querySelectorAll ? Array.from(current.querySelectorAll('*')) : [];
                    for (const el of all) {
                      if (el && el.shadowRoot) stack.push(el.shadowRoot);
                    }
                  }
                  const candidates = [];
                  for (const scope of roots) {
                    const found = scope.querySelectorAll
                      ? scope.querySelectorAll('button, a, tp-yt-paper-button, [role="button"], [role="link"]')
                      : [];
                    for (const item of found) candidates.push(item);
                  }
                  for (const candidate of candidates) {
                    const text = clean(
                      candidate.innerText ||
                      candidate.textContent ||
                      candidate.getAttribute('aria-label') ||
                      candidate.getAttribute('title') ||
                      ''
                    );
                    if (!text) continue;
                    if (!/(tomar medidas|take action)/i.test(text)) continue;
                    if (candidate.disabled || candidate.getAttribute('aria-disabled') === 'true') continue;
                    try { candidate.scrollIntoView({ block: 'center' }); } catch (e) {}
                    try { candidate.click(); return true; } catch (e) {}
                  }
                  if (target && target.scrollHeight > target.clientHeight + 120) {
                    target.scrollBy(0, Math.max(420, Math.floor(target.clientHeight * 0.75)));
                  }
                  return false;
                }
                """
            )
            if clicked:
                return True
        except Exception:
            pass
        time.sleep(delay_s)
    return False


def wait_take_action_count_drop(page, previous_count, timeout_s=16.0):
    if previous_count <= 0:
        return True, count_take_action_buttons(page)
    start = time.time()
    latest = previous_count
    while time.time() - start < timeout_s:
        latest = count_take_action_buttons(page)
        if latest < previous_count:
            return True, latest
        time.sleep(0.5)
    return False, latest


def wait_for_take_action_buttons(page, timeout_s=12.0):
    start = time.time()
    latest = 0
    while time.time() - start < timeout_s:
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        latest = count_take_action_buttons(page, require_modal=False)
        if latest > 0:
            return latest
        try:
            root = get_active_root(page)
            root.evaluate(
                """
                (node) => {
                  if (!node) return;
                  if (node.scrollHeight > node.clientHeight + 120) {
                    node.scrollBy(0, Math.max(400, Math.floor(node.clientHeight * 0.8)));
                    return;
                  }
                  window.scrollBy(0, 420);
                }
                """
            )
        except Exception:
            pass
        time.sleep(0.6)
    return latest


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
    locator = root.locator("textarea")
    candidate = None
    try:
        count = locator.count()
        for i in range(count):
            target = locator.nth(i)
            try:
                if not target.is_visible():
                    continue
            except Exception:
                continue
            aria = (target.get_attribute("aria-label") or "").lower()
            placeholder = (target.get_attribute("placeholder") or "").lower()
            hints = f"{aria} {placeholder}".strip()
            if "signature" in hints or "firma" in hints:
                continue
            if any(token in hints for token in ("specifics", "permission", "license", "licencia", "permiso", "rationale", "justificacion", "justificación", "motivo")):
                candidate = target
                break
            if candidate is None:
                candidate = target
        if candidate is not None:
            candidate.scroll_into_view_if_needed()
            candidate.fill(value)
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


def fill_signature_fallback(root, value):
    locator = root.locator("textarea")
    try:
        count = locator.count()
        for i in range(count):
            target = locator.nth(i)
            try:
                if not target.is_visible():
                    continue
            except Exception:
                continue
            aria = (target.get_attribute("aria-label") or "").lower()
            placeholder = (target.get_attribute("placeholder") or "").lower()
            hints = f"{aria} {placeholder}".strip()
            if "signature" in hints or "firma" in hints:
                target.scroll_into_view_if_needed()
                target.fill(value)
                return True
    except Exception:
        pass
    return False


def fill_signature_via_js(page, value):
    return page.evaluate(
        """
        (signatureValue) => {
          const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
          const dialog =
            document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]') || document;

          const hintTokens = ['firma', 'signature', 'full name', 'nombre completo', 'nombre y apellidos', 'apellidos'];
          const matchesHints = (text) => hintTokens.some((token) => clean(text).includes(token));

          const candidates = Array.from(
            dialog.querySelectorAll('input, textarea, [contenteditable="true"], [role="textbox"]')
          ).filter((el) => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
          });

          const pick = (el) => {
            if (!el) return false;
            const tag = (el.tagName || '').toLowerCase();
            const type = clean(el.getAttribute('type'));
            if (tag === 'input' && type && type !== 'text' && type !== 'search') return false;

            try {
              el.scrollIntoView({ block: 'center' });
            } catch (e) {}

            if (tag === 'input' || tag === 'textarea') {
              el.focus();
              el.value = signatureValue;
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
              return true;
            }

            if (el.getAttribute('contenteditable') === 'true' || clean(el.getAttribute('role')) === 'textbox') {
              el.focus();
              el.textContent = signatureValue;
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
              return true;
            }
            return false;
          };

          for (const el of candidates) {
            const hints = [
              el.getAttribute('aria-label'),
              el.getAttribute('placeholder'),
              el.getAttribute('name'),
              el.id ? ((document.querySelector(`label[for="${el.id}"]`) || {}).textContent || '') : '',
              (el.closest('ytcp-form-input-container, ytcp-form-textarea, div, section') || {}).innerText || '',
            ]
              .filter(Boolean)
              .join(' ');
            if (matchesHints(hints)) {
              if (pick(el)) return true;
            }
          }

          if (candidates.length === 1) {
            return pick(candidates[0]);
          }

          return false;
        }
        """,
        value,
    )


def fill_signature_field(page, root, selectors, value):
    if (
        fill_with_selector(root, selectors.get("firma"), [P_FIRMA], value)
        or fill_signature_fallback(root, value)
        or fill_last_textbox_fallback(root, value)
    ):
        return True

    try:
        if fill_signature_via_js(page, value):
            return True
    except Exception:
        pass
    return False


def click_rationale_tab(root):
    return click_action(
        root,
        [P_RATIONALE],
        roles=("tab", "button", "link", "option"),
        optional=True,
        descripcion="Rationale",
    )


def click_reason_tab(root):
    return click_action(
        root,
        [P_REASON],
        roles=("tab", "button", "link", "option"),
        optional=True,
        descripcion="Reason",
    )


def click_details_tab(root):
    return click_action(
        root,
        [P_DETAILS],
        roles=("tab", "button", "link", "option"),
        optional=True,
        descripcion="Details",
    )


def selector_matches_read(info):
    if not info:
        return False
    role = (info.get("role") or "").lower()
    if role == "checkbox":
        return True
    parts = []
    for key in ("text", "aria_label", "title", "name", "data_testid"):
        value = info.get(key)
        if value:
            parts.append(str(value).lower())
    if not parts:
        return False
    blob = " ".join(parts)
    return bool(re.search(P_READ_CONFIRM, blob) or re.search(P_CONFIRM_PERMISSION, blob))


def click_read_checkbox(root, page=None, selector_info=None):
    patterns = [P_READ_CONFIRM, P_CONFIRM_PERMISSION]
    if selector_info and selector_matches_read(selector_info):
        if click_by_selector_info(root, selector_info, optional=True):
            return True
    if page is not None:
        try:
            page.evaluate(
                """
                () => {
                  const dialog =
                    document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog');
                  if (dialog && dialog.scrollHeight > dialog.clientHeight) {
                    dialog.scrollTo(0, dialog.scrollHeight);
                  }
                }
                """
            )
        except Exception:
            pass
    try:
        checkboxes = root.locator("[role='checkbox'], tp-yt-paper-checkbox, ytcp-checkbox")
        count = checkboxes.count()
        for i in range(count):
            checkbox = checkboxes.nth(i)
            try:
                if not checkbox.is_visible():
                    continue
            except Exception:
                continue
            aria = (checkbox.get_attribute("aria-label") or "").lower()
            if "permiso" in aria or "permission" in aria or "titular de derechos" in aria:
                checkbox.scroll_into_view_if_needed()
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click()
                return True
        if page is not None:
            clicked = page.evaluate(
                """
                () => {
                  const dialog = document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog') || document;
                  const boxes = Array.from(dialog.querySelectorAll('[role="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox'));
                  for (const box of boxes) {
                    const label = (box.getAttribute('aria-label') || '').toLowerCase();
                    if (label.includes('permiso') || label.includes('permission') || label.includes('titular de derechos')) {
                      box.click();
                      return true;
                    }
                  }
                  return false;
                }
                """
            )
            if clicked:
                return True
    except Exception:
        pass
    if click_action(
        root,
        patterns,
        roles=("checkbox", "label", "button", "option", "link"),
        optional=True,
        descripcion="Confirmar lectura",
    ):
        return True

    try:
        checkboxes = root.locator("tp-yt-paper-checkbox, ytcp-checkbox, [role='checkbox']")
        count = checkboxes.count()
        for i in range(count):
            checkbox = checkboxes.nth(i)
            try:
                if not checkbox.is_visible():
                    continue
            except Exception:
                continue
            try:
                text = checkbox.inner_text().strip().lower()
            except Exception:
                text = ""
            aria = (checkbox.get_attribute("aria-label") or "").lower()
            hints = f"{text} {aria}".strip()
            if any(re.search(pattern, hints) for pattern in patterns):
                checkbox.scroll_into_view_if_needed()
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click()
                return True
        for i in range(count):
            checkbox = checkboxes.nth(i)
            try:
                if not checkbox.is_visible():
                    continue
            except Exception:
                continue
            aria = (checkbox.get_attribute("aria-label") or "").lower()
            if re.search(r"(read|check|leer|casilla|confirm)", aria) or any(
                re.search(pattern, aria) for pattern in patterns
            ):
                checkbox.scroll_into_view_if_needed()
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click()
                return True
        if count == 1:
            checkbox = checkboxes.first
            if checkbox.is_visible():
                checkbox.scroll_into_view_if_needed()
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click()
                return True
    except Exception:
        pass
    if page is not None:
        try:
            if click_read_checkbox_js(page, patterns):
                return True
        except Exception:
            pass
    return False


def click_read_checkbox_js(page, patterns):
    compiled = [pattern.pattern for pattern in patterns]
    return page.evaluate(
        """
        (patterns) => {
          const regexes = patterns.map((pattern) => new RegExp(pattern, 'i'));
          const matches = (text) => regexes.some((regex) => regex.test(text));
          const dialog =
            document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog') || document;
          const checkboxes = Array.from(
            dialog.querySelectorAll(
              'input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]'
            )
          );
          for (const cb of checkboxes) {
            let label = cb.getAttribute('aria-label') || '';
            const labelledBy = cb.getAttribute('aria-labelledby');
            if (labelledBy) {
              const ids = labelledBy.split(/\\s+/);
              const labels = ids
                .map((id) => (document.getElementById(id) || {}).textContent || '')
                .join(' ');
              label = `${label} ${labels}`.trim();
            }
            if (label && matches(label)) {
              cb.click();
              return true;
            }
          }
          const texts = Array.from(
            dialog.querySelectorAll(
              'yt-formatted-string, span, div, label, p, ytcp-checkbox, tp-yt-paper-checkbox'
            )
          );
          for (const el of texts) {
            const text = (el.innerText || el.textContent || '').trim();
            if (!text) continue;
            if (!matches(text)) continue;
            let container =
              el.closest('ytcp-checkbox, tp-yt-paper-checkbox, label, div, section') ||
              el.parentElement;
            let checkbox =
              container &&
              container.querySelector(
                'input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]'
              );
            if (!checkbox) {
              checkbox = dialog.querySelector(
                'input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]'
              );
            }
            try {
              el.click();
            } catch (e) {}
            if (checkbox) {
              checkbox.click();
              return true;
            }
          }
          const fallback = Array.from(
            dialog.querySelectorAll('input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]')
          );
          for (const cb of fallback) {
            const checked =
              cb.checked === true ||
              cb.getAttribute('aria-checked') === 'true' ||
              cb.hasAttribute('checked');
            if (!checked) {
              cb.click();
              return true;
            }
          }
          return false;
        }
        """,
        compiled,
    )


def click_dispute_step(page, selectors, timeout_ms=12000):
    def dispute_menu_option_visible():
        try:
            root = get_active_root(page)
            text = root.inner_text().strip().lower()
            if not P_ACTION_DIALOG_TITLE.search(text):
                return False
            if root.get_by_text(P_DISPUTE_MENU_OPTION).count() > 0:
                return True
        except Exception:
            pass
        return False

    def dispute_flow_ready():
        try:
            text = get_active_root(page).inner_text().strip().lower()
        except Exception:
            return False
        if not text:
            return False
        if P_SUMMARY_MODAL_HINTS.search(text):
            return False
        score = sum(1 for hint in P_DISPUTE_FLOW_HINTS if hint in text)
        return score >= 2

    def handle_action_selector_dialog():
        try:
            root = get_active_root(page)
            text = root.inner_text().strip().lower()
        except Exception:
            return False
        if not P_ACTION_DIALOG_TITLE.search(text):
            return False

        clicked_option = False
        try:
            option_selectors = [
                "ytcp-button:has-text('Impugnar')",
                "ytcp-button-shape:has-text('Impugnar')",
                "button:has-text('Impugnar')",
                "[role='button']:has-text('Impugnar')",
                "tp-yt-paper-item:has-text('Impugnar')",
            ]
            for selector in option_selectors:
                locator = root.locator(selector)
                if locator.count() <= 0:
                    continue
                target = locator.first
                if not target.is_visible():
                    continue
                target.scroll_into_view_if_needed()
                try:
                    target.click(timeout=1200)
                except Exception:
                    target.click(force=True, timeout=1200)
                clicked_option = True
                break
        except Exception:
            pass

        if not clicked_option:
            clicked_option = click_action(
                root,
                [re.compile(r"^\\s*(impugnar|dispute)\\b", re.IGNORECASE)],
                roles=("button", "menuitem", "link", "option", "radio"),
                timeout_ms=1000,
                optional=True,
            )

        if not clicked_option:
            return False

        time.sleep(0.6)
        click_action(
            root,
            [P_CONTINUAR],
            roles=("button", "link"),
            timeout_ms=1200,
            optional=True,
        )
        time.sleep(0.8)
        return True

    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        if dispute_flow_ready():
            return True
        clicked = False
        if click_by_selector_info(page, selectors.get("impugnar_confirmar"), timeout_ms=1000, optional=True):
            clicked = True
        elif click_action(
            page,
            [P_DISPUTE_MENU_OPTION],
            roles=("button", "menuitem", "link"),
            timeout_ms=1000,
            optional=True,
        ):
            clicked = True
        elif click_action(
            page,
            [P_IMPUGNAR_CONFIRMAR],
            roles=("button", "menuitem", "link"),
            timeout_ms=1000,
            optional=True,
        ):
            clicked = True

        if clicked:
            time.sleep(0.8)
            if dispute_flow_ready():
                return True
        elif dispute_menu_option_visible():
            if click_action(
                page,
                [P_DISPUTE_MENU_OPTION],
                roles=("button", "menuitem", "link"),
                timeout_ms=1000,
                optional=True,
            ):
                time.sleep(0.8)
                if dispute_flow_ready():
                    return True
        if handle_action_selector_dialog() and dispute_flow_ready():
            return True

        time.sleep(0.4)
    return dispute_flow_ready()


def open_take_action_menu(page, selectors, delay_s, modo_modal=False, max_rounds=None):
    selector_impugnar = None if modo_modal else selectors.get("impugnar")

    def dispute_menu_option_visible():
        try:
            root = get_active_root(page)
            text = root.inner_text().strip().lower()
            if not P_ACTION_DIALOG_TITLE.search(text):
                return False
            if root.get_by_text(P_DISPUTE_MENU_OPTION).count() > 0:
                return True
        except Exception:
            pass
        return False

    if dispute_menu_option_visible():
        print("Modal 'Seleccionar accion' ya abierto; saltando click de 'Tomar medidas'.")
        return True

    rounds = max_rounds if max_rounds is not None else (18 if modo_modal else 8)
    for _ in range(rounds):
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        root = get_claims_modal_root(page) or get_active_root(page)
        clicked = click_with_selector(
            root,
            selector_impugnar,
            [P_IMPUGNAR],
            roles=("button", "menuitem", "link"),
            optional=True,
            descripcion="Impugnar",
        )
        if not clicked:
            clicked = click_take_action_fallback(
                page,
                max_rounds=1,
                delay_s=min(1.0, max(0.5, delay_s)),
            )
        if clicked:
            time.sleep(min(1.0, delay_s))
            if dispute_menu_option_visible():
                return True
        time.sleep(min(0.8, delay_s))

    if modo_modal and count_take_action_buttons(page, require_modal=False) <= 0:
        return None
    print("No se logro abrir 'Tomar medidas' tras varios intentos.")
    debug_modal_controls(page)
    return False


def select_license_option(page, selectors, delay_s):
    for _ in range(3):
        root = get_active_root(page)
        try:
            radios = root.locator("tp-yt-paper-radio-button, ytcp-radio-button, [role='radio']")
            count = radios.count()
            for i in range(count):
                radio = radios.nth(i)
                try:
                    if not radio.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    text = radio.inner_text().strip().lower()
                except Exception:
                    text = ""
                if (
                    "license" in text
                    or "licencia" in text
                    or "permission" in text
                    or "permiso" in text
                    or "i have permission" in text
                    or "permission to use" in text
                ):
                    checked = (radio.get_attribute("aria-checked") or "").lower() == "true"
                    checked = checked or (radio.get_attribute("checked") is not None)
                    if checked:
                        return True
        except Exception:
            pass

        if click_with_selector(
            root,
            selectors.get("licencia"),
            [P_LICENCIA],
            roles=("radio", "button", "option"),
            timeout_ms=4000,
            optional=True,
            descripcion="Licencia",
        ):
            return True
        if click_license_fallback(root) or click_license_fallback(page):
            return True

        click_reason_tab(root)
        time.sleep(delay_s)
        root = get_active_root(page)
        if click_with_selector(
            root,
            selectors.get("licencia"),
            [P_LICENCIA],
            roles=("radio", "button", "option"),
            timeout_ms=4000,
            optional=True,
            descripcion="Licencia",
        ):
            return True
        if click_license_fallback(root) or click_license_fallback(page):
            return True
        time.sleep(delay_s)
    return False


def try_fill_info_licencia(page, root, selectors, mensaje, delay_s):
    if (
        fill_with_selector(root, selectors.get("info_licencia"), [P_INFO_LICENCIA], mensaje)
        or fill_textarea_fallback(root, mensaje)
    ):
        return True

    for _ in range(3):
        click_details_tab(root)
        click_rationale_tab(root)
        time.sleep(delay_s)
        root = get_active_root(page)
        if (
            fill_with_selector(root, selectors.get("info_licencia"), [P_INFO_LICENCIA], mensaje)
            or fill_textarea_fallback(root, mensaje)
        ):
            return True
        try:
            page.mouse.wheel(0, 800)
        except Exception:
            pass
        time.sleep(delay_s)
        root = get_active_root(page)
        if (
            fill_with_selector(root, selectors.get("info_licencia"), [P_INFO_LICENCIA], mensaje)
            or fill_textarea_fallback(root, mensaje)
        ):
            return True

        click_repeatedly(root, [P_CONTINUAR], max_clicks=1, selector_info=selectors.get("continuar"))
        time.sleep(delay_s)
        root = get_active_root(page)
        if (
            fill_with_selector(root, selectors.get("info_licencia"), [P_INFO_LICENCIA], mensaje)
            or fill_textarea_fallback(root, mensaje)
        ):
            return True

    return False


def click_continue_step(page, selectors, delay_s):
    for _ in range(2):
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        root = get_active_root(page)
        try:
            before_text = root.inner_text()[:3000]
        except Exception:
            before_text = ""
        before_lower = before_text.lower()
        locator = locator_from_selector_info(root, selectors.get("continuar"))
        if locator is None:
            try:
                locator = root.get_by_role("button", name=P_CONTINUAR)
            except Exception:
                locator = None
        if locator is None:
            try:
                locator = root.locator("button[aria-label='Continuar'], button[aria-label='Continue'], #continue-button button")
            except Exception:
                locator = None
        if locator is not None:
            try:
                target = locator.first
                target.wait_for(state="visible", timeout=2000)
                try:
                    if not target.is_enabled():
                        click_read_checkbox(root, page)
                        time.sleep(delay_s)
                except Exception:
                    pass
                target.scroll_into_view_if_needed()
                if "selecciona el motivo principal" in before_lower or "descripcion general" in before_lower or "descripción general" in before_lower:
                    target = root.locator("button[aria-label='Continuar'], button[aria-label='Continue'], #continue-button button").first
                    target.wait_for(state="visible", timeout=2000)
                    target.scroll_into_view_if_needed()
                    target.focus()
                    target.press("Enter")
                    time.sleep(0.8)
                    return True
                try:
                    target.click(timeout=1200)
                except Exception:
                    target.click(force=True, timeout=1200)
                time.sleep(0.8)
                try:
                    after_text = get_active_root(page).inner_text()[:3000]
                except Exception:
                    after_text = ""
                if after_text == before_text:
                    try:
                        target.focus()
                        page.keyboard.press("Enter")
                        time.sleep(0.8)
                    except Exception:
                        pass
                return True
            except Exception:
                pass
        if click_action(root, [P_CONTINUAR], roles=("button", "link"), timeout_ms=2000, optional=True):
            return True
        click_read_checkbox(root, page)
        time.sleep(delay_s)
    return False


def finalize_dispute_submission(page, selectors, delay_s, rounds=6):
    clicked_any = False
    click_count = 0
    for _ in range(rounds):
        clear_ui_blockers(page, delay_s=min(0.2, delay_s), rounds=1)
        if clicked_any and not is_dispute_form_active(page):
            return True

        root = get_active_root(page)
        dispute_before = is_dispute_form_active(page)
        clicked_send = click_dialog_primary_button(root, [P_PRIMARY_SEND])
        if clicked_send:
            print("Finalizar: clic en 'Enviar'.")
            if dispute_before:
                clicked_any = True
            click_count += 1
            time.sleep(max(0.7, delay_s))
            clear_ui_blockers(page, delay_s=min(0.2, delay_s), rounds=1)
            if clicked_any and not is_dispute_form_active(page):
                return True
            continue

        clicked_continue = click_dialog_primary_button(root, [P_PRIMARY_CONTINUE])
        if not clicked_continue:
            clicked_continue = click_with_selector(
                root,
                selectors.get("continuar"),
                [P_CONTINUAR],
                roles=("button", "link"),
                timeout_ms=1200,
                optional=True,
                descripcion="Continuar",
            )
        if clicked_continue:
            print("Finalizar: clic en 'Continuar'.")
            clicked_any = True
            time.sleep(max(0.7, delay_s))
            continue

        if clicked_any and not is_dispute_form_active(page):
            return True

        try:
            snippet = (root.inner_text() or "").strip().replace("\n", " ")
            if snippet:
                print(f"Finalizar: sin boton claro. Contexto: {snippet[:180]}")
        except Exception:
            pass
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass
        time.sleep(max(0.6, delay_s))

    if clicked_any:
        print(
            "Finalizar: no se confirmo cierre inmediato del formulario; "
            "se continuara con validacion por recuento en el modal."
        )
        return True

    if click_count > 0:
        print("Finalizar: se detectaron clics, pero fuera del formulario esperado.")
    return False


def press_continue_step(page, timeout_ms=2000):
    try:
        target = page.get_by_role("dialog").last.locator(
            "button[aria-label='Continuar'], button[aria-label='Continue'], #continue-button button"
        ).first
        target.wait_for(state="visible", timeout=timeout_ms)
        target.scroll_into_view_if_needed()
        target.focus()
        target.press("Enter")
        return True
    except Exception:
        return False


def click_permission_checkbox(page, timeout_ms=2000):
    try:
        checkbox = page.get_by_role("dialog").last.locator("[role='checkbox']").first
        checkbox.wait_for(state="visible", timeout=timeout_ms)
        checkbox.scroll_into_view_if_needed()
        checkbox.click(force=True)
        return True
    except Exception:
        return False


def advance_to_reason_step(page, selectors, delay_s):
    for _ in range(4):
        clear_ui_blockers(page, delay_s=0.12, rounds=1)
        root = get_active_root(page)
        if wait_for_license_step(root, timeout_ms=1200):
            return True

        try:
            text = root.inner_text().lower()
        except Exception:
            text = ""

        if "seleccionar accion" in text or "seleccionar acción" in text:
            click_continue_step(page, selectors, delay_s)
        elif "descripcion general" in text or "descripción general" in text:
            press_continue_step(page)
        else:
            click_continue_step(page, selectors, delay_s)

        time.sleep(delay_s)

    return wait_for_license_step(get_active_root(page), timeout_ms=1500)


def handle_options_step(page, selectors, delay_s):
    """Step 'Opciones' (Impugnar vs Apelar sin impugnar) que YouTube inserta
    entre 'Motivo' y 'Detalles' cuando hay flujo extendido. Confirma 'Impugnar'
    y avanza con Continuar. Devuelve True si actuó, False si no era ese step."""
    appeal_visible = False
    try:
        appeal_locator = page.get_by_text(re.compile(r"apelar sin impugnar|appeal without disputing", re.IGNORECASE))
        ct = appeal_locator.count()
        for i in range(min(ct, 8)):
            try:
                if appeal_locator.nth(i).is_visible():
                    appeal_visible = True
                    break
            except Exception:
                continue
    except Exception:
        appeal_visible = False

    if not appeal_visible:
        text_root = ""
        try:
            root_dbg = get_active_root(page)
            text_root = (root_dbg.inner_text() or "").strip().lower()
        except Exception:
            text_root = ""
        preview = text_root[:140].replace("\n", " | ")
        print(f"   [check Opciones] no es ese step (no encuentro 'Apelar sin impugnar'). root: {preview!r}")
        return False

    try:
        root = get_active_root(page)
    except Exception:
        root = page

    print("Paso 3.5/7: detectado step 'Opciones' (Impugnar vs Apelar). Confirmando 'Impugnar'.")

    selected_dispute = False
    for scope in (root, page):
        if selected_dispute:
            break
        try:
            radios = scope.locator("tp-yt-paper-radio-button, ytcp-radio-button, [role='radio']")
            n = radios.count()
        except Exception:
            continue
        for i in range(n):
            r = radios.nth(i)
            try:
                if not r.is_visible():
                    continue
                label = (r.inner_text() or "").lower().strip()
                aria = (r.get_attribute("aria-label") or "").lower()
                hint = f"{label} {aria}"
                if "apelar sin impugnar" in hint or "appeal without disputing" in hint:
                    continue
                if not (re.search(r"\bimpugnar\b", hint) or re.search(r"\bdispute\b", hint)):
                    continue
                already = (r.get_attribute("aria-checked") or "").lower() == "true"
                if not already:
                    try:
                        r.scroll_into_view_if_needed()
                        time.sleep(0.1)
                        bbox = r.bounding_box()
                        if bbox and bbox.get("width", 0) > 4 and bbox.get("height", 0) > 4:
                            cx = bbox["x"] + bbox["width"] / 2
                            cy = bbox["y"] + bbox["height"] / 2
                            page.mouse.move(cx, cy)
                            time.sleep(0.05)
                            page.mouse.click(cx, cy, delay=60)
                        else:
                            r.click()
                    except Exception:
                        try:
                            r.click(force=True)
                        except Exception:
                            pass
                selected_dispute = True
                break
            except Exception:
                continue

    if not selected_dispute:
        print("   [Opciones] no encontre radio 'Impugnar' — confiando en el default e intentando Continuar.")

    time.sleep(delay_s)
    if not (press_continue_step(page) or click_continue_step(page, selectors, delay_s)):
        print("Advertencia: no se pudo avanzar desde el step 'Opciones'.")
        return False
    time.sleep(delay_s)
    return True


def check_all_visible_checkboxes(root, page=None):
    checked = 0
    seen = set()
    try:
        checkboxes = root.locator("input[type='checkbox'], tp-yt-paper-checkbox, ytcp-checkbox, [role='checkbox']")
        for i in range(checkboxes.count()):
            checkbox = checkboxes.nth(i)
            try:
                if not checkbox.is_visible():
                    continue
                handle = checkbox.element_handle()
                if handle:
                    key = str(handle)
                    if key in seen:
                        continue
                    seen.add(key)
                checkbox.scroll_into_view_if_needed()
                try:
                    already_checked = checkbox.is_checked()
                except Exception:
                    aria_checked = (checkbox.get_attribute("aria-checked") or "").lower()
                    already_checked = aria_checked == "true" or checkbox.get_attribute("checked") is not None
                if already_checked:
                    continue
                try:
                    checkbox.check()
                except Exception:
                    checkbox.click(force=True)
                checked += 1
            except Exception:
                continue
    except Exception:
        pass

    if checked > 0:
        return checked

    if page is not None:
        try:
            clicked = page.evaluate(
                """
                () => {
                  const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const dialog =
                    document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]') || document;
                  const dialogText = clean(dialog.innerText || dialog.textContent || '');
                  const declarationStep =
                    dialogText.includes('revisa las siguientes declaraciones') ||
                    dialogText.includes('check the boxes to confirm') ||
                    dialogText.includes('declaraciones') ||
                    dialogText.includes('fraudulentas');

                  const boxes = Array.from(
                    dialog.querySelectorAll('input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]')
                  );
                  let count = 0;

                  const isChecked = (node) => {
                    const aria = clean(node.getAttribute && node.getAttribute('aria-checked'));
                    if (aria === 'true') return true;
                    if (node.checked === true) return true;
                    if (node.hasAttribute && node.hasAttribute('checked')) return true;
                    const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                    if (input && input.checked) return true;
                    return false;
                  };

                  for (const box of boxes) {
                    if (declarationStep && isChecked(box)) continue;
                    if (!declarationStep && isChecked(box)) continue;
                    try {
                      box.scrollIntoView({ block: 'center' });
                    } catch (e) {}
                    try {
                      box.click();
                      count += 1;
                    } catch (e) {}
                  }
                  return count;
                }
                """
            )
            if isinstance(clicked, int):
                checked += clicked
        except Exception:
            pass

    return checked


def count_unchecked_dialog_checkboxes(page):
    try:
        return int(
            page.evaluate(
                """
                () => {
                  const dialog =
                    document.querySelector('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]') || document;
                  const nodes = Array.from(
                    dialog.querySelectorAll('input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]')
                  );
                  const isVisible = (node) => {
                    if (!node || !node.getBoundingClientRect) return false;
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) return false;
                    const style = window.getComputedStyle(node);
                    return style.display !== 'none' && style.visibility !== 'hidden';
                  };
                  const isChecked = (node) => {
                    const aria = String(node.getAttribute && node.getAttribute('aria-checked') || '').toLowerCase();
                    if (aria === 'true') return true;
                    if (node.checked === true) return true;
                    if (node.hasAttribute && node.hasAttribute('checked')) return true;
                    const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                    return !!(input && input.checked);
                  };
                  let unchecked = 0;
                  for (const node of nodes) {
                    if (!isVisible(node)) continue;
                    if (!isChecked(node)) unchecked += 1;
                  }
                  return unchecked;
                }
                """
            )
        )
    except Exception:
        return 0


def ensure_all_dialog_checkboxes_checked(page):
    try:
        result = page.evaluate(
            """
            () => {
              const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const isVisible = (node) => {
                if (!node || !node.getBoundingClientRect) return false;
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(node);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const dialogs = Array.from(
                document.querySelectorAll('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]')
              ).filter((node) => isVisible(node));
              let dialog = null;
              for (const node of dialogs) {
                const text = clean(node.innerText || node.textContent || '');
                if (!text) continue;
                if (/enviar comentarios a google|send feedback/i.test(text)) continue;
                if (
                  /impugnar reclamaci[oó]n|revisa las siguientes declaraciones|firma \\(obligatorio\\)|license|licencia|argumentos/i.test(text)
                ) {
                  dialog = node;
                  break;
                }
              }
              if (!dialog) {
                dialog = dialogs[0] || document;
              }

              const isChecked = (node) => {
                const aria = String(node.getAttribute && node.getAttribute('aria-checked') || '').toLowerCase();
                if (aria === 'true') return true;
                if (node.checked === true) return true;
                if (node.hasAttribute && node.hasAttribute('checked')) return true;
                const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                return !!(input && input.checked);
              };

              const clickNode = (node) => {
                try { node.scrollIntoView({ block: 'center' }); } catch (e) {}
                try { node.click(); return true; } catch (e) {}
                const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                if (input) {
                  try { input.click(); return true; } catch (e) {}
                  try {
                    input.checked = true;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                  } catch (e) {}
                }
                try {
                  if (node.setAttribute) {
                    node.setAttribute('aria-checked', 'true');
                  }
                  node.dispatchEvent(new Event('input', { bubbles: true }));
                  node.dispatchEvent(new Event('change', { bubbles: true }));
                  node.dispatchEvent(new Event('click', { bubbles: true }));
                  return true;
                } catch (e) {}
                return false;
              };

              const collectBoxes = () =>
                Array.from(
                  dialog.querySelectorAll('input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]')
                ).filter((node) => {
                  if (isVisible(node)) return true;
                  const host =
                    (node.closest &&
                      node.closest('ytcp-checkbox, tp-yt-paper-checkbox, [role="checkbox"], label, li, div, section')) ||
                    null;
                  return !!(host && isVisible(host));
                });

              for (let pass = 0; pass < 4; pass += 1) {
                const boxes = collectBoxes();
                for (const box of boxes) {
                  if (!isChecked(box)) {
                    clickNode(box);
                    if (!isChecked(box)) {
                      clickNode(box);
                    }
                  }
                }
                try {
                  const step = Math.max(260, Math.floor((dialog.clientHeight || 600) * 0.65));
                  dialog.scrollBy(0, step);
                } catch (e) {}
              }

              const clickDeclarationByText = (regex) => {
                const nodes = Array.from(
                  dialog.querySelectorAll('label, div, span, p, li, yt-formatted-string, ytcp-checkbox, tp-yt-paper-checkbox')
                );
                let bestNode = null;
                let bestTextLen = Number.POSITIVE_INFINITY;
                for (const node of nodes) {
                  const text = clean(node.innerText || node.textContent || '');
                  if (!text) continue;
                  if (!regex.test(text)) continue;
                  if (text.length < bestTextLen) {
                    bestNode = node;
                    bestTextLen = text.length;
                  }
                }
                if (!bestNode) return false;
                try { bestNode.scrollIntoView({ block: 'center' }); } catch (e) {}
                try { bestNode.click(); } catch (e) {}

                let targetY = 0;
                try {
                  const rect = bestNode.getBoundingClientRect();
                  targetY = rect.top + rect.height / 2;
                } catch (e) {}

                const boxes = collectBoxes().filter((box) => !isChecked(box));
                if (boxes.length <= 0) return true;
                let bestBox = null;
                let bestDist = Number.POSITIVE_INFINITY;
                for (const box of boxes) {
                  let boxY = 0;
                  try {
                    const rect = box.getBoundingClientRect();
                    boxY = rect.top + rect.height / 2;
                  } catch (e) {}
                  const dist = Math.abs(boxY - targetY);
                  if (dist < bestDist) {
                    bestDist = dist;
                    bestBox = box;
                  }
                }
                if (bestBox) {
                  clickNode(bestBox);
                  if (!isChecked(bestBox)) clickNode(bestBox);
                }
                return true;
              };

              const clickCheckboxByContext = (regex) => {
                const boxes = collectBoxes();
                for (const box of boxes) {
                  const container =
                    (box.closest &&
                      box.closest('label, li, tr, ytcp-checkbox, tp-yt-paper-checkbox, div, section')) ||
                    null;
                  const context = clean(
                    (
                      (container && container.innerText) ||
                      ''
                    )
                  );
                  if (!context) continue;
                  if (!regex.test(context)) continue;
                  if (!isChecked(box)) {
                    if (container) {
                      try { container.scrollIntoView({ block: 'center' }); } catch (e) {}
                      try { container.click(); } catch (e) {}
                    }
                    clickNode(box);
                    if (!isChecked(box)) {
                      if (container) {
                        try { container.click(); } catch (e) {}
                      }
                      clickNode(box);
                    }
                  }
                }
              };

              try { dialog.scrollTo(0, dialog.scrollHeight); } catch (e) {}
              clickDeclarationByText(/mi v[ií]deo no infringe|my video does not infringe/i);
              clickDeclarationByText(/reclamante podr[aá] revisar|claimant may review|may review my video/i);
              clickDeclarationByText(/impugnaciones fraudulentas|fraudulent disputes|termination of my youtube account/i);
              clickCheckboxByContext(/impugnaciones fraudulentas|fraudulent disputes|termination of my youtube account/i);
              clickCheckboxByContext(/reclamante podr[aá] revisar|claimant may review|may review my video/i);
              clickCheckboxByContext(/mi v[ií]deo no infringe|my video does not infringe/i);

              for (const box of collectBoxes()) {
                if (!isChecked(box)) {
                  const container =
                    (box.closest &&
                      box.closest('label, li, tr, ytcp-checkbox, tp-yt-paper-checkbox, div, section')) ||
                    null;
                  if (container) {
                    try { container.scrollIntoView({ block: 'center' }); } catch (e) {}
                    try { container.click(); } catch (e) {}
                  }
                  clickNode(box);
                  if (!isChecked(box)) {
                    if (container) {
                      try { container.click(); } catch (e) {}
                    }
                    clickNode(box);
                  }
                }
              }

              const boxes = collectBoxes();
              const uncheckedLabels = [];
              let unchecked = 0;
              for (const box of boxes) {
                if (!isChecked(box)) {
                  unchecked += 1;
                  const host =
                    (box.closest &&
                      box.closest('ytcp-checkbox, tp-yt-paper-checkbox, [role="checkbox"], label, li, div, section')) ||
                    null;
                  const disabled = String(
                    box.getAttribute && box.getAttribute('aria-disabled') ||
                    box.getAttribute && box.getAttribute('disabled') ||
                    (host && host.getAttribute && (host.getAttribute('aria-disabled') || host.getAttribute('disabled'))) ||
                    ''
                  ).toLowerCase();
                  const label = clean(
                    box.getAttribute && box.getAttribute('aria-label') ||
                    box.innerText ||
                    box.textContent ||
                    (box.closest && box.closest('label, div, section') && box.closest('label, div, section').innerText) ||
                    ''
                  );
                  const text = label ? label.slice(0, 160) : '(sin etiqueta)';
                  uncheckedLabels.push(disabled ? `${text} [disabled=${disabled}]` : text);
                }
              }
              return { total: boxes.length, unchecked, unchecked_labels: uncheckedLabels.slice(0, 6) };
            }
            """
        )
        if isinstance(result, dict):
            total = int(result.get("total", 0))
            unchecked = int(result.get("unchecked", 0))
            unchecked_labels = result.get("unchecked_labels")
            if not isinstance(unchecked_labels, list):
                unchecked_labels = []
            return {"total": total, "unchecked": unchecked, "unchecked_labels": unchecked_labels}
    except Exception:
        pass
    return {"total": 0, "unchecked": 0, "unchecked_labels": []}


def click_declaration_checkbox_by_pattern(page, pattern):
    try:
        result = page.evaluate(
            """
            (pattern) => {
              const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const regex = new RegExp(pattern, 'i');
              const isVisible = (node) => {
                if (!node || !node.getBoundingClientRect) return false;
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(node);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const dialogs = Array.from(
                document.querySelectorAll('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]')
              ).filter((node) => isVisible(node));

              let dialog = null;
              for (const node of dialogs) {
                const text = clean(node.innerText || node.textContent || '');
                if (!text) continue;
                if (/enviar comentarios a google|send feedback/i.test(text)) continue;
                if (/impugnar reclamaci[oó]n|revisa las siguientes declaraciones|firma \\(obligatorio\\)|argumentos/i.test(text)) {
                  dialog = node;
                  break;
                }
              }
              if (!dialog) dialog = dialogs[0] || document;

              const isChecked = (node) => {
                const aria = String(node.getAttribute && node.getAttribute('aria-checked') || '').toLowerCase();
                if (aria === 'true') return true;
                if (node.checked === true) return true;
                if (node.hasAttribute && node.hasAttribute('checked')) return true;
                const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                return !!(input && input.checked);
              };

              const clickNode = (node) => {
                if (!node) return false;
                try { node.scrollIntoView({ block: 'center' }); } catch (e) {}
                try { node.click(); return true; } catch (e) {}
                const input = node.querySelector && node.querySelector('input[type="checkbox"]');
                if (input) {
                  try { input.click(); return true; } catch (e) {}
                }
                return false;
              };

              const nodeCenterY = (node) => {
                try {
                  const rect = node.getBoundingClientRect();
                  return rect.top + rect.height / 2;
                } catch (e) {
                  return 0;
                }
              };

              const textNodes = Array.from(
                dialog.querySelectorAll('label, div, span, p, li, yt-formatted-string, ytcp-checkbox, tp-yt-paper-checkbox')
              );
              let targetNode = null;
              let targetLen = Number.POSITIVE_INFINITY;
              for (const node of textNodes) {
                const text = clean(node.innerText || node.textContent || '');
                if (!text) continue;
                if (!regex.test(text)) continue;
                if (text.length < targetLen) {
                  targetNode = node;
                  targetLen = text.length;
                }
              }
              if (!targetNode) {
                return { matched: false, clicked: false, checked: false };
              }

              try { targetNode.scrollIntoView({ block: 'center' }); } catch (e) {}

              const boxes = Array.from(
                dialog.querySelectorAll('input[type="checkbox"], tp-yt-paper-checkbox, ytcp-checkbox, [role="checkbox"]')
              ).filter((node) => {
                if (isVisible(node)) return true;
                const host =
                  (node.closest &&
                    node.closest('ytcp-checkbox, tp-yt-paper-checkbox, [role="checkbox"], label, li, div, section')) ||
                  null;
                return !!(host && isVisible(host));
              });
              if (!boxes.length) {
                return { matched: true, clicked: false, checked: false };
              }

              let best = null;
              let bestDist = Number.POSITIVE_INFINITY;
              const y = nodeCenterY(targetNode);
              for (const box of boxes) {
                const context = clean(
                  (
                    (box.closest &&
                      box.closest('label, li, tr, ytcp-checkbox, tp-yt-paper-checkbox, div, section') &&
                      box.closest('label, li, tr, ytcp-checkbox, tp-yt-paper-checkbox, div, section').innerText) ||
                    ''
                  )
                );
                if (context && regex.test(context)) {
                  best = box;
                  break;
                }
                const dist = Math.abs(nodeCenterY(box) - y);
                if (dist < bestDist) {
                  best = box;
                  bestDist = dist;
                }
              }
              if (!best) {
                return { matched: true, clicked: false, checked: false };
              }

              if (isChecked(best)) {
                return { matched: true, clicked: false, checked: true };
              }

              let clicked = clickNode(best);
              if (!isChecked(best)) {
                try { targetNode.click(); clicked = true; } catch (e) {}
              }
              if (!isChecked(best)) {
                clicked = clickNode(best) || clicked;
              }
              return { matched: true, clicked, checked: isChecked(best) };
            }
            """,
            pattern,
        )
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"matched": False, "clicked": False, "checked": False}


def ensure_required_declarations_checked(page, rounds=6):
    patterns = [
        r"(mi v[ií]deo no infringe|my video does not infringe)",
        r"(reclamante podr[aá] revisar|claimant.*review|may review my video)",
        r"(impugnaciones fraudulentas|fraudulent disputes|termination of my youtube account|cancelaci[oó]n de mi cuenta)",
    ]
    latest = {"total": 0, "unchecked": 0, "unchecked_labels": []}
    for _ in range(max(1, rounds)):
        latest = ensure_all_dialog_checkboxes_checked(page)
        if int(latest.get("total", 0)) >= 3 and int(latest.get("unchecked", 0)) == 0:
            return latest

        for pattern in patterns:
            click_declaration_checkbox_by_pattern(page, pattern)
            time.sleep(0.15)

        try:
            page.evaluate(
                """
                () => {
                  const dialogs = Array.from(
                    document.querySelectorAll('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]')
                  );
                  const dialog = dialogs[0];
                  if (!dialog) return;
                  dialog.scrollBy(0, Math.max(240, Math.floor((dialog.clientHeight || 600) * 0.55)));
                }
                """
            )
        except Exception:
            pass
        time.sleep(0.2)

    return ensure_all_dialog_checkboxes_checked(page)


def signature_present_in_dialog(page, expected_value):
    expected = (expected_value or "").strip().lower()
    if not expected:
        return False
    try:
        return bool(
            page.evaluate(
                """
                (expected) => {
                  const clean = (v) => String(v || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const isVisible = (node) => {
                    if (!node || !node.getBoundingClientRect) return false;
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) return false;
                    const style = window.getComputedStyle(node);
                    return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                  };
                  const dialogs = Array.from(
                    document.querySelectorAll('tp-yt-paper-dialog#dialog, ytcp-dialog, [role="dialog"]')
                  ).filter((node) => isVisible(node));
                  let dialog = null;
                  for (const node of dialogs) {
                    const text = clean(node.innerText || node.textContent || '');
                    if (!text) continue;
                    if (/enviar comentarios a google|send feedback/i.test(text)) continue;
                    if (/impugnar reclamaci[oó]n|firma \\(obligatorio\\)|argumentos|licencia|license/i.test(text)) {
                      dialog = node;
                      break;
                    }
                  }
                  if (!dialog) {
                    dialog = dialogs[0] || document;
                  }
                  const hintTokens = ['firma', 'signature', 'nombre completo', 'full name', 'full legal name'];
                  const nodes = Array.from(
                    dialog.querySelectorAll('input, textarea, [contenteditable="true"], [role="textbox"]')
                  );
                  for (const node of nodes) {
                    const hints = clean(
                      [
                        node.getAttribute('aria-label'),
                        node.getAttribute('placeholder'),
                        node.getAttribute('name'),
                        node.id ? ((document.querySelector(`label[for="${node.id}"]`) || {}).textContent || '') : '',
                        (node.closest('ytcp-form-input-container, ytcp-form-textarea, div, section') || {}).innerText || '',
                      ].join(' ')
                    );
                    if (!hintTokens.some((token) => hints.includes(token))) continue;
                    const value = clean(node.value || node.textContent || node.innerText || '');
                    if (value.includes(expected)) return true;
                  }
                  return false;
                }
                """,
                expected,
            )
        )
    except Exception:
        return False


def impugnar_una_reclamacion(page, delay_s, espera_envio_s, selectors, modo_modal=False, skip_open_step=False, skip_dispute_step=False, stop_before_send=False):
    root = page
    selectors = selectors or {}
    if skip_dispute_step:
        # El usuario ya paso Tomar medidas + Impugnar + 2FA manualmente.
        # Evitar clear_ui_blockers agresivo que podria cerrar el formulario activo.
        print("Buscando la accion de impugnacion (modo skip_dispute_step: evito limpieza agresiva).")
    else:
        clear_ui_blockers(page, delay_s=min(0.3, delay_s), rounds=3)
        print("Buscando la accion de impugnacion...")

    if not skip_open_step:
        print("Paso 1/7: abrir menu Take action / Impugnar.")
        opened = open_take_action_menu(page, selectors, delay_s, modo_modal=modo_modal)
        if opened is None:
            print("No se encontraron mas reclamaciones en el modal.")
            return None
        if not opened:
            if modo_modal:
                print("Hay reclamaciones pendientes pero no se pudo abrir 'Tomar medidas'.")
            return False
    else:
        print("Paso 1/7: accion 'Tomar medidas' ya abierta.")

    time.sleep(delay_s)
    root = get_active_root(page)

    if skip_dispute_step:
        print("Paso 2/7: saltado (formulario de disputa ya abierto por interaccion manual).")
    else:
        print("Paso 2/7: seleccionar Dispute / Impugnar.")
        clear_ui_blockers(page, delay_s=min(0.25, delay_s), rounds=2)
        if not click_dispute_step(page, selectors):
            try:
                snippet = get_active_root(page).inner_text().strip().replace("\n", " ")
                print(f"Contexto Paso 2: {snippet[:500]}")
                if P_VERIFY_IDENTITY.search(snippet):
                    wait_s = _VERIFY_IDENTITY_WAIT_S
                    if wait_s > 0:
                        print(
                            f"DETECTADO 2FA: 'Verifica que eres tu'. Esperando hasta {wait_s:.0f}s "
                            "para que lo completes en la VENTANA DEL NAVEGADOR (sigue las pantallas, "
                            "ingresa contraseña/codigo). Cuando termines, el script continua solo."
                        )
                        deadline = time.time() + wait_s
                        last_log = 0.0
                        while time.time() < deadline:
                            time.sleep(2.0)
                            try:
                                current = get_active_root(page).inner_text().strip().replace("\n", " ")
                            except Exception:
                                current = ""
                            if not current or not P_VERIFY_IDENTITY.search(current):
                                print("2FA superado. Continuando flujo de impugnacion...")
                                time.sleep(1.5)
                                if click_dispute_step(page, selectors):
                                    return True
                                return "verify_identity_passed_but_dispute_failed"
                            if time.time() - last_log > 15:
                                remaining = int(deadline - time.time())
                                print(f"  ...esperando 2FA (faltan ~{remaining}s)")
                                last_log = time.time()
                        print(f"Tiempo agotado esperando 2FA ({wait_s:.0f}s).")
                    else:
                        print(
                            "DETECTADO: YouTube esta pidiendo verificacion de identidad (2FA). "
                            "Para superarlo, corre SIN --headless y con --esperar-verificacion-s 300."
                        )
                    try:
                        evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                        dump_dispute_evidence(page, {"video_id": "verify_identity"}, "verify_identity_required", str(evid_dir))
                    except Exception:
                        pass
                    return "verify_identity_required"
            except Exception:
                pass
            print("No se encontro el boton de Dispute/Impugnar.")
            try:
                evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                dump_dispute_evidence(page, {"video_id": "step2_fail"}, "no_impugnar_btn", str(evid_dir))
            except Exception:
                pass
            return False
    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 3/7: avanzar a las opciones de Reason/Details.")
    if not skip_dispute_step:
        clear_ui_blockers(page, delay_s=min(0.25, delay_s), rounds=1)
    if handle_options_step(page, selectors, delay_s):
        root = get_active_root(page)
    if not advance_to_reason_step(page, selectors, delay_s):
        try:
            snippet = get_active_root(page).inner_text().strip().replace("\n", " ")
            print(f"Contexto Paso 3: {snippet[:500]}")
        except Exception:
            pass
        print("No se encontraron las opciones de licencia.")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    if handle_options_step(page, selectors, delay_s):
        root = get_active_root(page)

    print("Paso 4/7: marcar 'I have permission / license'.")
    if not select_license_option(page, selectors, delay_s):
        print("No se encontro el elemento: Licencia")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    if handle_options_step(page, selectors, delay_s):
        root = get_active_root(page)
        print("Paso 4/7 (retry post-Opciones): marcar 'I have permission / license'.")
        if not select_license_option(page, selectors, delay_s):
            print("No se encontro el elemento: Licencia tras Opciones.")
            return False
        time.sleep(delay_s)
        root = get_active_root(page)

    print("Paso 5/7: continuar al aviso de lectura.")
    if not skip_dispute_step:
        clear_ui_blockers(page, delay_s=min(0.25, delay_s), rounds=1)
    if not (press_continue_step(page) or click_continue_step(page, selectors, delay_s)):
        print("No se pudo avanzar al aviso de lectura.")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    if handle_options_step(page, selectors, delay_s):
        root = get_active_root(page)
        print("Paso 5/7 (retry post-Opciones): continuar al aviso de lectura.")
        if not (press_continue_step(page) or click_continue_step(page, selectors, delay_s)):
            print("No se pudo avanzar al aviso de lectura tras Opciones.")
            return False
        time.sleep(delay_s)
        root = get_active_root(page)

    print("Paso 6/7: marcar la casilla de confirmacion y continuar.")
    if not skip_dispute_step:
        clear_ui_blockers(page, delay_s=min(0.25, delay_s), rounds=1)
    selector_lectura = selectors.get("confirmar_lectura") or selectors.get("aceptar_terminos")
    if not (click_permission_checkbox(page) or click_read_checkbox(root, page, selector_lectura)):
        if check_all_visible_checkboxes(root) == 0:
            print("No se pudo marcar la casilla de confirmacion.")
    time.sleep(delay_s)
    if not (press_continue_step(page) or click_continue_step(page, selectors, delay_s)):
        print("No se pudo avanzar al formulario de detalles.")
        return False
    root = get_active_root(page)

    click_with_selector(
        root,
        selectors.get("aceptar_terminos"),
        [P_ACEPTAR_TERMINOS],
        roles=("checkbox", "button", "radio"),
        optional=True,
        descripcion="Aceptar terminos",
    )
    time.sleep(delay_s)
    click_repeatedly(root, [P_CONTINUAR], max_clicks=1, selector_info=selectors.get("continuar"))

    root = get_active_root(page)
    print("Paso 7/7: escribir detalles y firma.")
    if not skip_dispute_step:
        clear_ui_blockers(page, delay_s=min(0.25, delay_s), rounds=1)
    if not try_fill_info_licencia(page, root, selectors, mensaje, delay_s):
        print("No se encontro el campo de informacion de licencia.")
        return False

    time.sleep(delay_s)
    if not stop_before_send:
        check_all_visible_checkboxes(root, page=page)
        checkbox_state = ensure_required_declarations_checked(page)
        unchecked = int(checkbox_state.get("unchecked", 0))
        total_boxes = int(checkbox_state.get("total", 0))
        unchecked_labels = checkbox_state.get("unchecked_labels") or []
        if total_boxes <= 0:
            print("No se detectaron casillas en el dialogo de impugnacion.")
            return False
        if total_boxes > 0 and unchecked > 0:
            print(f"Aun hay casillas sin marcar antes de enviar: {unchecked}/{total_boxes}")
            if unchecked_labels:
                print("Casillas pendientes detectadas:")
                for label in unchecked_labels:
                    print(f" - {label[:180]}")
            return False
    else:
        print("Paso 7/7 (stop_before_send): saltando auto-marcado de checkboxes — el user los clickea.")

    if not fill_signature_field(page, root, selectors, firma):
        print("No se encontro el campo de firma.")
        return False
    if not signature_present_in_dialog(page, firma):
        try:
            fill_signature_via_js(page, firma)
        except Exception:
            pass
        if not signature_present_in_dialog(page, firma):
            print("Advertencia: no se pudo confirmar la firma por lectura DOM; se continua.")

    time.sleep(delay_s)
    if stop_before_send:
        print("=" * 70)
        print(">>> PESTAÑA LISTA PARA ENVIAR <<<")
        print("    Formulario completo. Click en 'Enviar' MANUALMENTE en esta pestaña.")
        print("=" * 70)
        try:
            page.evaluate("() => { document.title = '[LISTA-ENVIAR] ' + document.title; }")
        except Exception:
            pass
        return "ready_to_send"
    if not finalize_dispute_submission(page, selectors, delay_s):
        print("No se pudo completar el envio final de la impugnacion.")
        return False
    time.sleep(espera_envio_s)
    root = get_active_root(page)
    click_with_selector(
        root,
        selectors.get("cerrar"),
        [P_CERRAR],
        roles=("button", "link"),
        optional=True,
        descripcion="Cerrar",
    )

    return True


_2FA_PASSED_THIS_SESSION = False

_PAUSE_REQUESTED = False
_INPUT_LISTENER_STARTED = False


_AUTO_ENVIAR_ACTIVO = False


def _input_listener_loop():
    global _PAUSE_REQUESTED
    while True:
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            return
        _PAUSE_REQUESTED = not _PAUSE_REQUESTED
        if _PAUSE_REQUESTED:
            print("\n" + "*" * 70)
            print("*** PAUSA SOLICITADA: el script se detendra antes del proximo reclamo. ***")
            if _AUTO_ENVIAR_ACTIVO:
                print("*** Auto-envio activo: el batch actual se enviara solo. ENTER reanuda. ***")
            else:
                print("*** Andá enviando las tabs ya armadas. ENTER cuando quieras reanudar.  ***")
            print("*" * 70)
        else:
            print("\n" + "*" * 70)
            print("*** REANUDANDO: el script sigue armando tabs. ENTER para pausar otra vez. ***")
            print("*" * 70)


def _start_input_listener():
    global _INPUT_LISTENER_STARTED
    if _INPUT_LISTENER_STARTED:
        return
    try:
        if not sys.stdin.isatty():
            return
    except Exception:
        return
    import threading
    threading.Thread(target=_input_listener_loop, daemon=True).start()
    _INPUT_LISTENER_STARTED = True
    print(">>> Tip: presioná ENTER en cualquier momento para PAUSAR (otra vez ENTER para reanudar). <<<")


def _wait_if_paused():
    if not _PAUSE_REQUESTED:
        return
    while _PAUSE_REQUESTED:
        time.sleep(0.5)
    print(">>> Pausa terminada. Continuando. <<<")


_MEM_PAUSA_MB = 0.0
_MEM_REANUDAR_MB = 0.0


def set_memoria_thresholds(min_mb, recover_mb):
    global _MEM_PAUSA_MB, _MEM_REANUDAR_MB
    _MEM_PAUSA_MB = float(min_mb or 0.0)
    rec = float(recover_mb or 0.0)
    if rec <= _MEM_PAUSA_MB:
        rec = _MEM_PAUSA_MB * 1.5
    _MEM_REANUDAR_MB = rec


def _wait_if_low_memory():
    if _MEM_PAUSA_MB <= 0:
        return
    try:
        import psutil
    except ImportError:
        return
    avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    if avail_mb >= _MEM_PAUSA_MB:
        return
    print("\n" + "!" * 70)
    print(f"!!! MEMORIA BAJA: {avail_mb:.0f} MB libres (umbral pausa: {_MEM_PAUSA_MB:.0f} MB).")
    print(f"!!! PAUSA AUTOMATICA. Reanudacion al recuperar {_MEM_REANUDAR_MB:.0f} MB libres.")
    print("!!! Andá enviando tabs (cerralas con Ctrl+W) para liberar RAM.")
    print("!" * 70)
    last_log = 0.0
    while True:
        time.sleep(5.0)
        try:
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return
        now = time.time()
        if avail_mb >= _MEM_REANUDAR_MB:
            print(f">>> MEMORIA OK: {avail_mb:.0f} MB libres. Reanudando. <<<")
            return
        if now - last_log > 30:
            print(f"  ...esperando memoria (actual: {avail_mb:.0f} MB / objetivo: {_MEM_REANUDAR_MB:.0f} MB)")
            last_log = now


def _click_nth_take_action_in_page(page, claim_index):
    """Devuelve {'ok': bool, 'total': int}. Click el i-th boton 'Tomar medidas' visible Y HABILITADO."""
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


def _click_impugnar_in_modal(page):
    """Intenta marcar 'Impugnar' en el modal de Seleccionar Acción.
    Primero ubica el elemento con JS, devuelve sus coords, y hace click via
    page.mouse.click(x,y) trusted. Si no encuentra coords, cae al click JS.
    Importante: el modal de Content ID tiene cards con descripción + link
    'Más información' interno. Si el click cae sobre ese link, NO selecciona
    la card. Por eso clickeamos cerca del HEADER (top) de la card."""
    info = None
    try:
        info = page.evaluate(
            """
            () => {
              const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
              let target = null;
              for (const d of dialogs) {
                const t = (d.innerText || '').toLowerCase();
                if (/seleccionar acci|select action|elige una acci|choose an action/.test(t)) {
                  target = d; break;
                }
              }
              const scope = target || document;
              const candidates = Array.from(scope.querySelectorAll(
                'button, [role="button"], ytcp-button-shape, tp-yt-paper-button, [role="radio"], [role="option"]'
              ));
              let best = null;
              let bestScore = -1;
              for (const el of candidates) {
                const raw = (el.innerText || el.textContent || '').trim();
                if (!raw) continue;
                const firstLine = raw.split(/\\n|\\r/)[0].trim().toLowerCase();
                if (!/^(impugnar|disputar|dispute)$/.test(firstLine)) continue;
                if (/apelar|appeal/.test(firstLine)) continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) continue;
                let score = 100;
                if ((el.tagName || '').toLowerCase() === 'button') score += 10;
                if (r.height > 60) score += 20;
                if (score > bestScore) {
                  bestScore = score;
                  const xCenter = r.x + r.width / 2;
                  const yHeader = r.y + Math.min(30, Math.max(18, r.height * 0.18));
                  best = { x: xCenter, y: yHeader, text: firstLine, tag: (el.tagName || '').toLowerCase(), h: r.height };
                  try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                }
              }
              return best;
            }
            """
        )
    except Exception:
        info = None

    if not info:
        return False

    clicked_mouse = False
    try:
        cx = float(info.get("x", 0))
        cy = float(info.get("y", 0))
        if cx > 0 and cy > 0:
            page.mouse.move(cx, cy)
            time.sleep(0.08)
            page.mouse.click(cx, cy, delay=70)
            clicked_mouse = True
    except Exception:
        clicked_mouse = False

    try:
        focused_via_enter = page.evaluate(
            """
            () => {
              const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
              let target = null;
              for (const d of dialogs) {
                const t = (d.innerText || '').toLowerCase();
                if (/seleccionar acci|select action|elige una acci|choose an action/.test(t)) {
                  target = d; break;
                }
              }
              const scope = target || document;
              const candidates = Array.from(scope.querySelectorAll('button'));
              for (const el of candidates) {
                const raw = (el.innerText || el.textContent || '').trim();
                if (!raw) continue;
                const firstLine = raw.split(/\\n|\\r/)[0].trim().toLowerCase();
                if (!/^(impugnar|disputar|dispute)$/.test(firstLine)) continue;
                if (/apelar|appeal/.test(firstLine)) continue;
                try { el.focus(); } catch (e) {}
                return true;
              }
              return false;
            }
            """
        )
        if focused_via_enter:
            time.sleep(0.1)
            page.keyboard.press("Enter")
            return True
    except Exception:
        pass

    if clicked_mouse:
        return True

    try:
        return bool(
            page.evaluate(
                """
                () => {
                  const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
                  let target = null;
                  for (const d of dialogs) {
                    const t = (d.innerText || '').toLowerCase();
                    if (/seleccionar acci|select action|elige una acci|choose an action/.test(t)) {
                      target = d; break;
                    }
                  }
                  const scope = target || document;
                  const candidates = Array.from(scope.querySelectorAll(
                    'button, [role="button"], ytcp-button-shape, tp-yt-paper-button, [role="radio"], [role="option"]'
                  ));
                  for (const el of candidates) {
                    const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (!/(^|\\b)(impugnar|disputar|dispute)(\\b|\\s|$)/.test(t)) continue;
                    if (/apelar|appeal/.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width <= 0) continue;
                    el.click();
                    return true;
                  }
                  return false;
                }
                """
            )
        )
    except Exception:
        return False


def _click_continuar_in_modal(page):
    info = None
    try:
        info = page.evaluate(
            """
            () => {
              const buttons = Array.from(document.querySelectorAll('button'));
              let best = null;
              let bestScore = -1;
              for (const el of buttons) {
                const t = (el.innerText || '').trim().toLowerCase();
                if (!/^continuar$|^continue$/.test(t)) continue;
                if ((el.tagName || '').toLowerCase() !== 'button') continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 4 || r.height <= 4) continue;
                const dis = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
                if (dis) continue;
                let score = 100;
                if (score > bestScore) {
                  bestScore = score;
                  best = { x: r.x + r.width / 2, y: r.y + r.height / 2, tag: 'button' };
                  try { el.scrollIntoView({block: 'center'}); } catch (e) {}
                }
              }
              return best;
            }
            """
        )
    except Exception:
        info = None

    if info:
        try:
            cx = float(info.get("x", 0))
            cy = float(info.get("y", 0))
            if cx > 0 and cy > 0:
                page.mouse.move(cx, cy)
                time.sleep(0.08)
                page.mouse.click(cx, cy, delay=70)
                return True
        except Exception:
            pass

    try:
        return bool(
            page.evaluate(
                """
                () => {
                  const all = Array.from(document.querySelectorAll(
                    'button, [role="button"], ytcp-button-shape'
                  ));
                  for (const el of all) {
                    const t = (el.innerText || '').trim().toLowerCase();
                    if (!/^continuar$|^continue$/.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width <= 0) continue;
                    const dis = el.disabled || el.getAttribute('aria-disabled') === 'true';
                    if (dis) continue;
                    el.click();
                    return true;
                  }
                  return false;
                }
                """
            )
        )
    except Exception:
        return False


def _wait_for_form_or_2fa(page, max_s=300.0):
    """Devuelve 'form', '2fa' o 'timeout'."""
    strict = [
        "descripción general", "descripcion general",
        "firma (obligatorio)", "firma obligatorio",
        "selecciona el motivo principal",
        "select the primary reason",
        "i have permission",
        "tengo permiso",
        "argumentos",
    ]
    deadline = time.time() + max_s
    while time.time() < deadline:
        time.sleep(1.5)
        try:
            txt = get_active_root(page).inner_text().strip().lower()
        except Exception:
            txt = ""
        if P_VERIFY_IDENTITY.search(txt):
            return "2fa"
        in_selector = bool(P_ACTION_SELECTOR_DIALOG.search(txt))
        hits = sum(1 for h in strict if h in txt)
        if hits >= 1 and not in_selector:
            return "form"
    return "timeout"


def _contar_checkboxes_via_locator(root_):
    """Cuenta total y unchecked via Playwright locator (pierces shadow DOM, usa el dialog correcto)."""
    tot = 0
    unc = 0
    try:
        boxes = root_.locator("input[type='checkbox'], tp-yt-paper-checkbox, ytcp-checkbox, [role='checkbox']")
        n = boxes.count()
    except Exception:
        return 0, 0
    for i in range(n):
        box = boxes.nth(i)
        try:
            if not box.is_visible():
                continue
        except Exception:
            continue
        tot += 1
        is_chk = False
        try:
            is_chk = box.is_checked()
        except Exception:
            aria = (box.get_attribute("aria-checked") or "").lower()
            is_chk = aria == "true" or (box.get_attribute("checked") is not None)
        if not is_chk:
            unc += 1
    return tot, unc


def _checkboxes_realmente_marcados(page):
    """Devuelve True si YouTube YA NO muestra el aviso 'Marca todas las casillas anteriores...'.
    Ese mensaje rojo aparece cuando los checkboxes no están marcados a ojos del server,
    incluso si aria-checked dice True (porque dispatchEvent no cuenta como trusted click)."""
    try:
        root = get_active_root(page)
        text = (root.inner_text() or "").strip().lower()
    except Exception:
        return False
    avisos = (
        "marca todas las casillas anteriores",
        "marca las casillas anteriores",
        "check all boxes above to submit",
        "check the boxes above to submit",
        "select all checkboxes above",
    )
    return not any(aviso in text for aviso in avisos)


def _force_trusted_checkboxes(page, max_passes=3, ignore_aria=True):
    """Marca checkboxes del dialog activo con page.mouse.click(x,y) en sus coordenadas
    REALES — el método más confiable para custom elements Polymer porque emula mouse físico
    via CDP. Si ignore_aria=True, NO confía en aria-checked (que puede ser mentira si
    alguien lo seteó por dispatchEvent); en su lugar, sólo confía en input[type=checkbox]:checked
    de un <input> interno o usa la primera pasada para clickear TODOS los checkboxes visibles
    sin verificar state previo."""
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


def _enviar_una_tab(page, args, selectors, evid_dir, idx, total):
    """Marca los 3 checks con Playwright trusted clicks y clickea 'Enviar' en UNA tab armada.

    Devuelve dict: {'ok': bool, 'razon': str, 'titulo': str}.
    Si OK, cierra la tab. Si falla, deja la tab abierta para revision manual.
    """
    delay_s = args.delay
    espera_envio_s = args.espera_envio
    titulo_actual = ""
    try:
        try:
            page.bring_to_front()
        except Exception:
            pass
        try:
            titulo_actual = page.title() or ""
        except Exception:
            pass
        print(f"\n  [auto-envio {idx}/{total}] {titulo_actual[:140]}")

        time.sleep(min(delay_s, 0.6))
        root = get_active_root(page)
        if not is_dispute_form_active(page):
            print("     [SKIP] form no activo.")
            return {"ok": False, "razon": "form_no_activo", "titulo": titulo_actual}

        clicks = _force_trusted_checkboxes(page, max_passes=2, ignore_aria=True)
        time.sleep(0.5)
        total_boxes, unchecked = _contar_checkboxes_via_locator(get_active_root(page))
        server_ok = _checkboxes_realmente_marcados(page)
        print(f"     [verif inicial] trusted_clicks={clicks} total={total_boxes} unchecked={unchecked} server_ok={server_ok}")
        for retry in range(5):
            if total_boxes > 0 and unchecked == 0 and server_ok:
                break
            clicks2 = _force_trusted_checkboxes(page, max_passes=2, ignore_aria=False)
            time.sleep(0.5)
            total_boxes, unchecked = _contar_checkboxes_via_locator(get_active_root(page))
            server_ok = _checkboxes_realmente_marcados(page)
            print(f"     [verif retry {retry+1}] trusted_clicks={clicks2} total={total_boxes} unchecked={unchecked} server_ok={server_ok}")
            time.sleep(0.3)

        if total_boxes <= 0:
            print("     [FALLO] no detecte casillas en el dialogo.")
            return {"ok": False, "razon": "sin_casillas", "titulo": titulo_actual}
        if unchecked > 0 or not server_ok:
            razon_detalle = f"unchecked_{unchecked}_de_{total_boxes}"
            if unchecked == 0 and not server_ok:
                razon_detalle = "server_no_registra_clicks"
                print("     [FALLO] aria-checked dice OK pero YouTube muestra aviso 'Marca todas las casillas anteriores'.")
            else:
                print(f"     [FALLO] {unchecked}/{total_boxes} casillas SIN MARCAR tras retries trusted.")
            if evid_dir is not None:
                try:
                    dump_dispute_evidence(page, {"video_id": titulo_actual[:80]}, "auto_envio_unchecked", str(evid_dir))
                except Exception:
                    pass
            return {"ok": False, "razon": razon_detalle, "titulo": titulo_actual}
        print(f"     {total_boxes}/{total_boxes} casillas marcadas (mouse coords) + server_ok confirmado.")

        time.sleep(min(delay_s, 0.5))
        root = get_active_root(page)
        try:
            botones = root.locator("button, [role='button'], tp-yt-paper-button, ytcp-button-shape")
            n_btn = botones.count()
        except Exception:
            n_btn = 0
        labels_visibles = []
        enviar_button_native = None
        enviar_wrapper = None
        for i in range(n_btn):
            b = botones.nth(i)
            try:
                if not b.is_visible():
                    continue
            except Exception:
                continue
            try:
                enabled = b.is_enabled()
            except Exception:
                enabled = True
            try:
                label = (b.inner_text().strip()
                         or (b.get_attribute("aria-label") or "").strip()
                         or (b.get_attribute("title") or "").strip())
            except Exception:
                label = ""
            if not label:
                continue
            try:
                tag = (b.evaluate("el => (el.tagName || '').toLowerCase()") or "").strip()
            except Exception:
                tag = ""
            labels_visibles.append(f"{'OK' if enabled else 'OFF'}:{tag}:{label[:40]}")
            if enabled and P_PRIMARY_SEND.search(label):
                if tag == "button" and enviar_button_native is None:
                    enviar_button_native = b
                elif enviar_wrapper is None:
                    enviar_wrapper = b
        print(f"     [debug] botones visibles: {labels_visibles[:12]}")

        enviar_locator = enviar_button_native or enviar_wrapper
        if enviar_locator is None:
            print("     [FALLO] NO encontre boton 'Enviar' habilitado.")
            if evid_dir is not None:
                try:
                    dump_dispute_evidence(page, {"video_id": titulo_actual[:80]}, "auto_envio_sin_boton_enviar", str(evid_dir))
                except Exception:
                    pass
            return {"ok": False, "razon": "sin_boton_enviar", "titulo": titulo_actual}

        cual = "button_nativo" if enviar_button_native is not None else "wrapper"
        print(f"     -> Click 'Enviar' (trusted via mouse coords, {cual})...")
        try:
            enviar_locator.scroll_into_view_if_needed()
            time.sleep(0.15)
            clicked_via_mouse = False
            try:
                bbox = enviar_locator.bounding_box()
                if bbox and bbox.get("width", 0) > 4 and bbox.get("height", 0) > 4:
                    cx = bbox["x"] + bbox["width"] / 2
                    cy = bbox["y"] + bbox["height"] / 2
                    page.mouse.move(cx, cy)
                    time.sleep(0.1)
                    page.mouse.click(cx, cy, delay=80)
                    clicked_via_mouse = True
                    print("     -> click 'Enviar' OK (mouse coords).")
            except Exception as exc:
                print(f"     -> mouse coords fallo: {exc}")
            if not clicked_via_mouse:
                enviar_locator.click(timeout=3000)
                print("     -> click 'Enviar' OK (locator fallback).")
        except Exception as exc:
            print(f"     -> click 'Enviar' fallo: {exc}")
            if evid_dir is not None:
                try:
                    dump_dispute_evidence(page, {"video_id": titulo_actual[:80]}, "auto_envio_no_envio", str(evid_dir))
                except Exception:
                    pass
            return {"ok": False, "razon": "no_envio", "titulo": titulo_actual}

        print(f"     -> Esperando {espera_envio_s}s...")
        time.sleep(espera_envio_s)
        sigue_activo = is_dispute_form_active(page)
        if sigue_activo:
            print("     [FALLO] form sigue activo despues del Enviar (probable bot detection).")
            if evid_dir is not None:
                try:
                    dump_dispute_evidence(page, {"video_id": titulo_actual[:80]}, "auto_envio_form_persistente", str(evid_dir))
                except Exception:
                    pass
            return {"ok": False, "razon": "form_persistente", "titulo": titulo_actual}

        try:
            page.evaluate(
                """() => { document.title = document.title.replace('[LISTA-ENVIAR] ', '[ENVIADA] '); }"""
            )
        except Exception:
            pass
        print("     [OK] impugnacion enviada. Cerrando tab.")
        try:
            page.close()
        except Exception:
            pass
        return {"ok": True, "razon": "ok", "titulo": titulo_actual}
    except Exception as exc:
        print(f"     [EXC] {exc}")
        return {"ok": False, "razon": f"exc:{exc}", "titulo": titulo_actual}


def enviar_pestanas_listas(context, args, selectors, max_a_enviar=None):
    """Recorre tabs con titulo '[LISTA-ENVIAR]' y las envia (modo BATCH).

    Devuelve dict: {'enviadas': N, 'fallos': [...], 'total_listas': N}.
    Las tabs que fallan quedan abiertas; las exitosas se cierran.
    """
    candidatas = []
    for page in context.pages:
        try:
            t = page.title() or ""
        except Exception:
            continue
        if "[LISTA-ENVIAR]" in t:
            candidatas.append(page)

    if not candidatas:
        return {"enviadas": 0, "fallos": [], "total_listas": 0}

    if isinstance(max_a_enviar, int) and max_a_enviar > 0:
        objetivo = candidatas[:max_a_enviar]
    else:
        objetivo = candidatas

    print("\n" + "=" * 70)
    print(f">>> AUTO-ENVIO BATCH: voy a enviar {len(objetivo)} tab(s) (de {len(candidatas)} listas).")
    print("=" * 70)
    enviadas = 0
    fallos = []
    evid_dir = Path(getattr(args, "evidencia_dir", "")) if getattr(args, "evidencia_dir", None) else None
    espera_entre_s = max(0.5, min(getattr(args, "espera_entre", 1.0), 3.0))

    for idx, page in enumerate(objetivo, start=1):
        result = _enviar_una_tab(page, args, selectors, evid_dir, idx, len(objetivo))
        if result.get("ok"):
            enviadas += 1
        else:
            fallos.append({"titulo": result.get("titulo", ""), "razon": result.get("razon", "?")})
        time.sleep(espera_entre_s)

    print("\n" + "=" * 70)
    print(f">>> AUTO-ENVIO TERMINADO. Enviadas: {enviadas}/{len(objetivo)} | Fallos: {len(fallos)}")
    if fallos:
        print(">>> Tabs con fallo (revisar y enviar a mano):")
        for f in fallos:
            print(f"   - [{f.get('razon')}] {f.get('titulo','')[:140]}")
    print("=" * 70)
    return {"enviadas": enviadas, "fallos": fallos, "total_listas": len(candidatas)}


def _scroll_copyright_page_to_bottom(page, max_iters=12, pause=0.35):
    """Scrollea la página /copyright hasta el fondo para forzar lazy render de todos los reclamos."""
    last_count = -1
    for _ in range(max_iters):
        try:
            page.evaluate(
                """
                () => {
                  const scrollables = Array.from(document.querySelectorAll('*')).filter(el => {
                    const s = window.getComputedStyle(el);
                    return (s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 10;
                  });
                  for (const el of scrollables) { el.scrollTop = el.scrollHeight; }
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


def preparar_pestana_para_reclamo(context, video_id, claim_index, args, selectors):
    """Abre nueva tab. Hace todo el flujo hasta dejar la pestaña en el botón Enviar (sin clickearlo)."""
    global _2FA_PASSED_THIS_SESSION
    page = context.new_page()
    try:
        url = f"https://studio.youtube.com/video/{video_id}/copyright"
        print(f"  -> Abriendo tab para video {video_id} reclamo #{claim_index} ({url})")
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass
        _scroll_copyright_page_to_bottom(page)

        deadline = time.time() + 20.0
        result = {"ok": False, "total": 0}
        while time.time() < deadline:
            time.sleep(1.0)
            try:
                result = _click_nth_take_action_in_page(page, claim_index)
            except Exception:
                result = {"ok": False, "total": 0}
            if result.get("ok"):
                break
            if result.get("total", 0) > 0 and claim_index >= result.get("total", 0):
                # Hay CTAs pero no tantos como pedimos → no hay mas reclamos.
                break
        if not result.get("ok"):
            return {"status": "no_more_claims", "total": result.get("total", 0), "page": page}
        print(f"     click Tomar medidas #{claim_index} OK (total CTAs visibles: {result.get('total')})")

        deadline = time.time() + 12.0
        modal_open = False
        while time.time() < deadline:
            try:
                txt = get_active_root(page).inner_text().lower()
                if "seleccionar acci" in txt or "select action" in txt or "elige una acci" in txt:
                    modal_open = True
                    break
            except Exception:
                pass
            time.sleep(0.3)
        if not modal_open:
            try:
                evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                dump_dispute_evidence(page, {"video_id": f"{video_id}_claim{claim_index}"}, "modal_no_abre", str(evid_dir))
            except Exception:
                pass
            return {"status": "modal_seleccionar_no_abre", "page": page}

        click_ok = False
        for intento in range(3):
            if _click_impugnar_in_modal(page):
                click_ok = True
                break
            time.sleep(0.6)
            try:
                txt2 = get_active_root(page).inner_text().lower()
                if "seleccionar acci" not in txt2 and "select action" not in txt2 and "elige una acci" not in txt2:
                    print(f"     [Impugnar modal] modal de acciones cerrado en intento {intento+1}, abortando retries.")
                    break
            except Exception:
                pass
            print(f"     [Impugnar modal] retry {intento+1}/3 buscando boton 'Impugnar' en modal...")
        if not click_ok:
            try:
                evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                dump_dispute_evidence(page, {"video_id": f"{video_id}_claim{claim_index}"}, "no_pude_click_impugnar", str(evid_dir))
            except Exception:
                pass
            return {"status": "no_pude_click_impugnar", "page": page}

        check_continuar_js = """
            () => {
              const all = Array.from(document.querySelectorAll('button'));
              for (const el of all) {
                const t = (el.innerText || '').trim().toLowerCase();
                if (!/^continuar$|^continue$/.test(t)) continue;
                if ((el.tagName || '').toLowerCase() !== 'button') continue;
                const r = el.getBoundingClientRect();
                if (r.width <= 0) continue;
                const dis = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
                if (!dis) return true;
              }
              return false;
            }
            """
        time.sleep(0.5)
        continuar_enabled = False
        deadline_cont = time.time() + 5.0
        while time.time() < deadline_cont:
            try:
                continuar_enabled = bool(page.evaluate(check_continuar_js))
            except Exception:
                continuar_enabled = False
            if continuar_enabled:
                break
            time.sleep(0.3)

        if not continuar_enabled:
            print("     [Impugnar modal] Continuar sigue DISABLED tras click. Re-intentando click en card Impugnar...")
            for retry_card in range(3):
                _click_impugnar_in_modal(page)
                time.sleep(0.7)
                try:
                    continuar_enabled = bool(page.evaluate(check_continuar_js))
                except Exception:
                    continuar_enabled = False
                if continuar_enabled:
                    print(f"     [Impugnar modal] Continuar habilitado tras retry {retry_card+1}.")
                    break
            if not continuar_enabled:
                print("     [Impugnar modal] Continuar SIGUE disabled tras retries. El click no esta seleccionando la card Impugnar.")

        if not _click_continuar_in_modal(page):
            print("     [Impugnar modal] no se pudo clickear Continuar.")
        time.sleep(1.5)

        estado = _wait_for_form_or_2fa(page, max_s=20.0)
        if estado == "2fa":
            print("     >>> 2FA detectado en esta tab. Pasalo manualmente. <<<")
            _2FA_PASSED_THIS_SESSION = True
            tope = float(getattr(args, "pausa_interactiva_s", 600.0))
            estado = _wait_for_form_or_2fa(page, max_s=tope)
            if estado != "form":
                try:
                    snippet = get_active_root(page).inner_text().strip().replace("\n", " | ")
                    print(f"     [no_se_detecto_form post-2fa] estado={estado}. preview: {snippet[:300]!r}")
                except Exception:
                    pass
                try:
                    evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                    dump_dispute_evidence(page, {"video_id": f"{video_id}_claim{claim_index}"}, f"no_se_detecto_form_post2fa_{estado}", str(evid_dir))
                except Exception:
                    pass
                return {"status": f"2fa_no_completado ({estado})", "page": page}
        if estado != "form":
            try:
                snippet = get_active_root(page).inner_text().strip().replace("\n", " | ")
                print(f"     [no_se_detecto_form] estado={estado}. preview: {snippet[:300]!r}")
            except Exception:
                snippet = ""
            try:
                page_url = page.url
                print(f"     [no_se_detecto_form] URL actual: {page_url[:200]}")
            except Exception:
                pass
            try:
                evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
                dump_dispute_evidence(page, {"video_id": f"{video_id}_claim{claim_index}"}, f"no_se_detecto_form_{estado}", str(evid_dir))
            except Exception:
                pass
            return {"status": f"no_se_detecto_form ({estado})", "page": page}

        set_suppress_ui_blockers(True)
        try:
            ok = impugnar_una_reclamacion(
                page,
                args.delay,
                args.espera_envio,
                selectors,
                modo_modal=True,
                skip_open_step=True,
                skip_dispute_step=True,
                stop_before_send=True,
            )
        finally:
            set_suppress_ui_blockers(False)
        if ok == "ready_to_send":
            return {"status": "ready_to_send", "page": page}
        try:
            evid_dir = Path(__file__).resolve().parent / "data" / "impugnar_evidencia"
            dump_dispute_evidence(page, {"video_id": f"{video_id}_claim{claim_index}"}, "tab_fallo_form", str(evid_dir))
        except Exception:
            pass
        return {"status": f"impugnar_devolvio_{ok!r}", "page": page}
    except Exception as exc:
        return {"status": f"excepcion: {exc}", "page": page}


def _procesar_lote_pestanas(context, page0, args, selectors):
    """Procesa una pasada: escanea videos pendientes y abre una tab por reclamo."""
    if not ensure_content_list(page0):
        print("No se pudo abrir la tabla de Contenido.")
        return None

    fecha_pat = getattr(args, "_omitir_fecha_pat", None)
    is_valid = (lambda it: not is_video_marcado_omitir(it, fecha_pat)) if fecha_pat else None
    target_valid = args.max if args.max else 30

    items = scan_claim_rows(
        page0,
        delay_s=max(0.8, min(args.delay, 1.5)),
        max_items=args.max_scan or args.max,
        is_valid=is_valid,
        target_valid=target_valid,
    )
    if not items:
        return {"items": 0, "listas": [], "fallos": [], "omitidos": 0}
    print(f"\nVideos detectados: {len(items)}")
    pestanas_listas = []
    fallos = []
    omitidos = 0

    max_tabs = int(getattr(args, "max_tabs", 0) or 0)
    cortar_por_tabs = False

    for vidx, item in enumerate(items, start=1):
        if args.max and vidx > args.max:
            break
        if cortar_por_tabs:
            break
        video_id = (item.get("video_id") or "").strip()
        title = item.get("title") or video_id
        if not video_id:
            continue
        print(f"\n[Video {vidx}/{len(items)}] {title}")

        if is_video_marcado_omitir(item, fecha_pat):
            print(f"     [OMITIDO] fecha marcador detectada ({item.get('date_text') or 'sin fecha visible'}). Skip.")
            omitidos += 1
            continue

        max_reclamos_video = 30
        intentos_sin_progreso = 0
        ci = 0
        while ci < max_reclamos_video:
            _wait_if_paused()
            _wait_if_low_memory()
            res = preparar_pestana_para_reclamo(context, video_id, 0, args, selectors)
            status = res.get("status")
            tab = res.get("page")
            if status == "no_more_claims":
                print(f"     No hay mas reclamos en este video (procesados: {ci}).")
                if tab is not None:
                    try: tab.close()
                    except Exception: pass
                break
            if status == "ready_to_send":
                pestanas_listas.append({"video_id": video_id, "claim": ci, "title": title})
                print(f"     [LISTA] tab #{len(pestanas_listas)} preparada (video {video_id}, reclamo {ci}).")
                intentos_sin_progreso = 0
                if getattr(args, "auto_enviar_listas", False) and tab is not None:
                    evid_dir_inm = Path(getattr(args, "evidencia_dir", "")) if getattr(args, "evidencia_dir", None) else None
                    _enviar_una_tab(tab, args, selectors, evid_dir_inm, len(pestanas_listas), max_tabs if max_tabs > 0 else len(pestanas_listas))
                if max_tabs > 0 and len(pestanas_listas) >= max_tabs:
                    print(f"     >>> Corte por --max-tabs {max_tabs}: ya tengo {len(pestanas_listas)} tab(s) listas. Paro de armar.")
                    cortar_por_tabs = True
                    break
            else:
                fallos.append({"video_id": video_id, "claim": ci, "razon": status})
                print(f"     [FALLO] reclamo {ci}: {status}")
                if tab is not None:
                    try: tab.close()
                    except Exception: pass
                intentos_sin_progreso += 1
                if intentos_sin_progreso >= 4:
                    print(f"     [STOP] {intentos_sin_progreso} fallos consecutivos en este video. Saltando al siguiente.")
                    break
            ci += 1

    return {"items": len(items), "listas": pestanas_listas, "fallos": fallos, "omitidos": omitidos}


def preparar_pestanas_modo(context, page0, args, selectors):
    """Modo principal: abre tabs en paralelo, una por reclamo. Loop infinito si --loop."""
    global _PAUSE_REQUESTED, _AUTO_ENVIAR_ACTIVO
    _AUTO_ENVIAR_ACTIVO = bool(getattr(args, "auto_enviar_listas", False))
    _start_input_listener()
    loop_on = bool(getattr(args, "loop", False))
    max_loops = max(1, int(getattr(args, "max_loops", 50)))

    total_listas = 0
    total_fallos = 0
    total_omitidos = 0

    for vuelta in range(1, max_loops + 1):
        print("\n" + "=" * 70)
        print(f"VUELTA {vuelta}/{max_loops if loop_on else 1}")
        print("=" * 70)
        resultado = _procesar_lote_pestanas(context, page0, args, selectors)
        if resultado is None:
            return False

        listas = resultado["listas"]
        fallos = resultado["fallos"]
        items_count = resultado["items"]
        omitidos = resultado.get("omitidos", 0)
        total_listas += len(listas)
        total_fallos += len(fallos)
        total_omitidos += omitidos

        print("\n" + "-" * 70)
        print(f"Vuelta {vuelta} terminada")
        print(f"  Videos detectados: {items_count}")
        print(f"  Pestañas listas: {len(listas)}")
        print(f"  Omitidos por fecha marcador: {omitidos}")
        print(f"  Fallos: {len(fallos)}")
        print(f"  Acumulado total: {total_listas} listas | {total_omitidos} omitidos | {total_fallos} fallos")

        if items_count == 0:
            print("\nNo quedan videos con reclamos. Terminando loop.")
            break

        if getattr(args, "auto_enviar_listas", False) and len(listas) > 0:
            max_envio = int(getattr(args, "probar_envio", 0) or 0)
            enviar_pestanas_listas(
                context, args, selectors,
                max_a_enviar=(max_envio if max_envio > 0 else None),
            )

        if not loop_on:
            print("\n>>> Andá pestaña por pestaña haciendo click los 3 checks + 'Enviar'. <<<")
            print(">>> Cuando termines, cerrá esta terminal con Ctrl-C. <<<")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\nCerrando.")
            return True

        if vuelta >= max_loops:
            print(f"\nLimite de loops alcanzado ({max_loops}).")
            break

        if getattr(args, "auto_enviar_listas", False):
            print(f"\n>>> Vuelta {vuelta} terminada (auto-envio activo). Re-escaneando para vuelta {vuelta+1}...")
            time.sleep(1.0)
            continue

        print("\n" + ">" * 70)
        print(">>> Andá enviando las tabs ya armadas (3 checks + Enviar en cada una).")
        print(">>> Cuando termines, apretá ENTER para arrancar la SIGUIENTE pasada.")
        print(">>> Ctrl-C para cortar el loop.")
        print(">" * 70)
        _PAUSE_REQUESTED = True
        try:
            while _PAUSE_REQUESTED:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nLoop cortado por usuario.")
            return True
        print(">>> Reanudando: arrancando re-escaneo. <<<")

    print("\n" + "=" * 70)
    print("LOOP FINALIZADO")
    print(f"  Total acumulado: {total_listas} pestañas listas | {total_fallos} fallos")
    print("=" * 70)
    if getattr(args, "auto_enviar_listas", False):
        print(">>> Auto-envio terminado. Saliendo.")
        return True
    print(">>> Cerrá esta terminal con Ctrl-C cuando hayas enviado todo. <<<")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nCerrando.")
    return True


def procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path):
    if not ensure_content_list(page):
        print("No se pudo abrir la tabla de Contenido en YouTube Studio.")
        return False

    limite_escaneo = args.max_scan or args.max
    print("Escaneando videos con restriccion 'Derechos de autor'...")
    items = scan_claim_rows(page, delay_s=max(0.8, min(args.delay, 1.5)), max_items=limite_escaneo)
    save_queue(queue_path, items)

    if not items:
        print("No se detectaron videos con reclamos en la lista actual.")
        return True

    processed = load_checkpoint(checkpoint_path)
    pendientes = [item for item in items if processed.get(claim_item_key(item)) != "ok"]
    print(f"Videos detectados: {len(items)} | Pendientes: {len(pendientes)}")

    if args.solo_detectar:
        print(f"Cola guardada en: {queue_path}")
        return True

    total = 0
    errores_consecutivos = 0
    reset_content_scroll(page)

    for index, item in enumerate(pendientes, start=1):
        if args.max and total >= args.max:
            break

        key = claim_item_key(item)
        titulo = item.get("title") or key
        print(f"\n[{index}/{len(pendientes)}] Abriendo reclamo: {titulo}")

        if getattr(args, "saltar_programados", False) and is_video_programado(item):
            print("Saltando video programado por --saltar-programados (opt-in).")
            item["status"] = "programado_no_publicado"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "programado_no_publicado")
            time.sleep(min(args.espera_entre, 1.0))
            continue

        fecha_pat = getattr(args, "_omitir_fecha_pat", None)
        if is_video_marcado_omitir(item, fecha_pat):
            print(f"Saltando video con fecha marcador ({item.get('date_text') or 'fecha en row_text'}).")
            item["status"] = "omitido_fecha_marcador"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "omitido_fecha_marcador")
            time.sleep(min(args.espera_entre, 1.0))
            continue

        _wait_if_low_memory()

        page.goto(args.url, wait_until="domcontentloaded")
        clear_ui_blockers(page, delay_s=min(0.25, args.delay), rounds=3)
        if not dismiss_claim_dialog(page, delay_s=min(args.delay, 1.0)):
            print("No se pudo cerrar un modal previo. Se intentara limpiar estado recargando Studio.")
            page.goto(args.url, wait_until="domcontentloaded")
            clear_ui_blockers(page, delay_s=min(0.25, args.delay), rounds=3)
        if not ensure_content_list(page):
            print("No se pudo volver a la tabla de Contenido antes de abrir el siguiente video.")
            return False
        reset_content_scroll(page)
        time.sleep(min(args.delay, 1.0))

        if not open_claim_modal_with_recovery(
            page,
            item,
            delay_s=args.delay,
            studio_url=args.url,
            max_attempts=max(1, args.reintentos_modal),
        ):
            print("No se pudo abrir el modal de derechos de autor para este video.")
            dump_dispute_evidence(page, item, "error_modal", getattr(args, "evidencia_dir", None))
            item["status"] = "error_modal"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "error_modal")
            errores_consecutivos += 1
            if errores_consecutivos >= 3:
                print("Se detuvo el proceso por demasiados errores consecutivos al abrir modales.")
                return False
            time.sleep(args.espera_entre)
            continue

        errores_consecutivos = 0
        reclamos_video = 0
        fallo_video = False
        max_reclamos_por_video = 30

        while reclamos_video < max_reclamos_por_video:
            clear_ui_blockers(page, delay_s=min(0.25, args.delay), rounds=2)
            if not claim_dialog_is_open(page):
                if not open_claim_modal_with_recovery(
                    page,
                    item,
                    delay_s=args.delay,
                    studio_url=args.url,
                    max_attempts=max(1, args.reintentos_modal),
                ):
                    print("Se cerro el modal y no se pudo reabrir para continuar este video.")
                    fallo_video = True
                    break

            acciones_actuales = wait_for_take_action_buttons(
                page,
                timeout_s=3.0 if reclamos_video == 0 else 2.0,
            )
            print(f"Acciones 'Tomar medidas' detectadas: {acciones_actuales}")

            if acciones_actuales > 0 and not claim_dialog_is_open(page):
                print("Click directo a 'Tomar medidas' en pagina /copyright para abrir modal Seleccionar accion...")
                clicked_via_js = False
                try:
                    clicked_via_js = page.evaluate(
                        """
                        () => {
                          const all = Array.from(document.querySelectorAll(
                            'button, a, [role=\"button\"], ytcp-button-shape, tp-yt-paper-button'
                          ));
                          for (const el of all) {
                            const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                            if (!/(tomar medidas|take action)/i.test(t)) continue;
                            if (t.length > 80) continue;
                            const r = el.getBoundingClientRect();
                            if (r.width <= 0 || r.height <= 0) continue;
                            const s = window.getComputedStyle(el);
                            if (s.display === 'none' || s.visibility === 'hidden') continue;
                            if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
                            el.scrollIntoView({block: 'center'});
                            el.click();
                            return true;
                          }
                          return false;
                        }
                        """
                    )
                except Exception:
                    pass
                if clicked_via_js:
                    deadline = time.time() + 6.0
                    while time.time() < deadline:
                        if claim_dialog_is_open(page):
                            break
                        time.sleep(0.3)
                    print(f"Modal Seleccionar accion abierto: {claim_dialog_is_open(page)}")

                    if getattr(args, "interactivo_impugnar", False):
                        tope = float(getattr(args, "pausa_interactiva_s", 600.0))
                        print("=" * 70)
                        print("MODO INTERACTIVO: el modal 'Seleccionar accion' esta abierto.")
                        print("  1) En la ventana del NAVEGADOR: click en 'Impugnar'.")
                        print("  2) Si aparece 'Verifica que eres tu' (2FA), completalo.")
                        print("  3) Cuando veas el formulario de disputa (Descripcion / Motivo / Firma),")
                        print("     el script va a continuar SOLO con el resto del flujo.")
                        print(f"  Tope de espera: {tope:.0f}s (Ctrl-C para abortar).")
                        print("=" * 70)
                        strict_hints = [
                            "descripción general", "descripcion general",
                            "firma (obligatorio)", "firma obligatorio",
                            "selecciona el motivo principal",
                            "select the primary reason",
                            "i have permission",
                            "tengo permiso",
                            "argumentos",
                        ]
                        deadline = time.time() + tope
                        last_log = 0.0
                        flow_detected = False
                        while time.time() < deadline:
                            time.sleep(2.0)
                            try:
                                txt = get_active_root(page).inner_text().strip().lower()
                            except Exception:
                                txt = ""
                            hits = sum(1 for h in strict_hints if h in txt)
                            in_selector_modal = bool(P_ACTION_SELECTOR_DIALOG.search(txt))
                            in_2fa = bool(P_VERIFY_IDENTITY.search(txt))
                            if hits >= 1 and not in_selector_modal and not in_2fa:
                                flow_detected = True
                                break
                            if time.time() - last_log > 20:
                                remaining = int(deadline - time.time())
                                estado = "selector" if in_selector_modal else ("2fa" if in_2fa else "otro")
                                print(f"  ...esperando formulario de disputa (~{remaining}s, hints={hits}, estado={estado})")
                                last_log = time.time()
                        if flow_detected:
                            print("Formulario de disputa detectado. Continuando flujo automatico desde Paso 3...")
                            ok = impugnar_una_reclamacion(
                                page,
                                args.delay,
                                args.espera_envio,
                                selectors,
                                modo_modal=True,
                                skip_open_step=True,
                                skip_dispute_step=True,
                            )
                            if ok is True:
                                dismiss_claim_dialog(page, delay_s=min(args.delay, 1.0))
                                open_claim_modal_with_recovery(
                                    page, item,
                                    delay_s=args.delay,
                                    studio_url=args.url,
                                    max_attempts=max(1, args.reintentos_modal),
                                )
                                acciones_revisadas = wait_for_take_action_buttons(page, timeout_s=10.0)
                                if acciones_revisadas < acciones_actuales:
                                    reclamos_video += 1
                                    print(f"Impugnacion #{reclamos_video} confirmada (CTAs: {acciones_actuales} -> {acciones_revisadas}).")
                                    time.sleep(args.espera_entre)
                                    continue
                                print("Envio no confirmado por reduccion de CTAs.")
                                fallo_video = True
                                break
                            print(f"impugnar_una_reclamacion devolvio: {ok!r}.")
                            fallo_video = True
                            break
                        print("Tiempo agotado sin detectar formulario de disputa.")
                        fallo_video = True
                        break

                    impugnar_clicked = False
                    try:
                        impugnar_clicked = page.evaluate(
                            """
                            () => {
                              const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
                              let target_dialog = null;
                              for (const d of dialogs) {
                                const t = (d.innerText || '').toLowerCase();
                                if (/seleccionar acci|select action/.test(t)) {
                                  target_dialog = d;
                                  break;
                                }
                              }
                              const scope = target_dialog || document;
                              const candidates = Array.from(scope.querySelectorAll(
                                'button, a, [role="button"], ytcp-button-shape, tp-yt-paper-button'
                              ));
                              for (const el of candidates) {
                                const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                                if (!/^impugnar(\\b|\\s)|^dispute(\\b|\\s)/.test(t)) continue;
                                const r = el.getBoundingClientRect();
                                if (r.width <= 0 || r.height <= 0) continue;
                                el.scrollIntoView({block: 'center'});
                                el.click();
                                return true;
                              }
                              return false;
                            }
                            """
                        )
                    except Exception:
                        pass
                    if impugnar_clicked:
                        print("Click directo en 'Impugnar' (JS dentro del modal Seleccionar accion).")
                        time.sleep(1.0)
                        try:
                            page.evaluate(
                                """
                                () => {
                                  const all = Array.from(document.querySelectorAll(
                                    'button, [role="button"], ytcp-button-shape'
                                  ));
                                  for (const el of all) {
                                    const t = (el.innerText || '').trim().toLowerCase();
                                    if (/^continuar$|^continue$/.test(t)) {
                                      const r = el.getBoundingClientRect();
                                      if (r.width <= 0) continue;
                                      const dis = el.disabled || el.getAttribute('aria-disabled') === 'true';
                                      if (dis) continue;
                                      el.click();
                                      return true;
                                    }
                                  }
                                  return false;
                                }
                                """
                            )
                        except Exception:
                            pass
                        time.sleep(1.5)

            if acciones_actuales <= 0:
                print("No se detecto accion por conteo; intentando abrir disputa de forma directa...")
                probe_open = open_take_action_menu(
                    page,
                    selectors,
                    args.delay,
                    modo_modal=True,
                    max_rounds=2,
                )
                if probe_open is None:
                    break
                if not probe_open:
                    break
                print("Se logro abrir disputa por intento directo.")
                ok = impugnar_una_reclamacion(
                    page,
                    args.delay,
                    args.espera_envio,
                    selectors,
                    modo_modal=True,
                    skip_open_step=True,
                )
            else:
                ok = impugnar_una_reclamacion(
                    page,
                    args.delay,
                    args.espera_envio,
                    selectors,
                    modo_modal=True,
                )
            if ok == "verify_identity_required":
                print("Marcando video como 'requiere_verificacion_2fa' y saltando al siguiente.")
                item["status"] = "requiere_verificacion_2fa"
                item["motivo"] = "YouTube exigio verificacion de identidad (2FA) antes de impugnar."
                save_queue(queue_path, items)
                update_checkpoint(checkpoint_path, processed, key, "requiere_verificacion_2fa")
                dismiss_claim_dialog(page, delay_s=min(args.delay, 1.0))
                fallo_video = False
                reclamos_video = -1
                break
            if ok is None:
                if wait_for_take_action_buttons(page, timeout_s=4.0) > 0:
                    print("Aun hay acciones 'Tomar medidas'. Reintentando en este video...")
                    time.sleep(min(args.espera_entre, 1.5))
                    continue
                break
            if not ok:
                fallo_video = True
                break
            dismiss_claim_dialog(page, delay_s=min(args.delay, 1.0))
            if not open_claim_modal_with_recovery(
                page,
                item,
                delay_s=args.delay,
                studio_url=args.url,
                max_attempts=max(1, args.reintentos_modal),
            ):
                print("No se pudo reabrir el modal tras enviar; se marca error para evitar falso OK.")
                fallo_video = True
                break
            acciones_revisadas = wait_for_take_action_buttons(page, timeout_s=10.0)
            print(
                "Validacion post-envio: "
                f"acciones {acciones_actuales} -> {acciones_revisadas}"
            )
            if acciones_revisadas >= acciones_actuales:
                print(
                    "No se confirmo reduccion real de reclamaciones en el mismo video; "
                    "se detiene para evitar falso envio."
                )
                fallo_video = True
                break
            reclamos_video += 1
            time.sleep(args.espera_entre)

        if item.get("status") == "requiere_verificacion_2fa":
            time.sleep(args.espera_entre)
            continue

        if reclamos_video >= max_reclamos_por_video:
            print("Se alcanzo el limite maximo de reclamaciones por video; se deja pendiente para reintento.")

        if not claim_dialog_is_open(page):
            if not open_claim_modal_with_recovery(
                page,
                item,
                delay_s=args.delay,
                studio_url=args.url,
                max_attempts=max(1, args.reintentos_modal),
            ):
                print("No se pudo reabrir modal al final para validar estado; se marca error.")
                fallo_video = True

        acciones_restantes = count_take_action_buttons(page) if claim_dialog_is_open(page) else -1
        if reclamos_video == 0 and not fallo_video:
            dump_dispute_evidence(page, item, "sin_accion_pre_dismiss", getattr(args, "evidencia_dir", None))
        dismiss_claim_dialog(page, delay_s=args.delay)

        if fallo_video:
            print("No se pudo completar la impugnacion de este video.")
            dump_dispute_evidence(page, item, "error_formulario", getattr(args, "evidencia_dir", None))
            item["status"] = "error_formulario"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "error_formulario")
            return False

        if acciones_restantes < 0:
            print("No se pudo validar estado final de acciones; se deja en error para reintento.")
            item["status"] = "error_validacion"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "error_validacion")
            return False

        if acciones_restantes > 0:
            print(f"Quedaron reclamaciones pendientes en este video ({acciones_restantes}). Se deja para reintento.")
            item["status"] = "pendiente"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "pendiente")
            time.sleep(args.espera_entre)
            continue

        if reclamos_video == 0:
            print("El video no tenia acciones pendientes dentro del modal.")
            dump_dispute_evidence(page, item, "sin_accion", getattr(args, "evidencia_dir", None))
            item["status"] = "sin_accion"
            save_queue(queue_path, items)
            update_checkpoint(checkpoint_path, processed, key, "sin_accion")
            time.sleep(args.espera_entre)
            continue

        total += 1
        item["status"] = "ok"
        item["reclamos_procesados"] = reclamos_video
        save_queue(queue_path, items)
        update_checkpoint(checkpoint_path, processed, key, "ok")
        print(f"Impugnaciones completadas para el video: {reclamos_video}")
        time.sleep(args.espera_entre)

    print(f"Proceso automatico finalizado. Videos procesados: {total}")
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
    profile_name = (args.profile or DEFAULT_PROFILE_NAME or "Default").strip()
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
        candidates = []
        if DEFAULT_USER_DATA_DIR:
            candidates.append(
                describe_profile_source(Path(DEFAULT_USER_DATA_DIR), profile_name=profile_name, name="chromium-pw")
            )
        candidates.append(describe_profile_source(PLAYWRIGHT_PROFILE_DIR, profile_name=profile_name, name="repo"))
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
        stamp = int(time.time())
        runtime_dir = DEFAULT_RUNTIME_PROFILE_DIR / f"{source['name']}_{profile_name.lower().replace(' ', '_')}_{stamp}"
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
    if headless:
        launch_args.extend(
            [
                "--disable-setuid-sandbox",
                "--no-zygote",
                "--single-process",
                "--disable-gpu",
            ]
        )
    if profile_name:
        launch_args.append(f"--profile-directory={profile_name}")

    chrome_user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    kwargs = dict(
        user_data_dir=str(user_data_dir),
        headless=headless,
        args=launch_args,
        viewport=None,
        chromium_sandbox=False,
        user_agent=chrome_user_agent,
        slow_mo=120 if not headless else 0,
    )
    if executable_path:
        kwargs["executable_path"] = executable_path
    elif channel:
        kwargs["channel"] = channel
    return playwright.chromium.launch_persistent_context(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Impugnar reclamaciones en YouTube Studio usando Playwright.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL inicial de YouTube Studio.")
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin interfaz grafica.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help="Canal del navegador (chrome, msedge). Usa 'none' para Chromium integrado.")
    parser.add_argument("--executable-path", default=None, help="Ruta del ejecutable del navegador (Chromium/Chrome).")
    parser.add_argument("--user-data-dir", default=None, help="Ruta del perfil del navegador para reutilizar sesion.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help="Nombre del perfil dentro del navegador (Default, Profile 1).")
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
    parser.add_argument("--desde-modal", action="store_true", default=True, help="Iniciar desde el modal de reclamaciones y procesar todas.")
    parser.add_argument("--max", type=int, default=0, help="Cantidad de impugnaciones a procesar (0 = infinito).")
    parser.add_argument("--delay", type=float, default=2.0, help="Segundos de espera corta entre pasos.")
    parser.add_argument("--espera-envio", type=float, default=6.0, help="Segundos de espera despues de enviar.")
    parser.add_argument("--no-esperar", action="store_true", help="No esperar confirmacion manual al inicio.")
    parser.add_argument("--espera-entre", type=float, default=2.0, help="Segundos de espera entre impugnaciones.")
    parser.add_argument(
        "--evidencia-dir",
        default=str(Path(__file__).resolve().parent / "data" / "impugnar_evidencia"),
        help="Directorio donde guardar HTML/screenshot/JSON cuando un video cae en sin_accion / error_modal.",
    )
    parser.add_argument(
        "--saltar-programados",
        action="store_true",
        default=False,
        help="(Opt-in) Saltar videos marcados como 'Programado' sin abrir modal. Por defecto NO se saltan: los videos privados/programados SI son disputables.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Re-escanear y re-procesar hasta que no queden reclamos pendientes (o hasta cortar con Ctrl-C).",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=20,
        help="Tope de iteraciones en modo --loop (seguridad).",
    )
    parser.add_argument(
        "--esperar-verificacion-s",
        type=float,
        default=0.0,
        help="Cuando aparece el modal 'Verifica que eres tu' (2FA), esperar N segundos para completarlo manualmente en la ventana del navegador. 0 = no esperar.",
    )
    parser.add_argument(
        "--interactivo-impugnar",
        action="store_true",
        help="Modo interactivo: cuando el modal 'Seleccionar accion' esta abierto, el script pausa para que VOS clickees 'Impugnar' y pases el 2FA manualmente. Sigue solo cuando detecta el formulario de disputa.",
    )
    parser.add_argument(
        "--preparar-pestanas",
        action="store_true",
        help="Modo PESTAÑAS: abre una pestaña por reclamo, completa el formulario, deja el boton 'Enviar' SIN clickear. Vos vas tab por tab dandole Send.",
    )
    parser.add_argument(
        "--auto-enviar-listas",
        action="store_true",
        help="Despues de armar las tabs en modo --preparar-pestanas, recorre las que tienen [LISTA-ENVIAR] y dispara los 3 checks + Enviar via Playwright (trusted clicks).",
    )
    parser.add_argument(
        "--probar-envio",
        type=int,
        default=0,
        help="Si --auto-enviar-listas esta activo, envia solo las primeras N tabs (para probar antes de soltar el resto). 0 = todas.",
    )
    parser.add_argument(
        "--max-tabs",
        type=int,
        default=0,
        help="Corte total por cantidad de tabs [LISTA-ENVIAR] armadas (cuenta reclamos, NO videos). Util para probar: --max-tabs 1 arma 1 tab y para. 0 = sin limite.",
    )
    parser.add_argument(
        "--pausa-interactiva-s",
        type=float,
        default=600.0,
        help="Tope de espera (segundos) para pausas interactivas (2FA, modo --interactivo-impugnar). NO se usa entre vueltas: el loop espera ENTER.",
    )
    parser.add_argument(
        "--omitir-fecha",
        type=str,
        default="",
        help="Fechas marcador en formato YYYY-MM-DD (CSV soportado). Videos cuya celda de fecha matchee CUALQUIERA "
             "de las fechas se SALTAN sin abrir modal. Ej: 2028-04-30 o 2028-04-30,2028-05-01.",
    )
    parser.add_argument(
        "--mem-pausa-mb",
        type=float,
        default=2000.0,
        help="Si la RAM disponible cae debajo de este umbral (MB), el script PAUSA automaticamente. 0 = desactivado.",
    )
    parser.add_argument(
        "--mem-reanudar-mb",
        type=float,
        default=3500.0,
        help="Umbral de reanudacion (MB) cuando la pausa por memoria baja esta activa. Debe ser > --mem-pausa-mb.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args._omitir_fecha_pat = build_omitir_fecha_pattern(getattr(args, "omitir_fecha", ""))
    if args._omitir_fecha_pat is not None:
        print(f"Filtro activo: se OMITEN videos con fecha {args.omitir_fecha} (variantes es/en/numericas).")
    set_memoria_thresholds(
        getattr(args, "mem_pausa_mb", 0.0),
        getattr(args, "mem_reanudar_mb", 0.0),
    )
    if _MEM_PAUSA_MB > 0:
        print(
            f"Monitor de memoria: pausa < {_MEM_PAUSA_MB:.0f} MB libres, "
            f"reanuda >= {_MEM_REANUDAR_MB:.0f} MB libres."
        )
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
        context = None
        launch_attempts = [(channel, executable_path)]
        primary_exec = executable_path
        if primary_exec:
            primary_exec_lower = str(primary_exec).lower()
        else:
            primary_exec_lower = ""

        fallback_candidates = []
        if not args.executable_path:
            chromium_exec = find_browser_executable("chromium")
            if chromium_exec and chromium_exec != primary_exec:
                fallback_candidates.append((None, chromium_exec))
            chrome_exec = find_browser_executable("chrome")
            if chrome_exec and chrome_exec != primary_exec and chrome_exec != chromium_exec:
                fallback_candidates.append((None, chrome_exec))

        if "brave" in primary_exec_lower:
            launch_attempts.extend(fallback_candidates)
        else:
            for candidate in fallback_candidates:
                if candidate not in launch_attempts:
                    launch_attempts.append(candidate)

        last_launch_error = None
        for attempt_index, (attempt_channel, attempt_executable) in enumerate(launch_attempts, start=1):
            try:
                if attempt_index > 1:
                    print(
                        "Reintentando apertura de navegador con ejecutable alterno: "
                        f"{attempt_executable or attempt_channel}"
                    )
                context = launch_context(
                    p,
                    user_data_dir,
                    args.headless,
                    attempt_channel,
                    profile_name,
                    attempt_executable,
                )
                executable_path = attempt_executable
                channel = attempt_channel
                break
            except Exception as exc:
                last_launch_error = exc
                print(
                    "Fallo al abrir navegador con "
                    f"{attempt_executable or attempt_channel}: {exc.__class__.__name__}"
                )
                time.sleep(1.0)

        if context is None:
            raise last_launch_error

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

            if getattr(args, "preparar_pestanas", False):
                set_verify_identity_wait(getattr(args, "esperar_verificacion_s", 0.0))
                preparar_pestanas_modo(context, page, args, selectors)
                return

            if args.auto_detect:
                set_verify_identity_wait(getattr(args, "esperar_verificacion_s", 0.0))
                if not getattr(args, "loop", False):
                    procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path)
                    return
                max_loops = max(1, int(getattr(args, "max_loops", 20)))
                for vuelta in range(1, max_loops + 1):
                    print(f"\n========== LOOP {vuelta}/{max_loops} ==========")
                    procesar_cola_automatica(page, args, selectors, queue_path, checkpoint_path)
                    try:
                        with open(queue_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        items = data.get("items", []) if isinstance(data, dict) else []
                        pendientes = [
                            it for it in items
                            if it.get("status") not in ("ok", "programado_no_publicado", "omitido_fecha_marcador")
                        ]
                        verify_blocked = sum(
                            1 for it in items if it.get("status") == "requiere_verificacion_2fa"
                        )
                        print(
                            f"Loop {vuelta} fin. Total items: {len(items)} | "
                            f"Pendientes: {len(pendientes)} | bloqueados-2fa: {verify_blocked}"
                        )
                        if not pendientes:
                            print("No quedan reclamos pendientes. Loop terminado.")
                            break
                        if pendientes and verify_blocked == len(pendientes):
                            print(
                                "Todos los pendientes estan bloqueados por 2FA. Sin avance posible "
                                "sin completar verificacion. Cortando loop."
                            )
                            break
                    except Exception as exc:
                        print(f"No se pudo evaluar la cola tras loop {vuelta}: {exc}")
                    time.sleep(min(args.espera_entre, 2.0))
                return

            total = 0
            while True:
                if args.max and total >= args.max:
                    break

                ok = impugnar_una_reclamacion(page, args.delay, args.espera_envio, selectors, modo_modal=args.desde_modal)
                if ok is None:
                    print("No hay mas reclamaciones para impugnar.")
                    break
                if not ok:
                    print("No se pudo completar la impugnacion. Revisa la pantalla y los selectores.")
                    break

                total += 1
                print(f"Impugnaciones completadas: {total}")
                time.sleep(args.espera_entre)
        except KeyboardInterrupt:
            print("\nProceso detenido por el usuario.")
        finally:
            try:
                context.close()
            except BaseException:
                pass


if __name__ == "__main__":
    main()
