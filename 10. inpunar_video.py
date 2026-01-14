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

from config import PLAYWRIGHT_PROFILE_DIR, PLAYWRIGHT_SELECTORS_DIR

mensaje = """Estimados representantes:

Me dirijo a ustedes con el propósito de solicitar autorización para compartir su música en mi canal de YouTube. He intentado obtener los permisos correspondientes del álbum, sin embargo, no he logrado encontrar información de contacto oficial de la banda o sus representantes para realizar esta solicitud de manera formal.

Me gustaría establecer una comunicación directa con ustedes para discutir los términos y condiciones que consideren apropiados para el uso de su material musical. Estoy dispuesto a evaluar diferentes opciones que beneficien a ambas partes.

Quedo atento a su respuesta y disponible para proporcionar cualquier información adicional que requieran.

Saludos cordiales desde Honduras,"""
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

P_CONTINUAR = re.compile(r"(continuar|siguiente|next|continue|submit|enviar|send)", re.IGNORECASE)
P_IMPUGNAR = re.compile(r"(impugnar|disputar|dispute|take action|tomar medidas)", re.IGNORECASE)
P_IMPUGNAR_CONFIRMAR = re.compile(r"(continuar con.*impugn|confirmar.*impugn|seguir.*impugn|dispute|dispute claim|impugnar)", re.IGNORECASE)
P_SELECCIONAR = re.compile(
    r"(seleccionar.*canci[oó]n|select.*song|ver detalles|see details|detalles de la reclamaci[oó]n|reclamaci[oó]n de derechos de autor|copyright claim)",
    re.IGNORECASE,
)
P_LICENCIA = re.compile(r"(licencia|license|permiso|permission)", re.IGNORECASE)
P_ACEPTAR_TERMINOS = re.compile(r"(acepto.*terminos|accept.*terms|agree.*terms|aceptar.*terminos)", re.IGNORECASE)
P_INFO_LICENCIA = re.compile(
    r"(informaci[oó]n.*licencia|license information|detalles.*licencia|describe.*licencia)",
    re.IGNORECASE,
)
P_FIRMA = re.compile(r"(firma|signature|nombre completo|nombre y apellido|full name)", re.IGNORECASE)
P_CERRAR = re.compile(r"(cerrar|finalizar|listo|done|close)", re.IGNORECASE)
P_RATIONALE = re.compile(r"(rationale|details|reason|motivo|justificaci[oó]n|razonamiento|detalles)", re.IGNORECASE)
P_REASON = re.compile(r"(reason|motivo|raz[oó]n)", re.IGNORECASE)
P_DETAILS = re.compile(r"(details|detalles)", re.IGNORECASE)
P_READ_CONFIRM = re.compile(
    r"(please read.*check the box|read the text above|check the box to continue|marque.*casilla|marca.*casilla|lee.*texto|leer.*arriba)",
    re.IGNORECASE,
)
P_CONFIRM_PERMISSION = re.compile(
    r"(i have permission to use the content|permission to use the content|copyright owner|tengo permiso|permiso para usar|estoy autorizado)",
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
            target.click()
            return True
        except Exception:
            pass
    css = info.get("css")
    if css:
        try:
            target = root.locator(css).first
            target.wait_for(state="visible", timeout=timeout_ms)
            target.scroll_into_view_if_needed()
            target.click()
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
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if click_by_selector_info(page, selectors.get("impugnar_confirmar"), timeout_ms=1000, optional=True):
            return True
        if click_action(
            page,
            [P_IMPUGNAR_CONFIRMAR],
            roles=("button", "menuitem", "link"),
            timeout_ms=1000,
            optional=True,
        ):
            return True
        time.sleep(0.4)
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
        root = get_active_root(page)
        locator = locator_from_selector_info(root, selectors.get("continuar"))
        if locator is None:
            try:
                locator = root.get_by_role("button", name=P_CONTINUAR)
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
                target.click()
                return True
            except Exception:
                pass
        if click_action(root, [P_CONTINUAR], roles=("button", "link"), timeout_ms=2000, optional=True):
            return True
        click_read_checkbox(root, page)
        time.sleep(delay_s)
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


def impugnar_una_reclamacion(page, delay_s, espera_envio_s, selectors, modo_modal=False):
    root = page
    selectors = selectors or {}
    print("Buscando la accion de impugnacion...")

    print("Paso 1/7: abrir menu Take action / Impugnar.")
    if not click_with_selector(
        root,
        selectors.get("impugnar"),
        [P_IMPUGNAR],
        roles=("button", "menuitem", "link"),
        optional=True,
        descripcion="Impugnar",
    ):
        if not click_with_selector(
            root,
            selectors.get("seleccionar_cancion"),
            [P_SELECCIONAR],
            roles=("button", "link", "row"),
            optional=True,
            descripcion="Seleccionar cancion",
        ):
            if modo_modal:
                print("No se encontraron mas reclamaciones en el modal.")
                return None
            return False
        time.sleep(delay_s)
        if not click_with_selector(
            root,
            selectors.get("impugnar"),
            [P_IMPUGNAR],
            roles=("button", "menuitem", "link"),
            optional=False,
            descripcion="Impugnar",
        ):
            return False

    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 2/7: seleccionar Dispute / Impugnar.")
    if not click_dispute_step(page, selectors):
        print("No se encontro el boton de Dispute/Impugnar.")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 3/7: esperar las opciones de Reason/Details.")
    if not wait_for_license_step(root):
        print("No se encontraron las opciones de licencia.")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 4/7: marcar 'I have permission / license'.")
    if not select_license_option(page, selectors, delay_s):
        print("No se encontro el elemento: Licencia")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 5/7: continuar al aviso de lectura.")
    if not click_continue_step(page, selectors, delay_s):
        print("No se pudo avanzar al aviso de lectura.")
        return False
    time.sleep(delay_s)
    root = get_active_root(page)

    print("Paso 6/7: marcar la casilla de confirmacion y continuar.")
    selector_lectura = selectors.get("confirmar_lectura") or selectors.get("aceptar_terminos")
    if not click_read_checkbox(root, page, selector_lectura):
        if check_all_visible_checkboxes(root) == 0:
            print("No se pudo marcar la casilla de confirmacion.")
    time.sleep(delay_s)
    if not click_continue_step(page, selectors, delay_s):
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
    if not try_fill_info_licencia(page, root, selectors, mensaje, delay_s):
        print("No se encontro el campo de informacion de licencia.")
        return False

    time.sleep(delay_s)
    check_all_visible_checkboxes(root)

    if not (
        fill_with_selector(root, selectors.get("firma"), [P_FIRMA], firma)
        or fill_signature_fallback(root, firma)
        or fill_last_textbox_fallback(root, firma)
    ):
        print("No se encontro el campo de firma.")
        return False

    time.sleep(delay_s)
    click_repeatedly(root, [P_CONTINUAR], max_clicks=1, selector_info=selectors.get("continuar"))
    time.sleep(espera_envio_s)
    click_with_selector(
        root,
        selectors.get("cerrar"),
        [P_CERRAR],
        roles=("button", "link"),
        optional=True,
        descripcion="Cerrar",
    )
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


def resolve_user_data_dir(args):
    if args.user_data_dir:
        return Path(args.user_data_dir), True

    env_dir = os.environ.get("PLAYWRIGHT_USER_DATA_DIR")
    if env_dir:
        return Path(env_dir), True

    env_profile_dir = os.environ.get("PLAYWRIGHT_PROFILE_DIR")
    if env_profile_dir:
        return Path(env_profile_dir), True

    if args.usar_perfil_chrome:
        chrome_dir = find_chrome_user_data_dir()
        if not chrome_dir:
            return None, False
        return chrome_dir, False

    return PLAYWRIGHT_PROFILE_DIR, True


def resolve_launch_config(args):
    executable_path = args.executable_path or DEFAULT_EXECUTABLE_PATH
    if executable_path:
        return None, executable_path

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
    parser = argparse.ArgumentParser(description="Impugnar reclamaciones en YouTube Studio usando Playwright.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL inicial de YouTube Studio.")
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin interfaz grafica.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help="Canal del navegador (chrome, msedge). Usa 'none' para Chromium integrado.")
    parser.add_argument("--executable-path", default=None, help="Ruta del ejecutable del navegador (Chromium/Chrome).")
    parser.add_argument("--user-data-dir", default=DEFAULT_USER_DATA_DIR, help="Ruta del perfil del navegador para reutilizar sesion.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help="Nombre del perfil dentro del navegador (Default, Profile 1).")
    parser.add_argument("--usar-perfil-chrome", action="store_true", help="Usar el perfil local de Chrome para reutilizar sesion.")
    parser.add_argument("--aprender", action="store_true", help="Grabar selectores manualmente para esta pantalla.")
    parser.add_argument("--selectores", default=None, help="Ruta del archivo JSON de selectores.")
    parser.add_argument("--grabar", action="store_true", help="Grabar todas las acciones para esta pantalla.")
    parser.add_argument("--acciones", default=None, help="Ruta del archivo JSON de acciones grabadas.")
    parser.add_argument("--desde-modal", action="store_true", default=True, help="Iniciar desde el modal de reclamaciones y procesar todas.")
    parser.add_argument("--max", type=int, default=0, help="Cantidad de impugnaciones a procesar (0 = infinito).")
    parser.add_argument("--delay", type=float, default=2.0, help="Segundos de espera corta entre pasos.")
    parser.add_argument("--espera-envio", type=float, default=6.0, help="Segundos de espera despues de enviar.")
    parser.add_argument("--no-esperar", action="store_true", help="No esperar confirmacion manual al inicio.")
    parser.add_argument("--espera-entre", type=float, default=2.0, help="Segundos de espera entre impugnaciones.")
    return parser.parse_args()


def main():
    args = parse_args()
    user_data_dir, allow_create = resolve_user_data_dir(args)
    if user_data_dir is None:
        print("No se encontro un perfil local de Chrome. Usa --user-data-dir para indicar la ruta.")
        return
    if not user_data_dir.exists():
        if allow_create:
            user_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"No existe la ruta del perfil: {user_data_dir}")
            return

    channel, executable_path = resolve_launch_config(args)
    profile_name = (args.profile or "").strip()
    if args.usar_perfil_chrome:
        print(f"Usando perfil de Chrome: {user_data_dir}")
        print("Cierra Chrome antes de continuar para evitar bloqueo del perfil.")
    if executable_path:
        print(f"Usando ejecutable del navegador: {executable_path}")

    selectors_path = Path(args.selectores) if args.selectores else SELECTORS_PATH
    actions_path = Path(args.acciones) if args.acciones else ACTIONS_PATH

    with sync_playwright() as p:
        context = launch_context(p, user_data_dir, args.headless, channel, profile_name, executable_path)
        try:
            page = context.new_page()
            page.set_default_timeout(15000)
            page.goto(args.url, wait_until="domcontentloaded")

            if not args.no_esperar:
                input("Inicia sesion y entra a la pantalla de reclamaciones. Presiona Enter para continuar...")

            if args.grabar:
                record_actions(page, actions_path)
                return

            if args.aprender:
                learn_selectors(page, selectors_path, SELECTOR_STEPS)
                return

            selectors = load_selectors(selectors_path)
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
