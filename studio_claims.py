import json
import os
import re
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


P_CONTENT_MENU = re.compile(r"(contenido|content)", re.IGNORECASE)
P_CHANNEL_CONTENT = re.compile(r"(contenido del canal|channel content)", re.IGNORECASE)
P_CLAIM = re.compile(r"(derechos de autor|copyright)", re.IGNORECASE)
P_VIEW_DETAILS = re.compile(r"(ver detalles|see details|details)", re.IGNORECASE)
P_CLAIM_DIALOG = re.compile(
    r"(detalles sobre los derechos de autor del video|copyright details|copyright claim|"
    r"seleccionar acci[oó]n|select action|"
    r"borrar canci[oó]n|reemplazar canci[oó]n|recortar segmento|"
    r"impugna una reclamaci[oó]n)",
    re.IGNORECASE,
)
P_CLOSE_DIALOG = re.compile(r"(cerrar|close)", re.IGNORECASE)
P_SWITCH_NEW_STUDIO = re.compile(
    r"(cambiar a la nueva versi[oó]n de studio|switch to the new studio|new version of studio)",
    re.IGNORECASE,
)
P_SELECT_CHANNEL = re.compile(r"(selecciona un canal|choose a channel)", re.IGNORECASE)
P_DONT_ASK_AGAIN = re.compile(r"(no volver a preguntar|don't ask again)", re.IGNORECASE)
P_FEEDBACK_TITLE = re.compile(r"(enviar comentarios a google|send feedback to google|send feedback)", re.IGNORECASE)
P_FEEDBACK_CLOSE = re.compile(r"(cerrar|close|dismiss)", re.IGNORECASE)
P_STUDIO_TOUR = re.compile(
    r"(haz clic aqu[ií] para a[nñ]adir una captura de pantalla|click here to add a screenshot)",
    re.IGNORECASE,
)
P_STUDIO_TOUR_ACK = re.compile(r"(entendido|got it)", re.IGNORECASE)


def claim_item_key(item):
    video_id = (item or {}).get("video_id")
    if video_id:
        return video_id
    title = ((item or {}).get("title") or "").strip()
    if title:
        return title
    return ((item or {}).get("row_text") or "").strip()


def load_checkpoint(path):
    processed = {}
    if not path or not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t", 2)
            key = parts[0].strip()
            if not key:
                continue
            status = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "ok"
            processed[key] = status
    return processed


def append_checkpoint(path, key, status="ok"):
    if not path or not key:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{key}\t{status}\n")


def update_checkpoint(path, processed, key, status):
    if not path or not key:
        return False
    if processed.get(key) == status:
        return False
    append_checkpoint(path, key, status=status)
    processed[key] = status
    return True


def save_queue(path, items):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(items),
        "items": items,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def wait_for_content_list(page, timeout_ms=15000):
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        dismiss_blocking_overlays(page, delay_s=0.1, max_rounds=1)
        try:
            if "signin_prompt" in (page.url or ""):
                ensure_channel_selected(page, timeout_ms=1500)
        except Exception:
            pass
        try:
            rows = page.locator("ytcp-video-row")
            if rows.count() > 0:
                return True
        except Exception:
            pass
        try:
            title = page.get_by_text(P_CHANNEL_CONTENT).first
            if title.is_visible() and "/videos" in (page.url or ""):
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _dismiss_feedback_panel(page):
    feedback_visible = False
    try:
        heading = page.get_by_text(P_FEEDBACK_TITLE).first
        feedback_visible = heading.is_visible()
    except Exception:
        feedback_visible = False
    if not feedback_visible:
        return False

    try:
        clicked = page.evaluate(
            """
            () => {
              const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const isVisible = (node) => {
                if (!node || !node.getBoundingClientRect) return false;
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(node);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const panels = Array.from(document.querySelectorAll('div, section, aside, [role="dialog"]')).filter((node) => {
                if (!isVisible(node)) return false;
                const text = clean(node.innerText || node.textContent || '');
                return /(enviar comentarios a google|send feedback to google|send feedback)/i.test(text);
              });
              for (const panel of panels) {
                const buttons = Array.from(
                  panel.querySelectorAll('button, [role="button"], tp-yt-paper-icon-button, ytcp-icon-button')
                );
                for (const button of buttons) {
                  const label = clean(
                    button.getAttribute('aria-label') ||
                    button.getAttribute('title') ||
                    button.innerText ||
                    button.textContent ||
                    ''
                  );
                  if (!label || /(cerrar|close|dismiss)/i.test(label)) {
                    try {
                      button.click();
                      return true;
                    } catch (e) {}
                  }
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

    try:
        return _click_first(page.get_by_role("button", name=P_FEEDBACK_CLOSE), timeout_ms=700)
    except Exception:
        return False


def _dismiss_studio_tour_popup(page):
    tour_visible = False
    try:
        tour_visible = page.get_by_text(P_STUDIO_TOUR).first.is_visible()
    except Exception:
        tour_visible = False
    if not tour_visible:
        return False

    if _click_first(page.get_by_role("button", name=P_STUDIO_TOUR_ACK), timeout_ms=900):
        return True

    try:
        clicked = page.evaluate(
            """
            () => {
              const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const isVisible = (node) => {
                if (!node || !node.getBoundingClientRect) return false;
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(node);
                return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
              };
              const cards = Array.from(document.querySelectorAll('div, section, tp-yt-paper-dialog, [role="dialog"]')).filter((node) => {
                if (!isVisible(node)) return false;
                const text = clean(node.innerText || node.textContent || '');
                return /(haz clic aqu[ií] para a[nñ]adir una captura de pantalla|click here to add a screenshot)/i.test(text);
              });
              for (const card of cards) {
                const buttons = Array.from(card.querySelectorAll('button, [role="button"], a, tp-yt-paper-button, ytcp-button-shape'));
                for (const button of buttons) {
                  const label = clean(
                    button.getAttribute('aria-label') ||
                    button.getAttribute('title') ||
                    button.innerText ||
                    button.textContent ||
                    ''
                  );
                  if (/(entendido|got it)/i.test(label)) {
                    try {
                      button.click();
                      return true;
                    } catch (e) {}
                  }
                }
              }
              return false;
            }
            """
        )
        return bool(clicked)
    except Exception:
        return False


def dismiss_blocking_overlays(page, delay_s=0.15, max_rounds=3):
    closed_any = False
    for _ in range(max_rounds):
        closed_round = False
        if _dismiss_feedback_panel(page):
            closed_round = True
            closed_any = True
        if _dismiss_studio_tour_popup(page):
            closed_round = True
            closed_any = True
        if not closed_round:
            break
        time.sleep(delay_s)
    return closed_any


def ensure_modern_studio(page, timeout_ms=5000):
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        for role in ("link", "button"):
            try:
                locator = page.get_by_role(role, name=P_SWITCH_NEW_STUDIO)
                target = locator.first
                target.wait_for(state="visible", timeout=800)
                target.click()
                time.sleep(1.5)
                return True
            except Exception:
                continue
        time.sleep(0.3)
    return False


def ensure_channel_selected(page, timeout_ms=6000):
    preferred_name = (
        os.environ.get("YOUTUBE_STUDIO_CHANNEL_NAME")
        or os.environ.get("PLAYWRIGHT_STUDIO_CHANNEL")
        or "Daniel Banariba"
    ).strip()
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        try:
            visible = page.evaluate(
                """
                () => {
                  const text = String(document.body && (document.body.innerText || document.body.textContent) || '');
                  return /(selecciona un canal|choose a channel)/i.test(text);
                }
                """
            )
            if not visible:
                return False

            _click_first(page.get_by_role("checkbox", name=P_DONT_ASK_AGAIN), timeout_ms=500)
            _click_first(page.get_by_text(P_DONT_ASK_AGAIN), timeout_ms=500)

            if preferred_name:
                pattern = re.compile(re.escape(preferred_name), re.IGNORECASE)
                for role in ("button", "link", "option", "menuitem"):
                    if _click_first(page.get_by_role(role, name=pattern), timeout_ms=1000):
                        time.sleep(1.2)
                        return True
                if _click_first(page.get_by_text(pattern), timeout_ms=1000):
                    time.sleep(1.2)
                    return True

            clicked = page.evaluate(
                """
                () => {
                  const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
                  const parseSubs = (value) => {
                    const text = clean(value).toLowerCase();
                    const match = text.match(/([\\d.,\\s]+)/);
                    if (!match) return 0;
                    const digits = match[1].replace(/[^\\d]/g, '');
                    if (!digits) return 0;
                    const number = Number.parseInt(digits, 10);
                    return Number.isFinite(number) ? number : 0;
                  };
                  const rows = Array.from(
                    document.querySelectorAll('ytd-account-item-renderer, tp-yt-paper-item, [role="button"], [role="link"], a, button, li, div')
                  );
                  let best = null;
                  let bestScore = -1;
                  for (const row of rows) {
                    const text = clean(row.innerText || row.textContent);
                    if (!text) continue;
                    if (!/(suscriptores|subscribers)/i.test(text)) continue;
                    const score = parseSubs(text);
                    if (score > bestScore) {
                      best = row;
                      bestScore = score;
                    }
                  }
                  if (!best) return false;
                  try { best.scrollIntoView({ block: 'center' }); } catch (e) {}
                  try { best.click(); return true; } catch (e) {}
                  const clickable = best.querySelector('a, button, [role="button"], [role="link"]');
                  if (clickable) {
                    try { clickable.click(); return true; } catch (e) {}
                  }
                  return false;
                }
                """
            )
            if clicked:
                time.sleep(1.2)
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def ensure_content_list(page, timeout_ms=15000):
    dismiss_blocking_overlays(page, delay_s=0.12, max_rounds=2)
    if "signin_prompt" in (page.url or ""):
        ensure_channel_selected(page, timeout_ms=12000)
        time.sleep(1.2)
    ensure_channel_selected(page, timeout_ms=5000)
    ensure_modern_studio(page, timeout_ms=2000)
    if "/videos" in (page.url or "") and wait_for_content_list(page, timeout_ms=timeout_ms):
        return True
    for role in ("link", "button", "tab"):
        try:
            locator = page.get_by_role(role, name=P_CONTENT_MENU)
            target = locator.first
            target.wait_for(state="visible", timeout=2000)
            target.click()
            dismiss_blocking_overlays(page, delay_s=0.1, max_rounds=1)
            if wait_for_content_list(page, timeout_ms=timeout_ms):
                return True
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue

    try:
        target = page.locator("a[href*='/videos']").first
        target.wait_for(state="visible", timeout=2000)
        target.click()
        dismiss_blocking_overlays(page, delay_s=0.1, max_rounds=1)
        if wait_for_content_list(page, timeout_ms=timeout_ms):
            return True
    except Exception:
        pass

    try:
        match = re.search(r"/channel/([^/?]+)", page.url or "")
        if match:
            channel_id = match.group(1)
            page.goto(
                f"https://studio.youtube.com/channel/{channel_id}/videos/upload",
                wait_until="domcontentloaded",
            )
            if wait_for_content_list(page, timeout_ms=timeout_ms):
                return True
    except Exception:
        pass

    return wait_for_content_list(page, timeout_ms=timeout_ms)


def reset_content_scroll(page):
    page.evaluate(
        """
        () => {
          window.scrollTo(0, 0);
          const nodes = Array.from(document.querySelectorAll('*'));
          for (const node of nodes) {
            try {
              if (node.scrollHeight > node.clientHeight + 100) {
                node.scrollTop = 0;
              }
            } catch (e) {}
          }
        }
        """
    )


def scroll_content_list(page):
    return page.evaluate(
        """
        () => {
          const nodes = Array.from(document.querySelectorAll('*')).filter((node) => {
            try {
              const style = window.getComputedStyle(node);
              const overflow = `${style.overflow} ${style.overflowY}`;
              return /(auto|scroll)/i.test(overflow) && node.scrollHeight > node.clientHeight + 100;
            } catch (e) {
              return false;
            }
          });
          nodes.sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight));
          const target = nodes[0] || document.scrollingElement || document.documentElement;
          const step = Math.max(700, Math.floor((target.clientHeight || window.innerHeight || 900) * 0.85));
          const before = target === document.body || target === document.documentElement || target === document.scrollingElement
            ? window.scrollY
            : target.scrollTop;
          if (target === document.body || target === document.documentElement || target === document.scrollingElement) {
            window.scrollBy(0, step);
            return { before, after: window.scrollY };
          }
          target.scrollBy(0, step);
          return { before, after: target.scrollTop };
        }
        """
    )


def extract_visible_claim_rows(page):
    return page.evaluate(
        """
        () => {
          const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
          const isDuration = (value) => /^\\d{1,2}:\\d{2}(?::\\d{2})?$/.test(clean(value));
          const isVisible = (node) => {
            if (!node) return false;
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
          };
          const rows = Array.from(document.querySelectorAll('ytcp-video-row, [role="row"], tbody tr'));
          const items = [];
          for (const row of rows) {
            if (!isVisible(row)) continue;
            const text = clean(row.innerText || row.textContent);
            if (!text) continue;
            if (/detalles sobre los derechos de autor del video|copyright details|copyright claim/i.test(text)) {
              continue;
            }
            if (!/(derechos de autor|copyright)/i.test(text)) continue;

            const links = Array.from(row.querySelectorAll('a[href]'));
            let videoId = '';
            let href = '';
            let title = '';

            const named = row.querySelector('#video-title, [id="video-title"], a[title], [title]');
            if (named) {
              const namedTitle = clean(named.getAttribute('title') || named.innerText || named.textContent);
              if (namedTitle && !isDuration(namedTitle)) {
                title = namedTitle;
              }
            }

            for (const link of links) {
              const rawHref = link.getAttribute('href') || '';
              const absoluteHref = link.href || rawHref;
              const linkText = clean(link.innerText || link.textContent);
              const match = absoluteHref.match(/(?:\\/video\\/|v=)([A-Za-z0-9_-]{11})/);
              if (!videoId && match) {
                videoId = match[1];
                href = absoluteHref;
              }
              if (
                !title &&
                linkText &&
                !isDuration(linkText) &&
                !/editar|edit|analytics|comment|copyright/i.test(linkText)
              ) {
                title = linkText;
              }
            }

            if (!title) {
              const textWithoutDuration = text.replace(/^\\d{1,2}:\\d{2}(?::\\d{2})?\\s+/, '');
              title = clean(textWithoutDuration).slice(0, 160);
            }

            let restrictionText = '';
            let dateText = '';
            const datePat = /(\\b\\d{1,2}\\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sept?|oct|nov|dic|jan|apr|aug|dec)\\.?\\s+\\d{4}\\b)|(\\b(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\\.?\\s+\\d{1,2},?\\s+\\d{4}\\b)|(\\b\\d{4}-\\d{2}-\\d{2}\\b)|(\\b\\d{1,2}\\/\\d{1,2}\\/\\d{4}\\b)/i;
            const dateCell = row.querySelector('#date, #date-cell, [id^="date"]');
            if (dateCell) {
              const cellTxt = clean(dateCell.innerText || dateCell.textContent);
              if (cellTxt) dateText = cellTxt;
            }
            const nodes = Array.from(row.querySelectorAll('*'));
            for (const node of nodes) {
              const nodeText = clean(node.innerText || node.textContent);
              const aria = clean(node.getAttribute && node.getAttribute('aria-label'));
              if (!restrictionText && /(derechos de autor|copyright)/i.test(`${nodeText} ${aria}`)) {
                restrictionText = nodeText || aria;
              }
              if (!dateText && nodeText && nodeText.length < 60) {
                const m = nodeText.match(datePat);
                if (m) dateText = m[0];
              }
            }

            items.push({
              video_id: videoId,
              href,
              title: title || text.slice(0, 160),
              row_text: text.slice(0, 1400),
              restriction_text: restrictionText || 'Derechos de autor',
              date_text: dateText,
            });
          }
          return items;
        }
        """
    )


def click_next_page(page):
    """Avanza a la siguiente pagina del paginador de YouTube Studio.

    Devuelve True si pudo clickear un boton 'siguiente pagina' habilitado.
    """
    try:
        result = page.evaluate(
            """
            () => {
              const selectors = [
                '#navigate-after',
                'tp-yt-paper-icon-button#navigate-after',
                'paper-icon-button#navigate-after',
                'ytcp-icon-button#navigate-after',
                'button#navigate-after',
                '[aria-label*="Next page" i]',
                '[aria-label*="Pagina siguiente" i]',
                '[aria-label*="Página siguiente" i]',
                '[aria-label*="Siguiente pagina" i]',
                '[aria-label*="Siguiente página" i]',
              ];
              const seen = new Set();
              const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return false;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') return false;
                return true;
              };
              const isDisabled = (el) => {
                let cur = el;
                for (let i = 0; i < 6 && cur; i++) {
                  if (cur.hasAttribute && (cur.hasAttribute('disabled')
                      || cur.getAttribute('aria-disabled') === 'true'
                      || cur.disabled === true)) return true;
                  cur = cur.parentElement;
                }
                return false;
              };
              const findClickableAncestor = (el) => {
                let cur = el;
                for (let i = 0; i < 10 && cur; i++) {
                  const tag = (cur.tagName || '').toLowerCase();
                  if (tag === 'button' || tag === 'a'
                      || tag === 'tp-yt-paper-icon-button'
                      || tag === 'paper-icon-button'
                      || tag === 'ytcp-icon-button'
                      || tag === 'yt-button-shape'
                      || (cur.getAttribute && cur.getAttribute('role') === 'button')
                      || (cur.hasAttribute && cur.hasAttribute('aria-label'))) {
                    return cur;
                  }
                  cur = cur.parentElement;
                }
                return el;
              };
              const tryClick = (btn) => {
                if (!btn || seen.has(btn)) return false;
                seen.add(btn);
                if (!isVisible(btn)) return false;
                if (isDisabled(btn)) return false;
                try { btn.scrollIntoView({ block: 'center' }); } catch (e) {}
                try { btn.click(); return true; } catch (e) {}
                return false;
              };

              // 1) Selectores conocidos por id/aria-label
              for (const sel of selectors) {
                let nodes = [];
                try { nodes = Array.from(document.querySelectorAll(sel)); } catch (e) { continue; }
                for (const btn of nodes) {
                  if (tryClick(btn)) return { ok: true, via: 'selector' };
                }
              }

              // 2) Fallback: buscar el SVG del chevron derecho por su path
              //    path d="M8.793 5.293a1 1 0 000 1.414L14.086 12l-5.293 5.293..."
              //    (icono "siguiente" del paginador moderno con yt-icon-shape)
              const chevronSig = 'M8.793 5.293';
              let svgs = [];
              try { svgs = Array.from(document.querySelectorAll('svg path')); } catch (e) {}
              for (const path of svgs) {
                const d = path.getAttribute && path.getAttribute('d');
                if (!d || d.indexOf(chevronSig) === -1) continue;
                const svg = path.closest('svg');
                if (!svg) continue;
                const btn = findClickableAncestor(svg);
                if (tryClick(btn)) return { ok: true, via: 'chevron-path' };
              }

              return { ok: false };
            }
            """
        )
        return bool(result and result.get("ok"))
    except Exception:
        return False


def scan_claim_rows(page, delay_s=1.0, max_items=0, max_scrolls=35, max_pages=50,
                    is_valid=None, target_valid=0):
    """Escanea filas de copyright en YouTube Studio con paginacion.

    Avanza a la siguiente pagina solo si:
      - todavia no se alcanzo target_valid items 'validos' (segun is_valid), Y
      - la pagina actual aporto al menos un item nuevo.

    is_valid: callable opcional (item) -> bool. Por defecto todo cuenta como valido.
    target_valid: corta la paginacion apenas se acumulen N items validos. 0 = sin tope.
    """
    ordered = []
    seen = set()
    valid_count = 0

    for page_num in range(1, max_pages + 1):
        reset_content_scroll(page)
        time.sleep(min(delay_s, 1.0))

        stale_rounds = 0
        items_at_page_start = len(ordered)

        for _ in range(max_scrolls):
            new_items = 0
            for item in extract_visible_claim_rows(page) or []:
                key = claim_item_key(item)
                if not key or key in seen:
                    continue
                item["status"] = "pendiente"
                ordered.append(item)
                seen.add(key)
                new_items += 1
                if is_valid is None or is_valid(item):
                    valid_count += 1
                if max_items and len(ordered) >= max_items:
                    return ordered
                if target_valid and valid_count >= target_valid:
                    return ordered

            moved = scroll_content_list(page)
            if new_items == 0 and moved.get("after") == moved.get("before"):
                stale_rounds += 1
            else:
                stale_rounds = 0
            if stale_rounds >= 3:
                break
            time.sleep(delay_s)

        new_on_page = len(ordered) - items_at_page_start
        if new_on_page == 0:
            break
        if target_valid and valid_count >= target_valid:
            break
        if not click_next_page(page):
            break

        print(
            f"     [PAGINACION] pagina {page_num} agotada "
            f"({new_on_page} videos, {valid_count} validos hasta ahora), "
            f"avanzando a pagina {page_num + 1}..."
        )
        time.sleep(max(1.5, delay_s * 1.5))

    return ordered


def claim_dialog_is_open(page):
    try:
        dialogs = page.get_by_role("dialog")
        if dialogs.count() == 0:
            return False
        for index in range(dialogs.count() - 1, -1, -1):
            dialog = dialogs.nth(index)
            try:
                if not dialog.is_visible():
                    continue
                text = dialog.inner_text().strip()
            except Exception:
                continue
            if P_CLAIM_DIALOG.search(text) or P_CLAIM.search(text):
                return True
        return False
    except Exception:
        return False


def _locate_row_now(page, item):
    video_id = (item.get("video_id") or "").strip()
    title = (item.get("title") or "").strip()
    row_selector = "ytcp-video-row, [role='row'], tbody tr"

    if video_id:
        try:
            row = page.locator(row_selector).filter(has=page.locator(f'a[href*="{video_id}"]')).first
            if row.count() > 0 and row.is_visible():
                return row
        except Exception:
            pass

    if title:
        snippet = title[:90]
        try:
            row = page.locator(row_selector).filter(has_text=snippet).first
            if row.count() > 0 and row.is_visible():
                return row
        except Exception:
            pass

    return None


def find_row_with_scroll(page, item, delay_s=1.0, max_scrolls=30):
    for attempt in range(max_scrolls):
        row = _locate_row_now(page, item)
        if row is not None:
            try:
                row.scroll_into_view_if_needed()
            except Exception:
                pass
            return row
        moved = scroll_content_list(page)
        if moved.get("after") == moved.get("before") and attempt > 1:
            break
        time.sleep(delay_s)
    return None


def _click_first(locator, timeout_ms=2500):
    try:
        target = locator.first
        target.wait_for(state="visible", timeout=timeout_ms)
        target.scroll_into_view_if_needed()
        target.click()
        return True
    except Exception:
        try:
            target = locator.first
            target.click(force=True)
            return True
        except Exception:
            return False


def wait_for_claim_dialog(page, timeout_ms=8000, poll_s=0.25):
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if claim_dialog_is_open(page):
            return True
        time.sleep(poll_s)
    return claim_dialog_is_open(page)


def _click_claim_trigger_in_row(row):
    try:
        return row.evaluate(
            """
            (node) => {
              const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
              const candidates = Array.from(
                node.querySelectorAll('button, a, tp-yt-paper-button, [role="button"], [role="link"], span, div')
              );
              for (const candidate of candidates) {
                const text = clean(
                  candidate.innerText || candidate.textContent || candidate.getAttribute('aria-label') || ''
                );
                if (!text) continue;
                if (!/(derechos de autor|copyright|ver detalles|see details)/i.test(text)) continue;
                candidate.click();
                return true;
              }
              return false;
            }
            """
        )
    except Exception:
        return False


def open_claim_modal_for_item(page, item, delay_s=1.0):
    if claim_dialog_is_open(page):
        return True

    row = find_row_with_scroll(page, item, delay_s=delay_s)
    if row is None:
        return False

    if _click_first(row.get_by_role("button", name=P_VIEW_DETAILS)):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    elif _click_first(row.get_by_role("link", name=P_VIEW_DETAILS)):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    elif _click_first(row.get_by_role("button", name=P_CLAIM)):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    elif _click_first(row.get_by_role("link", name=P_CLAIM)):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    elif _click_first(row.get_by_text(P_CLAIM)):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    elif _click_claim_trigger_in_row(row):
        if wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000))):
            return True
    else:
        return False

    if _click_first(page.get_by_role("button", name=P_VIEW_DETAILS), timeout_ms=4000):
        return wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000)))
    elif _click_first(page.get_by_role("link", name=P_VIEW_DETAILS), timeout_ms=4000):
        return wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000)))
    elif _click_first(page.get_by_text(P_VIEW_DETAILS), timeout_ms=4000):
        return wait_for_claim_dialog(page, timeout_ms=max(5000, int((delay_s + 1.5) * 1000)))

    return claim_dialog_is_open(page)


def dismiss_claim_dialog(page, delay_s=0.5):
    for _ in range(4):
        if not claim_dialog_is_open(page):
            return True

        closed = False
        try:
            dialogs = page.get_by_role("dialog")
            target_dialog = None
            for index in range(dialogs.count() - 1, -1, -1):
                dialog = dialogs.nth(index)
                if not dialog.is_visible():
                    continue
                text = dialog.inner_text().strip()
                if P_CLAIM_DIALOG.search(text) or P_CLAIM.search(text):
                    target_dialog = dialog
                    break
            if target_dialog is not None:
                for role in ("button", "link"):
                    try:
                        locator = target_dialog.get_by_role(role, name=P_CLOSE_DIALOG)
                        if _click_first(locator, timeout_ms=1200):
                            closed = True
                            break
                    except Exception:
                        continue
        except Exception:
            pass

        if not closed:
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass

        time.sleep(delay_s)

    return not claim_dialog_is_open(page)


def _try_open_via_copyright_url(page, item, delay_s=1.0):
    video_id = (item.get("video_id") or "").strip() if isinstance(item, dict) else ""
    if not video_id:
        return False
    url = f"https://studio.youtube.com/video/{video_id}/copyright"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
    except Exception:
        return False
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    time.sleep(min(2.5, max(1.0, delay_s)))

    try:
        page.set_viewport_size({"width": 1920, "height": 1080})
    except Exception:
        pass

    js_click_first_take_action = """
    () => {
      const matches = (el) => {
        const t = (el.innerText || el.textContent || el.getAttribute('aria-label') || '').trim().toLowerCase();
        return /^\\s*(tomar medidas|take action)\\s*$/i.test(t)
            || /(tomar medidas|take action)/i.test(t) && t.length < 80;
      };
      const all = Array.from(document.querySelectorAll(
        'button, a, [role="button"], ytcp-button-shape, tp-yt-paper-button'
      ));
      for (const el of all) {
        if (!matches(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden') continue;
        el.scrollIntoView({block: 'center'});
        el.click();
        return true;
      }
      return false;
    }
    """

    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            clicked = page.evaluate(js_click_first_take_action)
            if clicked:
                end = time.time() + 6.0
                while time.time() < end:
                    try:
                        dialogs = page.get_by_role("dialog")
                        for i in range(dialogs.count() - 1, -1, -1):
                            d = dialogs.nth(i)
                            if not d.is_visible():
                                continue
                            txt = (d.inner_text() or "").lower()
                            if "seleccionar acci" in txt or "select action" in txt:
                                return True
                    except Exception:
                        pass
                    time.sleep(0.3)
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def open_claim_modal_with_recovery(page, item, delay_s=1.0, studio_url=None, max_attempts=3):
    if _try_open_via_copyright_url(page, item, delay_s=delay_s):
        return True

    for attempt in range(1, max_attempts + 1):
        dismiss_blocking_overlays(page, delay_s=min(delay_s, 0.25), max_rounds=2)
        dismiss_claim_dialog(page, delay_s=min(delay_s, 1.0))

        if not ensure_content_list(page, timeout_ms=18000):
            if not studio_url:
                return False
            try:
                page.goto(studio_url, wait_until="domcontentloaded")
            except Exception:
                return False
            if not ensure_content_list(page, timeout_ms=18000):
                continue

        reset_content_scroll(page)
        time.sleep(min(delay_s, 1.0))
        if open_claim_modal_for_item(page, item, delay_s=delay_s):
            return True

        if attempt >= max_attempts:
            break

        if studio_url:
            try:
                page.goto(studio_url, wait_until="domcontentloaded")
            except Exception:
                pass
        else:
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass
        time.sleep(min(delay_s, 1.2))

    return False
