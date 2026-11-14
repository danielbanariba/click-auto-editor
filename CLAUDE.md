# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated pipeline for creating full-album music videos with VHS aesthetics and uploading them to YouTube. Runs on Linux (Arch/CachyOS). Each album folder (audio files + cover image) is rendered into a 4K video and uploaded with auto-generated descriptions, tags, and playlist assignments.

## Quick Start

```bash
pip install -r requirements.txt
pip install -r vhs_effect/requirements.txt   # OpenCV/SciPy for VHS effects
playwright install chromium                  # for YouTube Studio automation
python config.py                             # validate environment + create directories
```

### C++/CUDA renderer (recommended for GPU rendering)

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
# Binary: cpp/build/vhs_render
```

## Running the Pipeline

Scripts are numbered but **not all are sequential**. The active pipeline is:

```
1. verificacion_previa.py   → Pre-flight: checks if album is already on YouTube
2. ffmpeg_render.py         → Main render engine (VHS + CUDA/NVENC)
3. subir_video_API.py       → YouTube upload via Data API (scheduled batches)
10. inpunar_video.py        → Playwright: copyright dispute automation
11. apelacion.py            → Playwright: copyright appeal automation
12. mapear_playlists.py     → Maps videos to playlists (runs as systemd service)
13. filtrar_portadas.py     → NSFW cover art filter (ONNX model)
```

### Publishing service (Telegram approval)

`servicio_publicar.py` runs as a systemd user service
(`systemd/publicar-albumes.service`) and publishes one album at a time. It asks
before rendering: `buscar` picks the next album and screenshots its YouTube
search, which goes as one message; then one question with the album cover
(`portadas.elegir_portada`, the same image the render uses) and the buttons
"✅ Renderizar y subir" / "🔞 Censurar" / "🗑 Eliminar (ya está publicado)" (Censurar: see
below). Eliminar moves the
album's folder to the trash (`eliminar` records it in the published registry first) and
the service asks about the next one; nothing is rendered. The notice "🗑 Borré ..." carries
"↩️ Deshacer" for 24 h (see the trash below); it does not hold back the next question. Renderizar y subir
records the album in `aceptados` (`data/agente/servicio.json`), renders it, runs
the checks, uploads it privately with no other question and deletes the folder →
claim check → monetization → publish or schedule. If a check fails, nothing is
uploaded: the service asks with the video screenshot and the Subir / Rechazar /
Eliminar buttons, as it does for an album rendered before this flow. The next
album's question goes out once the upload ends. An accepted album still in its
source folder after a restart is rendered again without asking; one that is gone
(moved to Error404) is dropped from `aceptados`. `/aleatorio`, written in the chat
or picked from the bot's command menu (registered with `setMyCommands` at start),
toggles random choice and answers "🔀 Aleatorio activado" or "Aleatorio
desactivado"; while it is on, `buscar --aleatorio` picks at random among the three
source folders, still moving registered albums aside. It starts off and the
setting (`aleatorio`) survives restarts. `/crear_lista` (the menu does not take
hyphens; typing `/crear-lista` also works) turns list mode on: each question then shows
"➕ A la lista", "🔞 Censurar", "🗑 Eliminar (ya está publicado)" and "▶️ Renderizar la lista (N)".
➕ saves the album in `lista` without rendering it and the next question comes right away
(`buscar --excluir` skips the albums on the list). ▶️ moves the list into `aceptados`, so
they are rendered, checked and uploaded one after another with no question, and turns list
mode off. The album on screen does not join the list: it is asked again, as a normal
question, when the list ends (`despues_de_lista`). A failed check of a list album asks as
usual and the rest of the list goes on; a failed upload or a Reintentar render stops it.
While an album uploads the GPU is idle, so the next accepted album (from the list, or one
accepted after Censurar) is rendered meanwhile (`adelantar_render`, added on 2026-11-13: on
2026-11-12 renders took 2-8 min and uploads 2-10 min, one after the other). It is checked and
uploaded when that upload ends. Only one album is rendered ahead (the videos wait on the SSD),
never during a cover analysis; after a failed upload it waits in `Renderizando/` until the
question is answered. The render and the upload then log in to DeathGrind at the same time; on
2026-11-13 a second login did not end the first session.
With no albums left to add, one message comes with only ▶️. Writing `/crear_lista` again
only says how many albums the list holds, and a "Renderizar y subir" question that was
open comes back with the list buttons. Messages from anyone else, or any other
text, are ignored. Right after the upload, `subir` sets the custom thumbnail
(`thumbnails.set`, 50 quota units) to the frame from the middle of the video that `verificar`
saved (it takes it again if it is missing); otherwise YouTube picks a frame of the intro. A
thumbnail that fails never undoes the upload: the upload notice says why and gives the Studio
link to set it by hand. Videos uploaded before this keep the intro: their frame and folder are
gone. It needs `PUBLICAR_TELEGRAM_TOKEN` and
`PUBLICAR_TELEGRAM_CHAT_ID` in `.env` (its own bot: Hermes keeps
`@banar_precios_bot`). Open questions survive restarts in
`data/agente/servicio.json`.

Albums come from `01_limpieza_de_impurezas`, then `Album/`, then `EP/`
(`DIRS_ORIGEN_RENDER` in `config.py`), newest year first. The render moves
each one to the SSD and the folder is deleted after publishing. When
DeathGrind does not know the release type, an `[EP]` suffix in the folder
name makes the title "Full EP".

`data/agente/ya_publicados.tsv` is the registry of albums that are already
published (`registro_publicados.py`). Each line holds a key, the name, the
origin (`canal`, `eliminar` or `subir`), the video id and the date. `eliminar`
records the album before moving the folder to the trash; `subir` records it after the
upload got a video id. `sembrar_publicados` loads the channel titles and the
albums the service already deleted or uploaded (not one that "↩️ Deshacer" put back after
its last deletion: it reads the `servicio_restaurar_*.out` logs too), and it can run again without
duplicating rows. The key is "band|album": no case, accents, signs, year, type
tag or `__N` suffix. A censored title ("C**") matches any letter of the same
length. `buscar` (and `renderizar` without `--carpeta`) moves every registered
album it finds on its way to `Ya publicados/` (under `BASE_DIR`, with `__N` when
the name is taken) and returns (or renders) the next one. A move that fails leaves
that album in place, unrendered. With `--carpeta` the registry is not checked. The service lists by
Telegram what was moved (up to 10 names and how many more) and, once per album,
what could not be moved. It is a name match, not proof: a reissue with the same
name is moved too, which is why nothing is deleted.

`eliminar` never deletes: added on 2026-11-11, after two albums that were not on YouTube were
deleted by a mistaken press (the drive is NTFS and had no trash). It renames the album into
`Papelera/<date>-<id>/` next to its root, on the same disk (`BASE_DIR/Papelera` for the source
folders and `upload_video/`, `/mnt/Render/Papelera` for `Renderizando/`), with an `origen.json`
that says where it came from and when. "↩️ Deshacer" runs `restaurar --papelera <entry>`: it
first removes from the registry only the row `eliminar` added (same name, origin `eliminar`; a
`canal` or `subir` row stays), then renames the album back. It never overwrites a folder that is
already in its place (`ya_existe`), and an entry already emptied answers `ya_no_esta`.
`restaurar` and `buscar` share the `origenes` lock, so a `buscar` running at that moment never
sees the album back with the old row (it would move it to `Ya publicados/`). The service runs
`vaciar_papelera` once an hour (the first, one hour after it starts): entries older than 24 h
(`PLAZO_PAPELERA_S`) are deleted, and their notices lose the Deshacer button. A trash that
cannot be emptied is reported once.

"🔞 Censurar" (added on 2026-11-13, after YouTube warned the channel about an explicit painted
cover) moves the album to `BASE_DIR/Censura/` (`a_censura`, `DIR_CENSURA`) and the next question
comes right away: the analysis runs apart and holds back nothing. `censurar --intento N` asks the
local Qwen 3.6 (`MODELO_CENSURA`, ollama at `OLLAMA_HOST` or `localhost:11434`) two questions about
each piece of the cover: where a penis, testicles, a vulva, an anus or a female nipple are, painted
or drawn ones too; and where there is gore: blood, open wounds, exposed organs or entrails, severed
body parts and corpses, painted, drawn or photographed (gore added on 2026-11-13: "tambien aplica
para imagenes gore o cadaveres reales, no solo sexuales"; the user accepts that many covers end up
almost all black). The two go apart because one question for both found fewer sexual zones on
2026-11-13. The question lists what was found in Spanish (`etiqueta_legible`: "pezón", "sangre",
"vísceras", "cadáver"...). `OLLAMA_HOST` is read as the ollama CLI reads it: with no scheme and no
port it goes to port 11434.
- Attempt 1 sends the whole cover, attempt 2 four overlapping 2×2 pieces (60 % of the side each),
  attempt 3 and later nine 3×3 pieces plus the whole cover. NudeNet found nothing on the two
  painted covers tried on 2026-11-13; Qwen found one of them only in the 2×2 pieces.
- Each box grows by half its width and height on every side (`MARGEN_CENSURA`), boxes that touch
  are merged, and they are filled solid black: never blur or pixelation, which a classifier sees
  through. A new attempt adds its boxes to the earlier ones, so the censorship only grows.
- The candidate and its boxes live in `data/agente/censura/`: nothing changes inside the album
  folder until Aceptar. An analysis stops after 25 min (`PRESUPUESTO_ANALISIS_S`).
- A piece where either reply cannot be read does not drop the other boxes, but it is counted:
  `censurar` returns `pedazos` and `sin_leer`, and if no reply can be read the attempt fails. A
  reply with items but no usable box counts as unreadable, not as "nothing to cover"; a bare
  object (no list around it) counts as one box. A reply whose JSON is cut off, has a stray comma
  or is followed by a second value is unreadable too: no box is taken from inside it.

When it ends, the original comes as a photo and then the censored one, with "✅ Aceptar",
"🔁 Volver a analizar" (the next attempt) and "🗑 Eliminar". With nothing found, the question
says so and Aceptar keeps the cover as it is. With pieces that could not be read, the question adds
"⚠️ N de M pedazos no se pudieron leer: puede quedar algo sin tapar", and with nothing found it says
"no encontré nada que tapar en lo que pude leer" instead. A failed analysis asks with the reason,
and there "🔁 Volver a analizar" repeats the same attempt. When an earlier attempt of that album
did succeed, and when Aceptar itself fails (a folder with that name in the origin, a rename that
fails), the question also carries "✅ Aceptar" (`censura_reintentar`): it applies the last
censorship sent, without running the model again. Aceptar (`aceptar_censura`) writes the censored
cover as `cover.png`, so `elegir_portada` picks it for the render, the background and the
thumbnail; the old cover goes to `originales/` under `data/agente/censura/` and is never deleted.
`aplicada.json` keeps the sha256 of the candidate it applied: a second Aceptar after a failed move
does not apply that one again, but it does apply a newer candidate from a later analysis, and the
list of saved originals keeps growing.
The album then goes back to the folder it came from and joins `aceptados`, as with "Renderizar y
subir", or `lista` if list mode is on at that moment. Eliminar sends it to the trash, and
"↩️ Deshacer" puts it back where it was before Censurar, so it is asked again. None of these
questions holds back the queue, and the steps (`censura` in `servicio.json`) are taken up again
after a restart. If the restart comes after `eliminar` moved the album but before the service read
the result, the service finds its trash entry (`entrada_de_papelera`) and still sends the notice
with "↩️ Deshacer".

The model takes about 17 GB of VRAM (an ollama model made a render fail on 2026-11-03), so an
analysis and a render never run together: `renderizar` returns `analisis_en_curso` and `censurar`
returns `render_en_curso`. The service runs one analysis at a time, never while another session's
`censurar` runs (`analisis_en_curso`), and waits only for the render in progress, not for the whole
list; the next render waits for the analysis. The model is
unloaded after each analysis, and if it stays loaded the service says so.

```bash
journalctl --user -u publicar-albumes -f         # what the service is doing
systemctl --user restart publicar-albumes        # after changing the code
```

Each step is a `pipeline_agente.py` subcommand that prints one JSON line;
logs, screenshots and state go to `data/agente/` (gitignored). A video id that starts with "-"
(YouTube gives some) goes as `--video-id=<id>`: written apart, argparse reads it as another
option and the step ends with "expected one argument". The service does it by itself.

```bash
env/bin/python pipeline_agente.py buscar [--aleatorio] [--excluir NAME]...  # next album (or a random one) (cola, Album/, EP/), skipping the excluded ones, without rendering: its folder, cover and YouTube search screenshot; already published ones → Ya publicados/
env/bin/python pipeline_agente.py renderizar             # next album (cola, Album/, EP/) → Renderizando/; already published ones → Ya publicados/
env/bin/python pipeline_agente.py verificar --carpeta "<dir>"  # audio/duration/4K checks, screenshot + YouTube search screenshot
env/bin/python pipeline_agente.py subir --carpeta "<dir>"      # private upload, mid-video frame as thumbnail, then deletes the album folder
env/bin/python pipeline_agente.py eliminar --carpeta "<dir>"   # moves the album folder to the trash of its disk without uploading (already published); a direct child of Renderizando/, upload_video/ or a source folder
env/bin/python pipeline_agente.py restaurar --papelera "<entry>"  # puts back an album that eliminar moved to the trash and removes its eliminar row from the registry
env/bin/python pipeline_agente.py vaciar_papelera  # deletes the trash entries older than 24 h (the service runs it every hour)
env/bin/python pipeline_agente.py a_censura --carpeta "<dir>"  # 🔞 Censurar: moves the album (a direct child of a source folder) to Censura/
env/bin/python pipeline_agente.py censurar --carpeta "<Censura/dir>" --intento N  # asks the local Qwen what to cover and writes the censored candidate outside the album (never during a render)
env/bin/python pipeline_agente.py aceptar_censura --carpeta "<Censura/dir>"  # makes the candidate the album cover, keeps the original and moves the album back to where it was
env/bin/python pipeline_agente.py programar --video-id ID --esperar  # claim check → monetization on → public now / first free moment, or +2 years if claimed
env/bin/python pipeline_agente.py eliminar_video --video-id ID [--retirado]  # deletes the video from YouTube (irreversible; the service only runs it from a Telegram button); --retirado skips the existence check for a video YouTube removed as a copy
env/bin/python pipeline_agente.py impugnar --video-id ID [--sin-enviar]  # disputes every claim of the video in Studio (a legal statement in the user's name); --sin-enviar fills the first form and closes it without sending
env/bin/python pipeline_agente.py impugnar_con_ventana --video-id ID [--video-id ID]... [--sin-enviar]  # the same for several videos, in one visible Brave window that waits for the user to pass "Verifica que eres tú"
env/bin/python pipeline_agente.py apelar [--sin-enviar] [--limite N]  # appeals age restrictions and limited ads in Hebrew (in the user's name); --sin-enviar fills one form of each kind and closes it
env/bin/python pipeline_agente.py seguir_impugnaciones  # reads /claims of the disputed videos and reports what changed (read-only)
env/bin/python pipeline_agente.py mapear_canal  # reads /claims of every video parked by a claim and sorts them; lists the private videos with no date (read-only)
env/bin/python pipeline_agente.py estado [--esperar-render]  # unscheduled uploads, rendered albums awaiting approval, render running
env/bin/python pipeline_agente.py sembrar_publicados  # loads the channel titles (1 quota unit per 50 videos) and the albums the service deleted or uploaded into the published registry
```

Scheduling: the videos stay at least one hour apart. An unclaimed video is
made public right away when no other video was published or is scheduled
less than one hour from now (published times come from the newest page of the
channel's uploads, scheduled ones from the API and `programados.jsonl`);
otherwise it is scheduled for the earliest moment that keeps one hour from
every published or scheduled video. Claimed videos go two years ahead, on the
first free hour, and monetization is not attempted for them.
`programar --reclamo no` keeps the date it already gave the video as an unclaimed one
(`fecha_ya_programada`) and says "ya estaba programado para ..., no lo muevo".
- That date is still in the future, and the video's last `programados.jsonl` row has no claim and
  the same date YouTube has. On 2026-11-14 that held for 38 of the 39 parked videos; the other one
  had been moved since.
- This was added on 2026-11-14: `mapear_canal` and `seguir_impugnaciones` freed the same video,
  and the second run moved it from 11-14 13:00 to 11-16 21:00, because its own date counted as
  taken.
- A video parked by a claim is not kept, even when its date is less than a year away: its last row
  has the claim. It still goes into the queue.
- Monetization is still attempted, since the run that set the date may have failed at it.
- The date is read again inside the `programar` lock, so a run that waited for the lock does not
  move a date another run has just set.

Studio (claim check and monetization) runs Brave headless with a copy of the
signed-in profile (`.runtime_browser_profiles/agente`). It must launch without
Playwright's `--password-store=basic` / `--use-mock-keychain` flags (so Brave
can read the cookies from KWallet) and with a normal Chrome user agent; see
`abrir_contexto_studio`. The account has several channels, and YouTube sometimes
opens "Selecciona un canal" (`www.youtube.com/signin_prompt`) instead of Studio:
`pagina_de_studio` opens Studio first and, when that question shows, clicks the
exact, visible "Daniel Banariba" (`CANAL_DE_STUDIO`; "Daniel Banariba Drummer" is
another channel). If it is not listed, the step fails with `canal_no_disponible`.
Claim detection reads `/video/<id>/claims`; an unclear
page returns `reclamo_desconocido` and the service asks the user instead of
guessing, and a missing session returns `sin_sesion_en_studio`. The one exception
is `detalle = error_navegador` (the browser failed, so the page was never read):
the service counts it like any other `programar` failure (`fallos_programar`), says
"No pude revisar los reclamos de ... (falló el navegador). Reintento en 30 min." and
runs the same command again, and asks the question only after `INTENTOS_PROGRAMAR` (3)
failures in a row; that video's count then starts again, as it does after a success.

The claim rows carry no claimant name. `extract_claimants_from_video`
(`11. apelacion.py`) waits up to 30 s for the "Ver detalles" buttons, opens each
one (up to 20 claims), waits for the visible "Reclamantes" header and reads the
block under it. Studio keeps every closed dialog in the DOM, hidden, and appends a
new one per click, so only the visible header counts; a navigation during the wait
(`/copyright` redirects to `/claims`) is treated as "not ready yet". It is read-only (it never clicks "Tomar medidas") and
a browser error leaves that claim out instead of stopping the scheduling.

`lista_negra_reclamantes.txt` (project root, one label per line, `#` lines are
ignored) is the claimant blacklist. For a claimed video, `programar` checks each
claimant line against it, case-insensitively and by containment ("umg" matches
"Believe Music, UMG"), and returns the matches in `reclamantes_en_lista_negra`.
The match never changes the schedule: the video still goes two years ahead. A
missing or unreadable file means no matches.

When `programar` returns matches, the service sends one Telegram question for
that video with its title, the matching claimants, the `/video/<id>/claims` link
and the Studio screenshot (`captura_studio`). "Conservar" only closes the
question. "🗑️ Eliminar video" runs `pipeline_agente.py eliminar_video`, which
checks the video exists, calls `videos.delete` through `ejecutar_api` and prints
one JSON line; the service reports the result. A video that is already gone
(`no_existe`) is reported as already deleted, not as a failure. Deleting cannot be undone, so
nothing deletes a video without that button press. The question is persisted in
`data/agente/servicio.json` like the others, so it survives restarts.

YouTube sometimes removes a fresh upload because it is a copy of another video that was
taken down for copyright (seen on 2026-11-04: the band itself had filed the takedown). The
API then does not list the video, not even to its owner, so `programar` sees `no_existe`.
Only then it reads Studio, inside the `programar` lock (`leer_retiro_en_studio`): it opens
`/video/<id>/edit` and clicks only the "Retirado" notice and the "Revisar problemas" button
of its tooltip. That tooltip says "...es un duplicado de otro que se ha retirado...". The dialog
says who filed the takedown ("De:"), for which content, and when. Anything else, or a page that
cannot be read, keeps the old `no_existe` notice. Once the tooltip has confirmed the copy, a
screenshot that fails (browser or disk error) does not lose it: the result is still
`copia_retirada`, with `captura_studio` None, and the question goes without the photo. For a copy, `programar` returns
`motivo = copia_retirada`, removes the video from `pendientes.json` (it can never be published)
and adds its album to the published registry if it is not there yet. `album_registrado` says
whether there was an album to add: a video with no entry in `pendientes.json` (one released from
a claim) has none. The service sends one
Telegram question with only "🗑️ Eliminar video": "No se puede publicar porque es copia de
otro video que YouTube retiró por derechos de autor", the title, those lines, the Studio link and
the screenshot. Only when `album_registrado` is true does it add that the album is in the
registry and is not rendered or uploaded again. The button runs `eliminar_video --retirado`,
which calls `videos.delete` without
the `videos.list` check (it would always answer `no_existe`). The question keeps the title and
the notice that the video was deleted names it, because `--retirado` returns the id as title (a
question saved before this keeps the id). If YouTube refuses, the failure
notice gives the Studio link to delete it by hand. A 404 from `videos.delete` returns
`motivo = no_encontrado`: the notice says YouTube does not find the video (it may have been
deleted already) and gives the Studio link, instead of "Sigue en YouTube", which any other API
error keeps.

A claimed video with no blacklist match gets a different Telegram question: its
title, the claimants, the `/claims` link, the Studio screenshot and a note that
the dispute goes out with the reason "Licencia" and the configured text and
signature. "⚖️ Impugnar" runs `pipeline_agente.py impugnar --video-id <id>` and the
service reports "Impugné N de M reclamos" with the failures and the `/claims`
link; "Conservar" only closes the question; "🗑️ Eliminar video" runs
`pipeline_agente.py eliminar_video`, as in the blacklist question. Nothing is disputed or
deleted without that press, and a blacklisted video keeps only the delete question.

Nothing appeals a claim. The user's decision is: dispute yes, appeal never. Appealing a rejected
dispute risks a copyright strike, and the user has had bad experiences with it.
- After "Tomar medidas", `impugnar` goes on only when the menu offers an option named "Impugnar"
  (`P_DISPUTE_MENU_OPTION` in `10. inpunar_video.py`). An "Apelar" option never matches, so that
  claim fails without anything being sent.
- The dispute form itself says "puedes apelar la decisión" (seen in `data/impugnar_evidencia/`).
  So a text check for "apel" would also stop real disputes.
- The service asks "Conservar" or "🗑️ Eliminar video" about a rejected dispute
  (`impugnacion_rechazada`).
- The scheduling notice of a claimed video says "programado para ..., hasta que se resuelva el
  reclamo".
- On 2026-11-10, PR #63 removed disputing after a misreading of "ya tuve malas experiencia con
  apelar". It came back on 2026-11-11.

The daily `apelar` step is a separate thing: it appeals age restrictions and limited ads, not
claims.

When a dispute fails with `verificacion_de_identidad`, the notice carries "🖥 Impugnar con
ventana" and how to use it (be at the PC, pass the verification in the window, touch nothing
else). The press runs `impugnar_con_ventana` for that video, and the service reports the
result in one message. If the verification was not passed in time, the report offers the window
again for what was left. A window run that ends with no result (`sin_respuesta`) may have
sent something, so the notice lists the `/claims` links to check and offers nothing.
`seguir_impugnaciones` also reads the `servicio_impugnar_con_ventana_*.out` logs. The notice of
a single dispute that asks for the verification keeps the step's `Log:` line, like the other
failure notices. The press drops the videos that are already in the running «Impugnar todos»
batch or in another running window (if none is left, one line says they are already being
disputed and no window opens). The window's videos stay in `ventanas_en_curso`
(`data/agente/servicio.json`) until it ends. If the service restarts meanwhile, the window died
with it: the service sends one notice with each video's `/claims` link and does not launch it
again, since it may have sent a dispute.

Disputing (`impugnar`) reuses the recipe of `10. inpunar_video.py`: for each claim
whose "Tomar medidas" is visible and enabled it opens the form, picks "Licencia",
ticks the permission and the three declarations, fills `mensaje` and `firma`, and
presses "Enviar". It opens `/claims` directly (Studio redirects `/copyright` there from the browser, and
an `evaluate` caught in that navigation fails with "Execution context was destroyed";
the count also retries when that still happens). It runs inside the `programar` lock (same Studio profile copy)
and never waits for a 2FA (the service is headless): a verification page (also the
small "Verifica que eres tú" dialog Studio stacks over the form a few seconds after it opens:
it is looked for in the whole page, and for `VERIFY_DIALOG_DELAY_S` (6 s) more once the form
shows), a missing form or any unexpected page fails that claim without sending and dumps evidence to
`data/impugnar_evidencia/`. It never disputes more claims than there were
disputable claims at the start. A dispute counts as done only when `/claims`,
reloaded, shows exactly one enabled "Tomar medidas" fewer. Studio sometimes leaves
the form open after "Enviar" (`form_persistente`), with no error, and registers
nothing. Only in that case, and only when `/claims` still shows the same count, the
step waits `PAUSA_REINTENTO_IMPUGNAR_S` (30 s) and looks again: one fewer counts as
done (Studio was just slow), and the claim is sent once more only if both the
disputable count and the number of claim rows are unchanged (a new claim would
otherwise be disputed in its place). With the retry form ready, it checks `/claims`
once more and does not press "Enviar" if anything changed (that tab may have opened
the next claim). A form that closed is never resent. If the retry
is not registered either, the list cannot be read (no proof that nothing went out)
or the count is anything else, the claim is `no_confirmado` and the run stops. Any
failure stops the run, and the
JSON reports `reclamos`, `impugnados`, `fallos` and a `captura` of `/claims`. A
video with no disputable claims returns `motivo = sin_reclamos_impugnables`.

`seguir_impugnaciones` follows the disputes once a day; the service launches it right after
`apelar`, unless `apelar` failed with `sin_sesion_en_studio` (the follow-up would fail the same
way and repeat the notice). It takes the videos from the `servicio_impugnar_*.out` logs with at least one claim
disputed, opens their `/claims` and only reads it (it never presses anything). Each claim is
a `ytcr-video-content-list-row`: its name is the song and the artist, and its status is the
text of `#impact-container` without its buttons. The first read of a video is the baseline,
kept in `data/agente/impugnaciones.json`. After that, a status that changed (the day count
of "Caduca dentro de N días" does not count, nor does "1 día" instead of "N días") or a claim
that is no longer listed is a change: it is written there and reported once in one Telegram message, with the old and new text and
the `/claims` link. A page with neither the list nor Studio's "no claims" notice, or with
rows that do not match the "Ver detalles" buttons, is a failure and changes nothing. The list
draws in parts and Studio may show the "no claims" notice for a moment first, so a read
with a claim missing, or with none, counts only when a re-read of the same page
`PAUSA_RELECTURA_RECLAMOS_S` (10 s) later sees the same, and 30 s later when the list is empty
(an empty list frees the video, and the service then publishes it by itself); two re-reads that
keep changing are a failure. A video with no claim "en proceso" is no longer followed, unless
the service disputes another of its claims later (a `servicio_impugnar_*.out` log written after
its last read): then it is followed again. The run holds the `programar` lock, looks
first at the videos checked longest ago and stops after 30 min (`TIEMPO_MAXIMO_SEGUIMIENTO_S`).
A video whose read fails (a browser error, or a `/claims` that cannot be read) gets its check
time updated, so it goes to the end of the next run and does not hold back the rest; if the
browser or the session is not available, which fails every video, none is marked.

When a video stops being followed, the run reports it once. If `/claims` shows no claims, it
goes in `liberados` and the service runs `programar --video-id <id> --reclamo no` without
asking: monetization on and published now or at the first free moment, like an unclaimed
video (the title comes from `programados.jsonl`, since the video is no longer pending). If
it still has a claim (the dispute was rejected, or another claim remains), it goes in
`rechazados` with each claim's status and a `/claims` screenshot, and the service asks
"Conservar" or "🗑️ Eliminar video"; Eliminar runs `eliminar_video` with its usual check. The
screenshot is taken before the video is saved as finished: if it fails with a disk or browser
error, the video is not closed and its question comes in the next run.

`/mapear_canal` (also written `/mapear-canal`; in the bot's menu) runs `mapear_canal` once.
The step takes the parked videos from the API: the private ones scheduled more than a year
ahead among the last `ESCANEO_PROGRAMADOS` uploads (claimed videos go two years ahead, whoever
scheduled them). It reads each `/claims` under the `programar` lock and presses nothing.
- No claims, in two reads 30 s apart: the video goes in `liberados`, and the service runs
  `programar --reclamo no` (unless one is already running for that video).
- An enabled "Tomar medidas": the step reads the claimants. If they cannot be read, the video
  goes in `fallos`: without them the blacklist cannot be checked. A blacklist match goes in
  `lista_negra`, and the service asks the usual blacklist question unless one is already open
  for that video. Anything else goes in `sin_impugnar`.
- Claims, none disputable: `en_proceso`. A page that cannot be read: `fallos`. A browser error
  stops the run, and the video being read goes to the end of the next run (if the browser did
  not even open, nothing was read and no video is marked).

The run stops after 30 min (`TIEMPO_MAXIMO_MAPEO_S`); the rest comes back in `sin_revisar`.
`data/agente/mapeo_canal.json` keeps when each video was last read, so the next run starts
with the others. The service then sends one message: the counts, the videos that could not
be read, and every video left to dispute with its claimants and `/claims` link. If anything
is left to dispute, the message carries one button, "⚖️ Impugnar todos (N)". A message longer
than Telegram's 4096 characters is split, and the button goes on the last part.

Nothing is disputed without that press. The press stores the batch in `lote_impugnar`
(`data/agente/servicio.json`) and runs `impugnar` for one video at a time. A second press
adds to the running batch the videos that are not in it yet. At the end the service sends one
report: claims disputed per video, and failures with their `/claims` link. A
`verificacion_de_identidad`, `sin_sesion_en_studio` or `canal_no_disponible` stops the batch,
since the rest would fail the same way, and the report lists what was left. When it was
the verification, the report carries "🖥 Impugnar con ventana" for the video that asked and
the ones left: one window, one verification. After a restart,
the video in progress is not run again: it may have sent a dispute that Studio does not show
yet. The report marks it as interrupted, with its `/claims` link, and the batch goes on.

`mapear_canal` also returns `privados_sin_programar`: the private videos with no
`status.publishAt` among ALL the channel's uploads (not only the last `ESCANEO_PROGRAMADOS`)
whose id is not in `pendientes.json`, as `[{"video_id", "titulo", "subido"}]` (`subido` is
`snippet.publishedAt`). Such a video is never published and nothing waits for it. It comes from
the API only (`privados_sin_fecha`): Studio is never opened for these videos, and the parked-video
scan does not change. Reading every upload costs about 2 quota units per 50 uploads (one
`playlistItems.list` page and one `videos.list`), so a channel of 1,000 uploads spends about 40
units per run, on top of the parked-video scan. The list is in the result whenever `aparcados`
is, also when no video is parked (Studio is then not opened) or when Studio fails. This search
runs after the parked-video scan and on its own (each page and each `videos.list` through
`ejecutar_api`): if it fails, the parked videos are still reviewed, the list comes empty with
`motivo_privados`, and the summary says "❌ No pude buscar los privados sin fecha (...)". The service
adds one line to the summary, "🔒 Privados sin fecha de publicación (te pregunto aparte por cada
uno): N", and then one question per video (kind `privado_sin_programar`): title, upload date,
`/video/<id>/edit` link, "▶️ Programar", "Conservar" and, on its own row, "🗑️ Eliminar video". A
video that already has an open question of that kind, or whose `programar` is running, is not
asked again. "▶️ Programar" runs
`programar --video-id ID --esperar`, the command an upload gets (60-minute margin, claim check,
monetization, publish or schedule); that step adds the video to `pendientes.json`, and its
notices name it by id, since the title is only known for videos this pipeline uploaded.
"🗑️ Eliminar video" runs `eliminar_video`. "Conservar" only closes the question, so the next
`/mapear_canal` asks again. Nothing is scheduled or deleted without a press.

`impugnar_con_ventana` disputes one or several videos, one after another, in one visible
Brave window (`pagina_de_studio(visible=True)`) on the user's desktop. Passing "Verifica que
eres tú" in the everyday Brave does not help: the service's profile copy does not get it. When
the dialog shows, each claim waits up to `ESPERA_VERIFICACION_VENTANA_S` (10 min) for the user
to pass it in that window; once passed, it holds for the next claims of the same window (seen
on 2026-11-07). A verification not passed in time, a missing session or channel, or a browser
error stops the run (`FRENAN_LA_VENTANA`); the videos left come back in `sin_revisar`. It prints
one JSON line with `videos`, one dispute result each.

Monetization (`studio_monetizacion.py`) turns ads on at
`/video/<id>/monetization`. After opening it, Studio navigates several times and ends
on `/monetization/ads`; a read of the ads state caught in one of those navigations
("Execution context was destroyed") counts as not loaded yet, and any other browser
error fails the step. Studio then asks for the ad-suitability
questionnaire; its answers cannot be changed afterwards. The user's decision
(`RESPUESTA_CUESTIONARIO`) is to always answer "Nada de lo anterior" (his videos
are art) and select no radios. The step sends it (Enviar, then Guardar, then
reloads to confirm ads are on) only if the side panel reads "Adecuado para
anuncios"; with anything else, or no checkbox, it cancels and the video is left
as it was. A failure after Enviar returns `motivo = cuestionario_ya_enviado`.
Any monetization failure never stops the video: the service sends a Telegram
notice with the `/monetization` link of that video.

Appeals (`apelar`, `studio_apelaciones.py`) run once a day, launched by the service
without asking (the user's decision: "Mandalo solo"); the time of the last run is
`ultima_apelacion` in `data/agente/servicio.json`. The step walks the Studio content
list filtered by age restriction (`AGE_RESTRICTION`) and then by limited ads
(`MONETIZATION` `LIMITED`), page by page, and sends a fixed text in modern Hebrew
(`TEXTO_APELACION_EDAD`, `TEXTO_COMENTARIOS_ANUNCIOS`; the age field takes 300
characters):

- Age: the "Avisos" tooltip cause "YouTube ha limitado la audiencia...", "Ver todas
  las opciones", and in "Apelar la decisión" the checkbox, "Tu explicación" and
  "Enviar apelación". "Entendido" (confirm the problem) is never pressed. An already
  resolved appeal ("Apelación rechazada") is recorded as `ya_apelado`. Some tooltips
  show only "Revisar y tomar medidas" instead of the cause: it opens the same problem
  dialog, but only a resolved appeal is recorded from it. An appealable form opened
  that way does not say which problem it is about, so nothing is sent: it is recorded
  as `revisar_a_mano` and the service lists it with its Studio link.
- Limited ads (rows without an age restriction): "Ver comentarios", "Enviar
  comentarios", and the Google feedback panel. The panel is an `about:blank` iframe
  inside the `google.com/tools/feedback` one; its box is the textarea labelled
  "Explica tus comentarios (obligatorio)". This is not a formal appeal. A tooltip with
  no "Ver comentarios" (no manual review) is recorded as `sin_opcion`, without sending.

`data/agente/apelaciones.json` keeps one entry per `<tipo>:<video_id>`. Each one is
written before the send button is pressed (`enviando`), so a cut or crash never sends
twice, and nothing is ever sent twice to a video in that file. A send whose form does
not close is `no_confirmado`. It sends at most `LIMITE_APELACIONES` (10) per run, age
first. It runs under the `programar` lock and stops after 3 failures in a row, except
a dialog that does not open (`dialogo_desconocido`), which is counted apart and stops it only
after 10 in a row (`MAX_SIN_ABRIR`): it is usually a passing Studio failure, the video is
not recorded and it is tried again the next day. Appealing a video well resets both counts.
The "Revisar un problema" dialog does not close with Escape: it is closed with its X and
then "Salir". The service reports what was sent, what was not confirmed and what failed
in one Telegram message, and stays silent when there was nothing to appeal. This report and the
follow-up's are saved in `avisos_pendientes` (`data/agente/servicio.json`) before they are sent
and leave it only when Telegram accepts them: a message Telegram refuses goes out again on the
next steps, and after `INTENTOS_AVISO` (5) refusals while Telegram answers it is dropped, so a
text Telegram never accepts cannot block the next ones. With Telegram down it waits and spends
no attempts.

The same daily walk follows up what was sent (`enviando`, `enviado`, `no_confirmado`
with no `resultado`), also after the send limit is reached. Following up has its own
cap of 15 min (`TIEMPO_MAXIMO_SEGUIMIENTOS_S`) inside the 45 min of the run
(`TIEMPO_MAXIMO_S`): once it is spent the follow-ups are skipped, the list is still walked,
and the sends keep the rest of the time. For age, it opens the
problem dialog and only reads it: a first line "Apelación rechazada" is `rechazada`.
Limited ads have no formal rejection. A followed-up video that is missing from a list
walked to its last page is first marked `ausente` (a date in its entry; `estado` and
`resultado` stay as they are, and `--sin-enviar` writes nothing). It is `aceptada` only if
the next run finds it missing again: a video also leaves the list when it is deleted or
made private, and the result is written once. If it shows up in the list again, the mark is
removed (an empty list proves nothing, so nothing is marked or accepted from it). The
result is written to the entry (`resultado`, `detalle_resultado`,
`fecha_resultado`) and reported once in the same Telegram message, with the Studio link.
A follow-up that cannot be read is listed apart ("No pude ver cómo va la apelación") and
does not count toward the 3 failures that stop the run. The follow-up opens the tooltip's
cause or "Revisar y tomar medidas", which only opens the problem dialog (not the "Tomar
medidas" of `/claims`), and presses nothing in it.

### Common commands

```bash
python "2. ffmpeg_render.py"                     # render all (parallel by default)
python "2. ffmpeg_render.py" --test              # render 1 folder as smoke test
python "2. ffmpeg_render.py" --test "/path/Band - Album"  # render one specific folder (moves it like a real run)

python "3. subir_video_API.py" --limite 1        # upload 1 video
python "3. subir_video_API.py" --todo --cantidad-lote 24 --gap-horas 1
python "3. subir_video_API.py" --modo-inmediato --limite 1

python "12. mapear_playlists.py" --limite 200
python "10. inpunar_video.py" --aprender         # record new selectors
```

### Tests

```bash
env/bin/python -m pytest tests/ subir_video/ -q  # pipeline, publishing service and helpers
pytest vhs_effect/test_vhs.py                    # all VHS unit tests
pytest vhs_effect/test_vhs.py -k color_processor # specific test
python vhs_effect/test_vhs.py                    # without pytest
```

## Architecture

### Render pipeline (`2. ffmpeg_render.py`, ~111 KB - largest script)

For each album folder in `DIR_AUDIO_SCRIPTS`:
1. Picks RAM or SSD for each album and copies it there. It uses the ramdisk (`/mnt/ramdisk_render`)
   only if what stays in RAM (the video without audio plus a 15% margin, so about 50 minutes of
   album in 20 GB) fits in the free ramdisk and in `MemAvailable` minus `RAM_MARGEN_LIBRE_GB`, so a
   render never pushes the system into swap (zram does not compress video). Otherwise it uses
   `SSD_TEMP_DIR`. If `MemAvailable` cannot be read, it uses the SSD
2. Reads audio metadata (mutagen), generates tracklist overlay with auto-sized fonts
3. Creates drop shadow for cover image (`effects/sombra.py`)
4. Fetches track titles from DeathGrind API or falls back to filenames
5. Runs profanity censorship (`limpieza/censura.py`)
6. Renders via C++/CUDA binary (`cpp/build/vhs_render`) if `USE_CPP_VHS=True`, else FFmpeg
7. Muxes audio straight onto the SSD and moves the finished album folder to `DIR_RENDERIZANDO` (`Renderizando/`); `3. subir_video_API.py` reads `DIR_UPLOAD`.
   An album rendered in RAM gets its final video written to `SSD_TEMP_DIR/_mux/`, on the same
   filesystem as `Renderizando/`, and it reaches the queue with a rename: the video is not copied
   from RAM at the end and `Renderizando/` never holds a half-written video (`estado`, `verificar` and `subir`
   read that folder). The video without audio and the audio mix are deleted as soon as the mux ends

### Upload pipeline (`3. subir_video_API.py`, ~109 KB)

- Authenticates via `subir_video/authenticate.py` (rotates through multiple credential sets in `credentials/`)
- Builds rich video descriptions with streaming links (Bandcamp, Spotify, etc.) and social links,
  taken from the band's DeathGrind data. It never asks `/posts/<id>/links`: those are download
  links (Mega, Mediafire...), and each request may spend the account's download credits, which
  the scraper needs
- Schedules uploads in batches at configurable time slots
- Creates/manages playlists per band and genre
- Handles quota exceeded errors by rotating credentials

### Credential rotation (`subir_video/authenticate.py`)

Multiple OAuth credential pairs in `credentials/`:
- `client_secrets_{prefix}_{n}.json` / `token_{prefix}_{n}.json`
- Prefixes: `upload` and `playlists`, several sets each
- Auto-rotates on quota exhaustion via `authenticate_next(prefix)`

### Configuration (`config.py`)

Central config for all scripts. Key settings:
- `BASE_DIR`: root data directory (external drive at `/mnt/Entretenimiento/01_edicion_automatizada`)
- `RENDER_SSD_DIR`: SSD (`/mnt/Render`, ext4) that holds `Renderizando/`; rendered albums wait there for upload, and nothing renders while it is unmounted
- `SSD_TEMP_DIR`: `/mnt/Render/temp_render`, the render's temp folder for albums that do not fit in
  RAM (a 100-minute album made a 31 GB video, and in the SSD the mux writes a second copy next to the
  video without audio). It also holds `_mux/`, where an album rendered in RAM gets its final video
  before the rename into `Renderizando/`. The render empties only the ramdisk (and `_mux/`) when it
  exits, and a killed render cleans nothing, so `renderizar` deletes everything in `SSD_TEMP_DIR`
  before and after each render, except `_rescate/` (finished videos that could not be moved to
  `Renderizando/`). It only runs when no other render is running.
- `RAMDISK_SIZE_GB` / `RAM_MARGEN_LIBRE_GB`: size of the tmpfs (20 GB) and the memory the render leaves
  free beyond the album (8 GB). The album goes to RAM only if its estimate fits in the free ramdisk
  and in `MemAvailable` minus that margin. The estimate counts the video without audio (nominal
  45 Mbps, +15%); the CQ 20 output measured about 41 Mbps on two albums. With several renders at once
  (`MAX_PARALLEL_RENDERS` above 1) each one decides on its own `MemAvailable` reading
- `USE_CPP_VHS`: enable C++/CUDA VHS renderer (default `True`)
- `USE_GPU` / `USE_RAMDISK`: GPU and ramdisk flags
- `ALLOW_FFMPEG_FALLBACK`: fall back to FFmpeg if CUDA fails
- `MAX_PARALLEL_RENDERS`: concurrent render count (currently 1 for 4K)
- `VIDEO_WIDTH`/`VIDEO_HEIGHT`: 3840x2160 (4K)

### Key modules

| Module | Purpose |
|--------|---------|
| `vhs_effect/` | Self-contained Python VHS effect library with CPU/GPU paths |
| `cpp/` | C++/CUDA VHS renderer (targets SM 8.6 / RTX 3090 Ti) |
| `effects/sombra.py` | Drop shadow generation for album covers |
| `limpieza/censura.py` | Profanity filter with leetspeak handling |
| `limpieza/` (other) | Folder cleanup utilities (normalize names, remove junk files) |
| `selectores/` | JSON selector definitions for Playwright automation |
| `analisis_canal/` | Separate 3-step sub-pipeline for YouTube channel analytics |
| `content/` | Intro video (`0000000000000000.mp4`) and VHS noise overlay |

## Code Style

- **Language**: logs, prompts, and commit messages in **Spanish**
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- **Paths**: always use `pathlib.Path`, never `os.path.join()` or string concatenation
- **Imports**: stdlib → third-party → local (see AGENTS.md for full example)
- **Subprocesses**: `subprocess.run([...], check=True)`, avoid `shell=True`
- **Files**: always specify `encoding="utf-8"`
- **Errors**: catch specific exceptions, never bare `except Exception:`
- **Commits**: Spanish, Gitmoji + Conventional Commits, following `templates/commit-template.git.txt`
- **Pull requests**: body follows `.github/pull_request_template.md`

## Persistent State Files

- `mapear_playlists_checkpoint.txt`: tracks which videos have been mapped to playlists
- `data/playlist_links_cache.json`: cached playlist IDs per band/genre
- `Ya subidos/` (under `BASE_DIR`): album folders moved there after a successful upload
- `data/agente/ya_publicados.tsv`: registry of already published albums; `renderizar` moves the matching ones to `Ya publicados/` (under `BASE_DIR`) instead of rendering them
- `generos_activos.txt`: tab-separated list of active music genres with IDs

## Credentials and Secrets (never commit)

- `credentials/`: OAuth client secrets and tokens (gitignored)
- `.env`: `DEATHGRIND_EMAIL`, `DEATHGRIND_PASSWORD`
- `client_secrets.json`, `token.json`, `PASS.py`, `.playwright_profile`

## Known Friction

- DeathGrind archives often carry a DeathGrind Club promo image next to the real cover:
  `DGC.jpg`, `DGC - PURE HATRED.jpg`, `DGC Pure Hatred.jpeg` and similar names.
  - `portadas.elegir_portada` decides the cover for both the Telegram question and the render.
    Images whose name starts with "DGC" never count. It takes `cover.*` first, then an image whose
    name has a word like front, cover or portada, then any other image, by name.
  - Images whose name says they are something else go last: back, booklet, logo, scan,
    spectrogram, small, AlbumArtSmall (the Windows Media thumbnail)... A name with "cd" or "disc"
    goes last only when it has no cover word, because FLAC release names carry "cd"
    (`...-cd-flac-2016-front.jpg`). A shadow counts only when nothing else is there.
  - On 2026-11-10, 145 albums in the queue had no `cover.*` and more than one image. Picking by
    name alone took the back cover in 18 of them and the thumbnail in 29.
  - With only promos, there is no cover, and the render takes the one embedded in the audio.
  - The render used to take whichever image the filesystem listed first. On 2026-11-09 that put
    the promo on Marcio Dentro's video, while the question had shown the real cover.
- Render is 4K and heavy; `MAX_PARALLEL_RENDERS=1` is the safe default
- Ramdisk requires `sudo` without password for mount (configured via sudoers)
- CUDA fallback chain: C++/CUDA → FFmpeg NVENC → FFmpeg libx264
- A CUDA "out of memory" means another program holds the VRAM (an ollama model took
  16 GB on 2026-11-03): it is not the album's fault, so the render stops without moving
  the album to Error404 and the service asks "Reintentar" before rendering another one
- Playwright sleeps are calibrated; reducing them breaks YouTube Studio automation
- A network outage reaches the API calls as a plain `OSError` ("Network is unreachable", "No route to host",
  "Network is down": `ENETUNREACH`, `EHOSTUNREACH`, `ENETDOWN`), not as a `ConnectionError`. `es_error_de_conexion`
  (`pipeline_agente.py`) counts it as a connection failure like any other: `ejecutar_api` retries it
  `REINTENTOS_CONEXION` times and the steps then return `motivo = sin_conexion`. Any other `OSError` (missing
  file, permissions) is neither retried nor turned into `sin_conexion`
- `12. mapear_playlists.py` runs as a systemd service (`systemd/mapear_playlists.service`, installed as a copy in `/etc/systemd/system/`) with auto-restart
