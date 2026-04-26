import shutil
import sqlite3
from pathlib import Path


MIN_GOOGLE_COOKIE_COUNT = 10

KNOWN_BROWSER_PROFILES = (
    {
        "name": "brave",
        "user_data_dir": Path.home() / ".config" / "BraveSoftware" / "Brave-Browser",
        "executables": ("brave", "brave-browser"),
    },
    {
        "name": "chromium",
        "user_data_dir": Path.home() / ".config" / "chromium",
        "executables": ("chromium", "chromium-browser"),
    },
    {
        "name": "chromium-pw",
        "user_data_dir": Path.home() / ".config" / "chromium-pw",
        "executables": ("chromium", "chromium-browser"),
    },
)

ROOT_ENTRIES_TO_COPY = (
    "Local State",
    "First Run",
    "Last Version",
    "Variations",
    "NativeMessagingHosts",
)

IGNORE_NAMES = {
    "cache",
    "code cache",
    "gpucache",
    "graphitedawncache",
    "grshadercache",
    "shadercache",
    "dawngraphitecache",
    "dawnwebgpucache",
    "component_crx_cache",
    "extensions_crx_cache",
    "crash reports",
    "singletonlock",
    "singletoncookie",
    "singletonsocket",
    "lock",
    ".org.chromium.chromium.0ko3wt",
}


def find_executable(executables):
    for candidate in executables:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def browser_candidate_for_path(user_data_dir):
    if not user_data_dir:
        return None
    try:
        resolved = Path(user_data_dir).expanduser().resolve()
    except Exception:
        resolved = Path(user_data_dir).expanduser()

    for candidate in KNOWN_BROWSER_PROFILES:
        try:
            known = candidate["user_data_dir"].resolve()
        except Exception:
            known = candidate["user_data_dir"]
        if resolved == known:
            return {
                "name": candidate["name"],
                "user_data_dir": resolved,
                "executables": candidate["executables"],
                "executable_path": find_executable(candidate["executables"]),
            }

    label = resolved.name.lower()
    if "brave" in str(resolved).lower():
        executables = ("brave", "brave-browser")
        name = "brave"
    else:
        executables = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
        name = label or "manual"

    return {
        "name": name,
        "user_data_dir": resolved,
        "executables": executables,
        "executable_path": find_executable(executables),
    }


def describe_profile_source(user_data_dir, profile_name="Default", name=None):
    info = browser_candidate_for_path(user_data_dir)
    if info is None:
        return None
    if name:
        info["name"] = name
    info["cookie_count"] = google_cookie_count(info["user_data_dir"], profile_name=profile_name)
    return info


def google_cookie_count(user_data_dir, profile_name="Default"):
    cookies_path = Path(user_data_dir) / profile_name / "Cookies"
    if not cookies_path.exists():
        return 0

    try:
        conn = sqlite3.connect(f"file:{cookies_path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return 0

    try:
        row = conn.execute(
            """
            select count(*)
            from cookies
            where host_key like '%youtube%'
               or host_key like '%google%'
            """
        ).fetchone()
        return int((row or [0])[0] or 0)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def has_google_session(user_data_dir, profile_name="Default", min_count=MIN_GOOGLE_COOKIE_COUNT):
    return google_cookie_count(user_data_dir, profile_name=profile_name) >= min_count


def discover_signed_in_profile(profile_name="Default"):
    best = None
    for candidate in KNOWN_BROWSER_PROFILES:
        root = candidate["user_data_dir"]
        if not root.exists():
            continue
        count = google_cookie_count(root, profile_name=profile_name)
        info = {
            "name": candidate["name"],
            "user_data_dir": root,
            "executables": candidate["executables"],
            "executable_path": find_executable(candidate["executables"]),
            "cookie_count": count,
        }
        if best is None or info["cookie_count"] > best["cookie_count"]:
            best = info
    if best and best["cookie_count"] >= MIN_GOOGLE_COOKIE_COUNT:
        return best
    return None


def _ignore_browser_entries(directory, names):
    ignored = set()
    for name in names:
        lowered = name.lower()
        if lowered in IGNORE_NAMES:
            ignored.add(name)
            continue
        if lowered.startswith(".org.chromium.chromium."):
            ignored.add(name)
            continue
        if lowered.endswith(("cache", "cachestorage")):
            ignored.add(name)
            continue
    return ignored


def prepare_runtime_profile(source_user_data_dir, profile_name, runtime_user_data_dir):
    source_root = Path(source_user_data_dir)
    runtime_root = Path(runtime_user_data_dir)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    for entry_name in ROOT_ENTRIES_TO_COPY:
        source_entry = source_root / entry_name
        destination_entry = runtime_root / entry_name
        if not source_entry.exists():
            continue
        if source_entry.is_dir():
            shutil.copytree(source_entry, destination_entry, symlinks=True, ignore=_ignore_browser_entries)
        else:
            shutil.copy2(source_entry, destination_entry)

    profile_source = source_root / profile_name
    if not profile_source.exists():
        raise FileNotFoundError(f"No existe el perfil '{profile_name}' en {source_root}")

    shutil.copytree(
        profile_source,
        runtime_root / profile_name,
        symlinks=True,
        ignore=_ignore_browser_entries,
    )
    return runtime_root


def load_google_cookies_for_playwright(profile_source, profile_name="Default"):
    if not profile_source:
        return []

    try:
        import browser_cookie3
    except ImportError:
        return []

    source_root = Path(profile_source.get("user_data_dir") or "")
    if not source_root:
        return []

    browser_name = (profile_source.get("name") or "").lower()
    cookie_file = source_root / profile_name / "Cookies"
    key_file = source_root / "Local State"

    if "brave" in browser_name or "bravesoftware" in str(source_root).lower():
        loader = browser_cookie3.brave
    else:
        loader = browser_cookie3.chromium

    kwargs = {}
    if cookie_file.exists():
        kwargs["cookie_file"] = str(cookie_file)
    if key_file.exists():
        kwargs["key_file"] = str(key_file)

    try:
        jar = loader(**kwargs)
    except Exception:
        try:
            jar = loader()
        except Exception:
            return []

    cookies = []
    seen = set()
    for cookie in jar:
        domain = (cookie.domain or "").lower()
        if "youtube.com" not in domain and "google.com" not in domain:
            continue
        key = (cookie.domain, cookie.path, cookie.name)
        if key in seen:
            continue
        seen.add(key)

        rest = {str(name).lower(): value for name, value in cookie._rest.items()}
        item = {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path or "/",
            "secure": bool(cookie.secure),
            "httpOnly": "httponly" in rest,
        }
        if cookie.expires and cookie.expires > 0:
            item["expires"] = float(cookie.expires)
        same_site = rest.get("samesite")
        if same_site:
            normalized = str(same_site).strip().capitalize()
            if normalized in {"Lax", "None", "Strict"}:
                item["sameSite"] = normalized
        cookies.append(item)
    return cookies
