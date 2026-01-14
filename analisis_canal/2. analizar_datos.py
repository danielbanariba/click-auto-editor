import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone

from config import (
    MIN_DAYS_FOR_RATE,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    STOPWORDS,
    TOP_N,
    ensure_directories,
)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def parse_date(value: str):
    if not value:
        return None
    return datetime.strptime(value, DATE_FORMAT).replace(tzinfo=timezone.utc)


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def extract_keywords(title: str):
    text = normalize_text(title.lower())
    words = re.findall(r"[a-z0-9]+", text)
    return [word for word in words if word not in STOPWORDS and len(word) >= 3]


def build_metrics(videos):
    now = datetime.now(timezone.utc)
    metrics = []

    for video in videos:
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})

        published_at = snippet.get("publishedAt")
        published_dt = parse_date(published_at)
        days = 0
        if published_dt:
            days = max((now - published_dt).days, 1)

        views = to_int(stats.get("viewCount"))
        likes = to_int(stats.get("likeCount"))
        comments = to_int(stats.get("commentCount"))

        adjusted_days = max(days, MIN_DAYS_FOR_RATE)
        views_per_day = views / adjusted_days if adjusted_days > 0 else 0.0
        engagement = 0.0
        if views > 0:
            engagement = (likes + comments) / views * 1000

        metrics.append(
            {
                "id": video.get("id"),
                "title": snippet.get("title", ""),
                "publishedAt": published_at,
                "views": views,
                "likes": likes,
                "comments": comments,
                "days": days,
                "views_per_day": views_per_day,
                "engagement_per_1000": engagement,
            }
        )

    return metrics


def write_report(channel, metrics, keyword_counts):
    title = channel.get("snippet", {}).get("title", "")
    stats = channel.get("statistics", {})
    subs = stats.get("subscriberCount", "0")
    total_views = stats.get("viewCount", "0")
    total_videos = stats.get("videoCount", "0")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    report_lines = []
    report_lines.append(f"# Reporte de canal ({today})")
    report_lines.append("")
    report_lines.append(f"Canal: {title}")
    report_lines.append(f"Suscriptores: {subs}")
    report_lines.append(f"Vistas totales: {total_views}")
    report_lines.append(f"Videos totales: {total_videos}")
    report_lines.append("")

    def add_section(header, items, formatter):
        report_lines.append(header)
        for item in items:
            report_lines.append(formatter(item))
        report_lines.append("")

    top_views = sorted(metrics, key=lambda x: x["views"], reverse=True)[:TOP_N]
    top_velocity = sorted(metrics, key=lambda x: x["views_per_day"], reverse=True)[:TOP_N]
    top_engagement = sorted(
        metrics, key=lambda x: x["engagement_per_1000"], reverse=True
    )[:TOP_N]

    add_section(
        f"## Top {TOP_N} por vistas",
        top_views,
        lambda x: (
            f"- {x['title']} | {x['views']} vistas | {x['publishedAt']}"
        ),
    )

    add_section(
        f"## Top {TOP_N} por vistas/dia",
        top_velocity,
        lambda x: (
            f"- {x['title']} | {x['views_per_day']:.2f} vistas/dia | {x['views']} vistas"
        ),
    )

    add_section(
        f"## Top {TOP_N} por engagement (likes+comentarios por 1000 vistas)",
        top_engagement,
        lambda x: (
            f"- {x['title']} | {x['engagement_per_1000']:.2f} | {x['likes']} likes | {x['comments']} comentarios"
        ),
    )

    report_lines.append("## Palabras mas frecuentes en titulos")
    for word, count in keyword_counts.most_common(15):
        report_lines.append(f"- {word}: {count}")
    report_lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "reporte_resumen.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Reporte generado: {report_path}")


def main():
    print("=== Analizar datos ===")
    ensure_directories()

    channel = load_json(RAW_DIR / "canal.json")
    videos = load_json(RAW_DIR / "videos.json")

    metrics = build_metrics(videos)
    save_json(PROCESSED_DIR / "videos_resumen.json", metrics)
    print(f"[OK] Resumen guardado: {len(metrics)} videos")

    keyword_counts = Counter()
    for item in metrics:
        keyword_counts.update(extract_keywords(item.get("title", "")))

    write_report(channel, metrics, keyword_counts)


if __name__ == "__main__":
    main()
