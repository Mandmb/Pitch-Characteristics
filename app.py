import io
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Pitcher Similarity Finder", layout="wide")

DEFAULT_METRICS = [
    "Vel", "VelMax", "Spin", "Extension", "IndVertBrk", "HorzBrk", "Rel. Height", "RSd"
]
PITCH_ORDER = ["Fastball", "Sinker", "Cutter", "Slider", "Curveball", "Change", "Splitter", "Sweeper"]
PITCH_ALIASES = {
    "four-seam fastball": "Fastball", "four seam fastball": "Fastball", "4-seam fastball": "Fastball", "4 seam fastball": "Fastball", "four-seam": "Fastball", "four seam": "Fastball", "4-seam": "Fastball", "4 seam": "Fastball", "fastball": "Fastball", "fb": "Fastball", "ff": "Fastball",
    "sinker": "Sinker", "two-seam": "Sinker", "two seam": "Sinker", "si": "Sinker",
    "cutter": "Cutter", "cut fastball": "Cutter", "fc": "Cutter",
    "slider": "Slider", "sl": "Slider",
    "curveball": "Curveball", "curve": "Curveball", "cu": "Curveball", "knuckle curve": "Curveball",
    "change": "Change", "changeup": "Change", "ch": "Change",
    "splitter": "Splitter", "split": "Splitter", "fs": "Splitter",
    "sweeper": "Sweeper", "sweep": "Sweeper", "st": "Sweeper",
}


def normalize_person_name(value) -> str:
    """Normalize pitcher names so self-comparison removal is more reliable.

    Handles case differences, extra spaces, accents, punctuation, and CSV values
    that come through as numbers/objects.
    """
    if pd.isna(value):
        return ""
    import unicodedata
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    cleaned_chars = []
    for ch in text:
        cleaned_chars.append(ch if ch.isalnum() else " ")
    return " ".join("".join(cleaned_chars).split())


HEADSHOT_DEBUG = []

def normalize_identifier(value) -> str:
    """Normalize ID values like 807842, 807842.0, or ' 807842 '."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "playerid", "id"}:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return re.sub(r"\s+", "", text).lower()


def possible_id_columns(columns: List[str]) -> List[str]:
    candidates = [
        "playerId", "PlayerId", "player_id", "playerID", "MLBAMID", "mlbamId",
        "mlbam_id", "pitcherId", "pitcher_id", "PitcherId", "bamId", "entityKey"
    ]
    found = []
    for cand in candidates:
        col = guess_column(columns, [cand])
        if col and col not in found:
            found.append(col)
    return found


def possible_name_columns(columns: List[str]) -> List[str]:
    candidates = [
        "playerFullName", "Pitcher", "Pitcher Name", "Player", "Player Name",
        "Name", "pitcher_name", "playerName", "lastName", "playerLastName"
    ]
    found = []
    for cand in candidates:
        col = guess_column(columns, [cand])
        if col and col not in found:
            found.append(col)
    return found



def slugify_player_name_for_milb(name: str) -> str:
    """Build the MiLB/MLB player-page slug: firstname-lastname-playerid."""
    if not name:
        return ""
    import unicodedata
    text = str(name).strip()
    # Remove common CSV artifacts before slugifying.
    text = re.sub(r"\([^)]*\)", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _normalize_image_bytes_for_reportlab(data: bytes) -> Optional[bytes]:
    """Return JPEG/PNG bytes that ReportLab can reliably draw.

    MLB/MiLB's image service often returns WEBP/AVIF when the request advertises
    support for those formats. Streamlit can display them, but ReportLab may not
    draw them into the PDF on every Mac/Python install. To make auto-headshots
    dependable, convert anything Pillow can open into PNG bytes.
    """
    if not data or len(data) < 500:
        return None
    # JPEG / PNG are already safe for ReportLab.
    if data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG"):
        return data
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None




def _headshot_cache_dir():
    """Local persistent cache so headshots that work once keep working later.

    MiLB/MLB image services occasionally throttle or return a temporary failure.
    A local cache makes the report dependable once a player image has been found.
    """
    from pathlib import Path
    d = Path.cwd() / "headshot_cache"
    try:
        d.mkdir(exist_ok=True)
    except Exception:
        pass
    return d


def _headshot_cache_path(pid: str):
    return _headshot_cache_dir() / f"{pid}.png"


def _load_cached_headshot(pid: str) -> Optional[bytes]:
    try:
        path = _headshot_cache_path(pid)
        if path.exists() and path.stat().st_size > 500:
            data = path.read_bytes()
            normalized = _normalize_image_bytes_for_reportlab(data)
            if normalized:
                return normalized
    except Exception:
        return None
    return None


def _save_cached_headshot(pid: str, data: bytes) -> None:
    try:
        normalized = _normalize_image_bytes_for_reportlab(data)
        if normalized:
            _headshot_cache_path(pid).write_bytes(normalized)
    except Exception:
        pass

def _download_image_bytes(url: str, headers: Dict[str, str], timeout: int = 12, attempts: int = 3) -> Optional[bytes]:
    """Download, validate, and normalize an image URL for PDF rendering.

    Uses requests first because it handles TLS/certificates more reliably on
    many Macs than urllib. Falls back to urllib if requests is unavailable.
    Also records a short diagnostic trail so the app can show whether the
    headshot URL was reached, blocked, or returned a non-image response.
    """
    global HEADSHOT_DEBUG

    def _ok(data: bytes, content_type: str) -> Optional[bytes]:
        content_type = (content_type or "").lower()
        if not data:
            return None
        is_image = (
            "image" in content_type
            or data.startswith(b"\x89PNG")
            or data.startswith(b"\xff\xd8")
            or data[:4] == b"RIFF"
        )
        if not is_image:
            return None
        return _normalize_image_bytes_for_reportlab(data)

    import time
    try:
        import requests
        session = requests.Session()
        for i in range(max(1, attempts)):
            try:
                r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                HEADSHOT_DEBUG.append(f"try {i+1}: {r.status_code} {r.headers.get('content-type','')} {url}")
                if r.status_code == 200:
                    normalized = _ok(r.content, r.headers.get("Content-Type", ""))
                    if normalized:
                        return normalized
                # retry temporary blocks / throttles
                if r.status_code in (403, 408, 429, 500, 502, 503, 504):
                    time.sleep(0.6 + i * 0.5)
                    continue
                break
            except Exception as exc:
                HEADSHOT_DEBUG.append(f"requests try {i+1} error: {type(exc).__name__}: {url}")
                time.sleep(0.6 + i * 0.5)
    except Exception as exc:
        HEADSHOT_DEBUG.append(f"requests setup error: {type(exc).__name__}: {url}")

    import urllib.request
    for i in range(max(1, attempts)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
                HEADSHOT_DEBUG.append(f"urllib try {i+1}: {getattr(resp, 'status', '')} {content_type} {url}")
            normalized = _ok(data, content_type)
            if normalized:
                return normalized
        except Exception as exc:
            HEADSHOT_DEBUG.append(f"urllib try {i+1} error: {type(exc).__name__}: {url}")
            time.sleep(0.6 + i * 0.5)
    return None


def get_auto_headshot_bytes(id_keys, name_candidates=None) -> Optional[bytes]:
    """Fetch a player headshot using playerId and the MiLB/MLB player page.

    Best-effort order:
    1) Open the MiLB player page built as firstname-lastname-playerid, e.g.
       https://www.milb.com/es/player/jacob-degrom-594798, then parse its
       og:image / twitter:image / embedded headshot URLs.
    2) Fall back to public MLB image-service URLs that MiLB pages commonly use.

    If MiLB/MLB blocks the request or there is no photo, return None so the app
    still generates the report and the user can upload a photo manually.
    """
    import html
    import urllib.request

    clean_ids = []
    for raw in id_keys or []:
        pid = normalize_identifier(raw)
        if pid.isdigit() and 4 <= len(pid) <= 10 and pid not in clean_ids:
            clean_ids.append(pid)

    clean_names = []
    for raw in name_candidates or []:
        name = str(raw or "").strip()
        slug = slugify_player_name_for_milb(name)
        if slug and slug not in clean_names and slug not in {"playerid", "splitby", "nan", "none"}:
            clean_names.append(slug)

    if not clean_ids:
        return None

    # 0) Use local cache first. This is the key reliability layer: once a
    # headshot successfully loads, future exports do not depend on MiLB/MLB
    # responding perfectly every time.
    for pid in clean_ids:
        cached = _load_cached_headshot(pid)
        if cached:
            HEADSHOT_DEBUG.append(f"cache hit: {pid}")
            return cached

    html_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    }
    image_headers = {
        "User-Agent": html_headers["User-Agent"],
        # Do NOT advertise AVIF/WEBP. Those can download successfully but fail
        # when ReportLab tries to place them in the PDF. Prefer PNG/JPEG.
        "Accept": "image/png,image/jpeg,image/apng,image/*;q=0.8,*/*;q=0.5",
        "Referer": "https://www.milb.com/",
    }

    # 1) Try the MiLB/MLB page URL the user described and scrape the image URL.
    for pid in clean_ids:
        page_urls = []
        for slug in clean_names:
            page_urls.extend([
                f"https://www.milb.com/es/player/{slug}-{pid}",
                f"https://www.milb.com/player/{slug}-{pid}",
                f"https://www.mlb.com/player/{slug}-{pid}",
            ])
        for page_url in page_urls:
            try:
                req = urllib.request.Request(page_url, headers=html_headers)
                with urllib.request.urlopen(req, timeout=7) as resp:
                    page = resp.read().decode("utf-8", errors="ignore")
            except Exception:
                continue

            # Pull likely image URLs from meta tags or embedded JSON.
            candidates = []
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<img[^>]+src=["\']([^"\']*(?:headshot|people)[^"\']*)["\']',
                r'<source[^>]+srcset=["\']([^"\']*(?:headshot|people)[^"\']*)["\']',
                r'"(?:headshot|image|imageUrl|photoUrl)"\s*:\s*"([^"]+)"',
                r'(https?:\\/\\/[^" <]+(?:headshot|people)[^" <]+)',
                r'(https?://[^"\'<\s]+(?:headshot|people)[^"\'<\s]+)',
            ]
            for pat in patterns:
                for match in re.findall(pat, page, flags=re.IGNORECASE):
                    raw_match = html.unescape(match).replace("\\/", "/")
                    # srcset values can contain multiple URLs separated by commas.
                    for piece in re.split(r",\s*", raw_match):
                        url = piece.strip().split(" ")[0]
                        if not url:
                            continue
                        if url.startswith("//"):
                            url = "https:" + url
                        elif url.startswith("/"):
                            url = "https://www.milb.com" + url
                        # Avoid tiny 1x transparent/data images.
                        if url.startswith("http") and "headshot" in url.lower() and url not in candidates:
                            candidates.append(url)

            # Prefer URLs that include this playerId or headshot/current.
            candidates = sorted(
                candidates,
                key=lambda u: (str(pid) not in u, "headshot" not in u.lower(), len(u)),
            )
            for image_url in candidates[:10]:
                data = _download_image_bytes(image_url, image_headers)
                if data:
                    _save_cached_headshot(str(pid), data)
                    return data

    # 2) Fallback: public MLB/MiLB image endpoints. Try several sizes/transforms.
    # The img.mlbstatic Cloudinary route is the same image family used on
    # MiLB player pages. The d_people... transform means the request returns a
    # valid generic placeholder if a true headshot is unavailable, so we try
    # non-placeholder style URLs first and the Cloudinary fallback last.
    url_templates = [
        # Direct image route from MiLB player pages. Try both no-transform and transformed versions.
        "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_360/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_180/v1/people/{pid}/headshot/milb/current",
        # MiLB player pages currently expose this exact route. Note the encoded comma.
        # Example from Jacob deGrom's MiLB page:
        # https://img.mlbstatic.com/mlb-photos/image/upload/c_fill%2Cg_auto/w_180/v1/people/594798/headshot/milb/current
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill%2Cg_auto/w_360/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill%2Cg_auto/w_240/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill%2Cg_auto/w_180/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_360/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_240/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_180/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_png,c_fill,g_auto,w_360/v1/people/{pid}/headshot/milb/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_png,c_fill,g_auto,w_240/v1/people/{pid}/headshot/milb/current",
        # Force PNG output from Cloudinary next so ReportLab can draw it.
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_png,w_720,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_png,w_360,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_png,w_240,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_jpg,w_720,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/f_jpg,w_360,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_720,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_360,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:67:current.png,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,d_people:generic:headshot:67:current.png,q_auto:best/v1/people/{pid}/headshot/67/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_720,q_auto:best/v1/people/{pid}/headshot/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_360,q_auto:best/v1/people/{pid}/headshot/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best/v1/people/{pid}/headshot/current",
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/current",
        "https://content.mlb.com/images/headshots/current/240x240/{pid}.png",
        "https://content.mlb.com/images/headshots/current/120x120/{pid}.png",
        "https://content.mlb.com/images/headshots/current/60x60/{pid}.png",
    ]

    for pid in clean_ids:
        for template in url_templates:
            data = _download_image_bytes(template.format(pid=pid), image_headers)
            if data:
                _save_cached_headshot(str(pid), data)
                return data

    return None


def fetch_mlb_person_bio(id_keys) -> Dict[str, str]:
    """Fetch age/hand/position from MLB Stats API using MLBAM playerId.

    This is optional and best-effort: if the user is offline or the API has no
    record for the player, return an empty dict so the report still generates.
    """
    import urllib.request, json
    clean_ids = []
    for raw in id_keys or []:
        pid = normalize_identifier(raw)
        if pid.isdigit() and 4 <= len(pid) <= 10 and pid not in clean_ids:
            clean_ids.append(pid)
    for pid in clean_ids:
        try:
            req = urllib.request.Request(
                f"https://statsapi.mlb.com/api/v1/people/{pid}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            people = payload.get("people") or []
            if not people:
                continue
            p = people[0]
            out = {}
            if p.get("currentAge") not in [None, ""]:
                out["Age"] = str(p.get("currentAge"))
            if p.get("pitchHand", {}).get("code"):
                out["Throws"] = str(p.get("pitchHand", {}).get("code"))
            if p.get("primaryPosition", {}).get("abbreviation"):
                out["Position"] = str(p.get("primaryPosition", {}).get("abbreviation"))
            return out
        except Exception:
            continue
    return {}


def read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def normalize_pitch_name(value):
    if pd.isna(value):
        return np.nan
    raw = str(value).strip()
    key = raw.lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())

    # Exact alias match first.
    if key in PITCH_ALIASES:
        return PITCH_ALIASES[key]

    # Then token/phrase match. This catches values like:
    # "Fastball Pitch Characteristics", "Four-Seam Fastball", "FF", etc.
    aliases_by_length = sorted(PITCH_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
    for alias, canonical in aliases_by_length:
        alias_clean = alias.lower().replace("_", " ").replace("-", " ")
        alias_clean = " ".join(alias_clean.split())
        if alias_clean and alias_clean in key:
            return canonical

    return raw




def canonical_pitch_for_reports(value):
    """Return a clean single pitch type or None for aggregate/header rows.

    The target CSV can contain summary rows such as TOTAL, FastSink, Hard
    (fast/si/ct), Breaking (cv/sld/sw), and Sweepers and Sliders. Those rows
    should not be treated as individual pitches because they double-count pitch
    usage and distort the report visuals. This helper keeps only real pitch rows.
    """
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # Remove TrackMan-style sort suffixes like "Slider^3".
    raw = re.sub(r"\^\d+\s*$", "", raw).strip()
    key = raw.lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())

    # Obvious non-pitch / aggregate rows.
    bad_exact = {"total", "pitch type", "unknown", "nan", "none", "null", "fastsink"}
    if key in bad_exact:
        return None
    bad_phrases = [
        " and ", "breaking", "soft", "hard", "+", "cv/sld", "chg/splt",
        "fast/si", "sweepers and sliders", "breaking+", "special"
    ]
    if any(b in key for b in bad_phrases):
        return None

    # Special cases before generic alias matching.
    if "2s" in key or "two seam" in key or "two-seam" in key:
        if "sinker" in key or "fastball" in key:
            return "Sinker"
    if "4s" in key or "four seam" in key or "four-seam" in key:
        return "Fastball"

    canonical = normalize_pitch_name(raw)
    return canonical if canonical in PITCH_ORDER else None

def infer_pitch_from_filename(filename: str) -> Optional[str]:
    """Infer pitch type from uploaded comparison CSV filename.

    Examples:
    - Fastball Pitch Characteristics.csv -> Fastball
    - Slider Pitch Characteristics.csv -> Slider
    - 2026 Sweeper Pitch Characteristics.csv -> Sweeper
    """
    if not filename:
        return None
    cleaned = str(filename).lower().replace("_", " ").replace("-", " ")
    # Check longer aliases first so phrases like four seam are captured before generic words.
    aliases_by_length = sorted(PITCH_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
    for alias, canonical in aliases_by_length:
        alias_clean = alias.lower().replace("_", " ").replace("-", " ")
        if alias_clean in cleaned:
            return canonical
    return None

def guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    for c in columns:
        c_norm = c.lower().strip().replace("_", " ")
        for cand in candidates:
            if cand.lower().strip().replace("_", " ") in c_norm:
                return c
    return None


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def safe_filename(value: str) -> str:
    text = normalize_person_name(value).replace(" ", "_")
    return text or "pitcher_similarity"


def similarity_tier(score) -> str:
    try:
        score = float(score)
    except Exception:
        return ""
    if score >= 95:
        return "Elite match"
    if score >= 90:
        return "Very similar"
    if score >= 85:
        return "Similar"
    return "Loose comp"


def fmt_num(value, decimals=1):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def has_any_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    return guess_column(list(df.columns), names)


def classify_level(value: str) -> str:
    text = str(value).lower()
    mlb_terms = ["mlb", "major", "majors", "big league"]
    milb_terms = ["milb", "minor", "aaa", "aa", "a+", "high-a", "high a", "low-a", "low a", "a-ball", "rookie", "dsl", "acl", "fcl"]
    if any(term in text for term in mlb_terms):
        return "MLB"
    if any(term in text for term in milb_terms):
        return "MiLB"
    return "Other"


def filter_comp_pool(comp_df: pd.DataFrame, level_filter: str, min_pitch_count: int) -> pd.DataFrame:
    out = comp_df.copy()
    if level_filter != "All Players":
        level_col = has_any_column(out, ["Level", "level", "League", "league", "competition", "Competition"])
        if level_col:
            desired = "MLB" if level_filter == "MLB Players Only" else "MiLB"
            out = out[out[level_col].apply(classify_level) == desired].copy()
        else:
            st.info("No Level/League column was found, so the MLB/MiLB filter was skipped.")
    if min_pitch_count > 1:
        count_col = has_any_column(out, ["PitchCount", "pitchCount", "Pitches", "pitches", "Count", "count", "pitch_count"])
        if count_col:
            out[count_col] = pd.to_numeric(out[count_col], errors="coerce")
            out = out[out[count_col] >= min_pitch_count].copy()
        else:
            # Raw pitch-level files are handled by aggregation PitchCount after upload.
            pass
    return out


def make_radar_chart(target_row: pd.Series, comp_row: pd.Series, metrics: List[str]) -> Optional[io.BytesIO]:
    usable = [m for m in metrics if m in target_row.index and m in comp_row.index and not pd.isna(target_row[m]) and not pd.isna(comp_row[m])]
    if len(usable) < 3:
        return None
    try:
        import matplotlib.pyplot as plt
        vals = []
        comp_vals = []
        for m in usable:
            a = float(target_row[m])
            b = float(comp_row[m])
            lo, hi = min(a, b), max(a, b)
            if math.isclose(lo, hi):
                vals.append(0.5); comp_vals.append(0.5)
            else:
                vals.append((a - lo) / (hi - lo))
                comp_vals.append((b - lo) / (hi - lo))
        angles = np.linspace(0, 2 * np.pi, len(usable), endpoint=False).tolist()
        vals += vals[:1]
        comp_vals += comp_vals[:1]
        angles += angles[:1]
        fig = plt.figure(figsize=(4.8, 4.8))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2, label="Target")
        ax.fill(angles, vals, alpha=0.12)
        ax.plot(angles, comp_vals, linewidth=2, label="Closest Comp")
        ax.fill(angles, comp_vals, alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(usable, fontsize=8)
        ax.set_yticklabels([])
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None



def make_score_gauge(score: float) -> Optional[io.BytesIO]:
    """Create a compact similarity score gauge image."""
    try:
        import matplotlib.pyplot as plt
        score = 0 if pd.isna(score) else max(0, min(100, float(score)))
        fig, ax = plt.subplots(figsize=(2.2, 1.15))
        ax.axis("off")
        theta = np.linspace(np.pi, 0, 120)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=8, solid_capstyle="round")
        pct = score / 100
        theta2 = np.linspace(np.pi, np.pi * (1 - pct), 120)
        ax.plot(np.cos(theta2), np.sin(theta2), linewidth=8, solid_capstyle="round")
        needle_theta = np.pi * (1 - pct)
        ax.plot([0, 0.78*np.cos(needle_theta)], [0, 0.78*np.sin(needle_theta)], linewidth=2)
        ax.text(0, -0.20, f"{score:.1f}%", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.35, 1.15)
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None



def best_overall_score_and_name(overall: pd.DataFrame, comp_pitcher_col: str) -> Tuple[Optional[float], str, str]:
    """Return top overall score, pitcher name, and matched pitch list safely."""
    if overall is None or overall.empty:
        return None, "No overall comp", ""
    row = overall.iloc[0]
    score = row.get("Similarity Score", row.get("Avg_Similarity", np.nan))
    try:
        score = float(score)
    except Exception:
        score = None
    name = str(row.get(comp_pitcher_col, row.get("Pitcher", "Top Comp")))
    pitches = str(row.get("Matched_Pitches", ""))
    return score, name, pitches


def lookup_target_bio(target_name: str, comp_df_for_lookup: pd.DataFrame, target_id_keys: set) -> Dict[str, str]:
    """Find org/team/hand/level fields for the target pitcher from comparison files when possible."""
    bio = {}
    if comp_df_for_lookup is None or comp_df_for_lookup.empty or not target_id_keys:
        return bio
    id_mask = pd.Series(False, index=comp_df_for_lookup.index)
    for col in possible_id_columns(list(comp_df_for_lookup.columns)):
        if col in comp_df_for_lookup.columns:
            id_mask = id_mask | comp_df_for_lookup[col].apply(normalize_identifier).isin(target_id_keys)
    rows = comp_df_for_lookup.loc[id_mask].copy()
    if rows.empty:
        return bio
    field_aliases = {
        "Org": ["currentOrg", "newestOrg", "Organization", "Org", "TeamOrg"],
        "Team": ["currentTeamName", "newestTeamName", "Team", "teamName", "Club"],
        "Level": ["currentTeamLevel", "newestTeamLevel", "Level", "league", "League"],
        "Throws": ["throwsHand", "Throws", "PitcherHand", "Handedness"],
        "Position": ["pos", "Position"],
        "Age": ["Age", "age", "PlayerAge", "playerAge", "currentAge"],
        "BirthDate": ["birthDate", "BirthDate", "DOB", "dob", "dateOfBirth"],
    }
    for label, aliases in field_aliases.items():
        col = guess_column(list(rows.columns), aliases)
        if col:
            vals = rows[col].dropna().astype(str).str.strip()
            vals = vals[~vals.str.lower().isin(["nan", "none", "null", ""])]
            if not vals.empty:
                if label == "BirthDate" and "Age" not in bio:
                    try:
                        bd = pd.to_datetime(vals.iloc[0], errors="coerce")
                        if pd.notna(bd):
                            today = pd.Timestamp.today()
                            age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
                            bio["Age"] = str(int(age))
                    except Exception:
                        pass
                elif label != "BirthDate":
                    bio[label] = vals.iloc[0]
    return bio


def build_pitch_usage_rows(target_agg: pd.DataFrame, target_pitch_col: str) -> List[List[str]]:
    """Build compact pitch usage table from target PitchCount, if available."""
    if target_agg is None or target_agg.empty or "PitchCount" not in target_agg.columns:
        return []
    work = target_agg.copy()
    work = work[work[target_pitch_col].isin(PITCH_ORDER)].copy()
    work["PitchCount"] = pd.to_numeric(work["PitchCount"], errors="coerce").fillna(0)
    # Only true individual pitch rows should drive usage. Aggregate categories
    # are removed earlier, so this total produces a clean 100% distribution.
    total = float(work["PitchCount"].sum())
    if total <= 0:
        return []
    rows = [["Pitch", "P", "Usage"]]
    for p in [x for x in PITCH_ORDER if x in set(work[target_pitch_col])]:
        r = work[work[target_pitch_col] == p].iloc[0]
        cnt = float(r.get("PitchCount", 0))
        rows.append([p, f"{int(cnt):,}", f"{cnt/total*100:.0f}%"])
    return rows


def make_reportlab_score_card(score: Optional[float], comp_name: str, pitches: str, primary, accent):
    """ReportLab-native score card so PDF does not depend on matplotlib for the gauge."""
    from reportlab.graphics.shapes import Drawing, Wedge, Circle, String, Line
    from reportlab.lib import colors
    w, h = 165, 78
    d = Drawing(w, h)
    score_val = 0 if score is None or pd.isna(score) else max(0, min(100, float(score)))
    cx, cy, r = 43, 38, 28
    # Background ring and score wedge
    d.add(Circle(cx, cy, r, fillColor=None, strokeColor=colors.HexColor("#d1d5db"), strokeWidth=7))
    # Full-circle wedge proportion approximates a gauge while staying robust in reportlab.
    angle = 360 * (score_val / 100)
    if angle > 0:
        d.add(Wedge(cx, cy, r, 90, 90-angle, fillColor=None, strokeColor=accent, strokeWidth=7))
    d.add(String(cx, cy-5, f"{score_val:.1f}", textAnchor="middle", fontSize=15, fontName="Helvetica-Bold", fillColor=primary))
    d.add(String(cx, cy-19, "SIM", textAnchor="middle", fontSize=6, fillColor=colors.HexColor("#4b5563")))
    d.add(String(82, 55, "Top Comp", fontSize=7, fillColor=colors.HexColor("#4b5563")))
    d.add(String(82, 41, comp_name[:26], fontSize=9, fontName="Helvetica-Bold", fillColor=primary))
    d.add(String(82, 27, pitches[:32] if pitches else "Matched pitch types", fontSize=6.5, fillColor=colors.HexColor("#4b5563")))
    return d


def make_reportlab_movement_plot(target_agg: pd.DataFrame, pitch_results: Dict[str, pd.DataFrame], target_pitch_col: str, primary, accent):
    """ReportLab-native movement plot using HorzBrk/IndVertBrk, no matplotlib dependency."""
    from reportlab.graphics.shapes import Drawing, Line, Circle, String, Rect
    from reportlab.lib import colors
    # Accept aliases if columns were renamed differently.
    hb_col = guess_column(list(target_agg.columns), ["HorzBrk", "HB", "Horizontal Break", "HorizontalBreak"])
    ivb_col = guess_column(list(target_agg.columns), ["IndVertBrk", "IVB", "Induced Vertical Break", "InducedVerticalBreak"])
    if not hb_col or not ivb_col:
        return None
    rows = []
    for pitch, res in pitch_results.items():
        if res is None or res.empty:
            continue
        tr = target_agg[target_agg[target_pitch_col] == pitch]
        if tr.empty:
            continue
        t = tr.iloc[0]
        c = res.iloc[0]
        c_hb = guess_column(list(res.columns), ["HorzBrk", "HB", "Horizontal Break", "HorizontalBreak"])
        c_ivb = guess_column(list(res.columns), ["IndVertBrk", "IVB", "Induced Vertical Break", "InducedVerticalBreak"])
        if not c_hb or not c_ivb:
            continue
        vals = [t.get(hb_col), t.get(ivb_col), c.get(c_hb), c.get(c_ivb)]
        if any(pd.isna(v) for v in vals):
            continue
        try:
            rows.append((pitch, float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])))
        except Exception:
            continue
    if not rows:
        return None
    w, h = 250, 118
    left, bottom, plot_w, plot_h = 35, 23, 198, 75
    xs = [v for row in rows for v in (row[1], row[3])]
    ys = [v for row in rows for v in (row[2], row[4])]
    xmin, xmax = min(xs + [-20]), max(xs + [20])
    ymin, ymax = min(ys + [-5]), max(ys + [25])
    # Add padding
    xr = xmax - xmin or 1
    yr = ymax - ymin or 1
    xmin -= xr*0.08; xmax += xr*0.08; ymin -= yr*0.08; ymax += yr*0.08
    def sx(x): return left + (x-xmin)/(xmax-xmin)*plot_w
    def sy(y): return bottom + (y-ymin)/(ymax-ymin)*plot_h
    d = Drawing(w, h)
    d.add(Rect(left, bottom, plot_w, plot_h, fillColor=colors.white, strokeColor=colors.HexColor("#cbd5e1"), strokeWidth=0.5))
    if xmin < 0 < xmax:
        d.add(Line(sx(0), bottom, sx(0), bottom+plot_h, strokeColor=colors.HexColor("#9ca3af"), strokeWidth=0.6))
    if ymin < 0 < ymax:
        d.add(Line(left, sy(0), left+plot_w, sy(0), strokeColor=colors.HexColor("#9ca3af"), strokeWidth=0.6))
    d.add(String(left+plot_w/2, 5, "Horizontal Break", textAnchor="middle", fontSize=6, fillColor=colors.HexColor("#4b5563")))
    d.add(String(2, bottom+plot_h/2, "IVB", fontSize=6, fillColor=colors.HexColor("#4b5563")))
    for pitch, tx, ty, cx, cy in rows:
        d.add(Line(sx(tx), sy(ty), sx(cx), sy(cy), strokeColor=colors.HexColor("#94a3b8"), strokeWidth=0.7))
        d.add(Circle(sx(tx), sy(ty), 3.2, fillColor=accent, strokeColor=accent))
        d.add(Circle(sx(cx), sy(cy), 3.2, fillColor=primary, strokeColor=primary))
        d.add(String(sx(tx)+4, sy(ty)+2, pitch[:2].upper(), fontSize=5.5, fillColor=accent))
    d.add(String(left, h-10, "Target", fontSize=6, fillColor=accent))
    d.add(String(left+45, h-10, "Closest comps", fontSize=6, fillColor=primary))
    return d



def build_why_this_comp(target_agg: pd.DataFrame, pitch_results: Dict[str, pd.DataFrame], target_pitch_col: str, metrics: List[str], comp_pitcher_col: str) -> List[str]:
    """Build short, readable reasons explaining why the top overall comp matched.

    Uses the closest comp on each matched pitch and finds the smallest metric gaps.
    Kept intentionally simple so PDF generation never fails if columns are missing.
    """
    reasons: List[str] = []
    if target_agg is None or target_agg.empty or not pitch_results:
        return reasons

    friendly = {
        "Vel": "velocity",
        "VelMax": "max velocity",
        "Spin": "spin",
        "Extension": "extension",
        "IndVertBrk": "IVB",
        "HorzBrk": "HB",
        "Rel. Height": "release height",
        "RSd": "release side",
    }

    for pitch in PITCH_ORDER:
        res = pitch_results.get(pitch)
        if res is None or res.empty:
            continue
        tr = target_agg[target_agg[target_pitch_col] == pitch]
        if tr.empty:
            continue
        t = tr.iloc[0]
        c = res.iloc[0]
        gaps = []
        for m in metrics:
            if m not in t.index or m not in c.index:
                continue
            tv = pd.to_numeric(pd.Series([t.get(m)]), errors="coerce").iloc[0]
            cv = pd.to_numeric(pd.Series([c.get(m)]), errors="coerce").iloc[0]
            if pd.isna(tv) or pd.isna(cv):
                continue
            gaps.append((abs(float(tv) - float(cv)), m, float(tv), float(cv)))
        if not gaps:
            continue
        gaps.sort(key=lambda x: x[0])
        comp_name = str(c.get(comp_pitcher_col, c.get("Pitcher", "closest comp")))
        best_bits = []
        for gap, m, tv, cv in gaps[:2]:
            unit = " mph" if m in {"Vel", "VelMax"} else ""
            if m == "Spin":
                unit = " rpm"
            elif m in {"Extension", "IndVertBrk", "HorzBrk", "Rel. Height", "RSd"}:
                unit = '"'
            best_bits.append(f"{friendly.get(m, m)} gap {gap:.1f}{unit}")
        if best_bits:
            reasons.append(f"{pitch}: {comp_name} — " + "; ".join(best_bits))
        if len(reasons) >= 5:
            break
    return reasons


def make_percentile_bars(target_agg: pd.DataFrame, comp_agg: pd.DataFrame, target_pitch_col: str, comp_pitch_col: str, metrics: List[str], primary, accent):
    """Create compact target percentile bars across all target pitches.

    Percentiles are computed from the comparison pool by pitch type when possible.
    Returns a ReportLab Drawing or None.
    """
    try:
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.lib import colors
    except Exception:
        return None

    if target_agg is None or target_agg.empty or comp_agg is None or comp_agg.empty:
        return None

    metric_candidates = [m for m in ["Vel", "VelMax", "Spin", "Extension", "IndVertBrk", "HorzBrk"] if m in metrics and m in target_agg.columns and m in comp_agg.columns]
    if not metric_candidates:
        return None

    rows = []
    for m in metric_candidates[:5]:
        vals = []
        for _, tr in target_agg.iterrows():
            pitch = tr.get(target_pitch_col)
            tv = pd.to_numeric(pd.Series([tr.get(m)]), errors="coerce").iloc[0]
            if pd.isna(tv):
                continue
            pool = comp_agg
            if comp_pitch_col in comp_agg.columns:
                same_pitch = comp_agg[comp_agg[comp_pitch_col] == pitch]
                if not same_pitch.empty:
                    pool = same_pitch
            pv = pd.to_numeric(pool[m], errors="coerce").dropna()
            if pv.empty:
                continue
            pct = float((pv <= tv).mean() * 100)
            vals.append(pct)
        if vals:
            label = m.replace("IndVertBrk", "IVB").replace("HorzBrk", "HB").replace("Extension", "Ext")
            rows.append((label, sum(vals) / len(vals)))

    if not rows:
        return None

    w, h = 150, 22 + 16 * len(rows)
    d = Drawing(w, h)
    d.add(String(0, h-9, "Target Percentiles", fontSize=8.5, fontName="Helvetica-Bold", fillColor=primary))
    y = h - 24
    bar_x, bar_w = 48, 72
    for label, pct in rows:
        pct = max(0, min(100, pct))
        d.add(String(0, y+2, label, fontSize=6.5, fillColor=colors.HexColor("#374151")))
        d.add(Rect(bar_x, y, bar_w, 7, fillColor=colors.HexColor("#e5e7eb"), strokeColor=None))
        d.add(Rect(bar_x, y, bar_w * pct / 100, 7, fillColor=accent, strokeColor=None))
        d.add(String(bar_x + bar_w + 5, y, f"{pct:.0f}", fontSize=6.5, fillColor=colors.HexColor("#374151")))
        y -= 16
    return d

def make_movement_plot(target_agg: pd.DataFrame, pitch_results: Dict[str, pd.DataFrame], target_pitch_col: str) -> Optional[io.BytesIO]:
    """Movement plot using HorzBrk and IndVertBrk for target vs closest comp by pitch."""
    if "HorzBrk" not in target_agg.columns or "IndVertBrk" not in target_agg.columns:
        return None
    try:
        import matplotlib.pyplot as plt
        rows = []
        for pitch, res in pitch_results.items():
            if res is None or res.empty:
                continue
            tr = target_agg[target_agg[target_pitch_col] == pitch]
            if tr.empty:
                continue
            t = tr.iloc[0]
            c = res.iloc[0]
            if pd.isna(t.get("HorzBrk")) or pd.isna(t.get("IndVertBrk")) or pd.isna(c.get("HorzBrk")) or pd.isna(c.get("IndVertBrk")):
                continue
            rows.append((pitch, float(t["HorzBrk"]), float(t["IndVertBrk"]), float(c["HorzBrk"]), float(c["IndVertBrk"])))
        if not rows:
            return None
        fig, ax = plt.subplots(figsize=(4.3, 3.1))
        ax.axhline(0, linewidth=0.7)
        ax.axvline(0, linewidth=0.7)
        for pitch, tx, ty, cx, cy in rows:
            ax.scatter([tx], [ty], marker="o", s=42)
            ax.scatter([cx], [cy], marker="x", s=42)
            ax.plot([tx, cx], [ty, cy], linewidth=0.8, alpha=0.7)
            ax.text(tx, ty, pitch[:2].upper(), fontsize=7, ha="left", va="bottom")
        ax.set_xlabel("Horizontal Break")
        ax.set_ylabel("Induced Vertical Break")
        ax.set_title("Pitch Movement: Target vs Closest Comp", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def make_pdf_report(
    target_name: str,
    target_agg: pd.DataFrame,
    comp_agg: pd.DataFrame,
    pitch_results: Dict[str, pd.DataFrame],
    overall: pd.DataFrame,
    target_pitcher_col: str,
    target_pitch_col: str,
    comp_pitcher_col: str,
    comp_pitch_col: str,
    metrics: List[str],
    logo_bytes: Optional[bytes] = None,
    headshot_bytes: Optional[bytes] = None,
    primary_color: str = "#0b1f3a",
    accent_color: str = "#c1121f",
    target_bio: Optional[Dict[str, str]] = None,
) -> bytes:
    """Premium one-page PDF renderer.

    The PDF uses the same aspect ratio and design grid as the clean mockup the
    report is based on (1536 x 1024). Using a custom page size is intentional:
    it prevents the cramped letter-page look and lets the exported PDF match
    the dashboard-style image much more closely.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
    except ImportError as exc:
        raise ImportError("PDF export requires ReportLab. Install it with: python3 -m pip install reportlab") from exc

    # Mockup-sized page. This is the key to keeping the layout spacious.
    W, H = 1536, 1024
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))

    # --- Theme ----------------------------------------------------------------
    def C(hexval, fallback="#0b1f3a"):
        try:
            return colors.HexColor(hexval or fallback)
        except Exception:
            return colors.HexColor(fallback)

    navy = C(primary_color, "#0b1f3a")
    accent = C(accent_color, "#c1121f")
    textc = colors.HexColor("#071f49")
    muted = colors.HexColor("#64748b")
    line = colors.HexColor("#d7dee8")
    line2 = colors.HexColor("#edf1f6")
    pale = colors.HexColor("#f8fafc")
    photo_bg = colors.HexColor("#f1f5f9")
    blue = colors.HexColor("#1457d9")
    green = colors.HexColor("#2e7d32")
    red = colors.HexColor("#d71920")
    orange = colors.HexColor("#f28e2b")
    gold = colors.HexColor("#d99a00")
    purple = colors.HexColor("#6f42c1")
    gray = colors.HexColor("#a6adb7")

    pitch_colors = {
        "Fastball": "#e52620", "Sinker": "#f27a14", "Cutter": "#4e79a7", "Slider": "#6f42c1",
        "Curveball": "#43a047", "Change": "#17becf", "Splitter": "#8c564b", "Sweeper": "#43a047", "Other": "#9ca3af"
    }
    pitch_abbr = {"Fastball":"FA","Sinker":"SI","Cutter":"CT","Slider":"SL","Curveball":"CB","Change":"CH","Splitter":"SP","Sweeper":"SW"}

    def safe(v, default=""):
        if v is None:
            return default
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        s = str(v).strip()
        return default if s.lower() in {"nan", "none", "null", ""} else s

    def shorten(s, max_chars):
        s = safe(s)
        return s if len(s) <= max_chars else s[:max_chars-1] + "…"

    def fmt(v, dec=1):
        try:
            if pd.isna(v):
                return ""
            return f"{float(v):,.{dec}f}"
        except Exception:
            return safe(v)

    def num(v):
        try:
            return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        except Exception:
            return np.nan

    def metric_col(df, aliases):
        if df is None or df.empty:
            return None
        lookup = {}
        for col in df.columns:
            key = str(col).lower().replace(" ", "").replace(".", "").replace("_", "")
            lookup[key] = col
        for name in aliases:
            key = str(name).lower().replace(" ", "").replace(".", "").replace("_", "")
            if key in lookup:
                return lookup[key]
        return None

    def draw_text(x, y, txt, size=12, color=None, bold=False, maxw=None, leading=None, align="left"):
        color = color or textc
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFillColor(color)
        c.setFont(font, size)
        txt = safe(txt)
        if maxw is None:
            if align == "center":
                c.drawCentredString(x, y, txt)
            elif align == "right":
                c.drawRightString(x, y, txt)
            else:
                c.drawString(x, y, txt)
            return y - (leading or size + 4)
        words = txt.split()
        yy = y
        line_txt = ""
        lead = leading or size + 4
        def _draw_line(line_value, yy_value):
            if align == "center":
                c.drawCentredString(x, yy_value, line_value)
            elif align == "right":
                c.drawRightString(x, yy_value, line_value)
            else:
                c.drawString(x, yy_value, line_value)

        for word in words:
            test = (line_txt + " " + word).strip()
            if c.stringWidth(test, font, size) <= maxw or not line_txt:
                line_txt = test
            else:
                _draw_line(line_txt, yy)
                yy -= lead
                line_txt = word
        if line_txt:
            _draw_line(line_txt, yy)
            yy -= lead
        return yy

    def card(x, y, w, h, title=None, subtitle=None):
        c.setFillColor(colors.white)
        c.setStrokeColor(line)
        c.setLineWidth(1.0)
        c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
        if title:
            draw_text(x+18, y+h-30, title.upper(), 15, navy, True, maxw=w-36)
        if subtitle:
            draw_text(x+18 + c.stringWidth(title.upper(), "Helvetica-Bold", 15) + 12, y+h-30, subtitle, 10, muted, False)

    def img_reader(b):
        if not b:
            return None
        try:
            return ImageReader(io.BytesIO(b))
        except Exception:
            return None

    def draw_img(b, x, y, w, h):
        r = img_reader(b)
        if not r:
            return False
        try:
            c.drawImage(r, x, y, w, h, preserveAspectRatio=True, anchor='c', mask='auto')
            return True
        except Exception:
            return False

    def draw_table_lines(x, y_top, w, headers, rows, col_fracs, row_h=34, fs=11, header_fs=9, header_fill=None, blue_col=None):
        """Clean table with subtle row lines, like the mockup."""
        colw = [w*f for f in col_fracs]
        header_fill = header_fill
        if header_fill:
            c.setFillColor(header_fill)
            c.rect(x, y_top-row_h, w, row_h, fill=1, stroke=0)
            header_color = colors.white
        else:
            header_color = muted
        c.setFont("Helvetica-Bold", header_fs)
        c.setFillColor(header_color)
        xx = x
        for i, h in enumerate(headers):
            c.drawCentredString(xx + colw[i]/2, y_top-row_h+11, safe(h).upper())
            xx += colw[i]
        c.setStrokeColor(line2)
        c.setLineWidth(0.8)
        c.line(x, y_top-row_h-2, x+w, y_top-row_h-2)
        yy = y_top - row_h - row_h
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                c.setFillColor(colors.HexColor("#fbfdff"))
                c.rect(x-4, yy+1, w+8, row_h-2, fill=1, stroke=0)
            xx = x
            for i, val in enumerate(row):
                if blue_col is not None and i == blue_col:
                    c.setFillColor(blue); c.setFont("Helvetica-Bold", fs)
                else:
                    c.setFillColor(textc); c.setFont("Helvetica", fs)
                cell_txt = shorten(val, 32)
                # Keep names/labels readable on the left, but center all numeric columns
                # so the table feels much closer to the clean dashboard mockup.
                h_label = safe(headers[i]).lower() if i < len(headers) else ""
                if h_label in {"pitcher", "metric", "matched pitches"}:
                    c.drawString(xx+4, yy+11, cell_txt)
                else:
                    c.drawCentredString(xx + colw[i]/2, yy+11, cell_txt)
                xx += colw[i]
            c.setStrokeColor(line2); c.line(x, yy, x+w, yy)
            yy -= row_h
        return yy

    def draw_header_table(x, y_top, w, headers, rows, col_fracs, row_h=28, fs=9.5, header_color=None, blue_col=None):
        """Compact table with navy/red header for pitch cards."""
        header_color = header_color or navy
        colw = [w*f for f in col_fracs]
        c.setFillColor(header_color); c.rect(x, y_top-row_h, w, row_h, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", fs); c.setFillColor(colors.white)
        xx = x
        for i, h in enumerate(headers):
            c.drawCentredString(xx + colw[i]/2, y_top-row_h+10, safe(h))
            xx += colw[i]
        yy = y_top - row_h
        for r_i, row in enumerate(rows):
            yy -= row_h
            c.setFillColor(colors.white if r_i % 2 == 0 else pale)
            c.rect(x, yy, w, row_h, fill=1, stroke=0)
            xx = x
            for i, val in enumerate(row):
                if blue_col is not None and i == blue_col:
                    c.setFillColor(blue); c.setFont("Helvetica-Bold", fs+0.5)
                else:
                    c.setFillColor(textc); c.setFont("Helvetica", fs)
                cell_txt = shorten(val, 30)
                h_label = safe(headers[i]).lower() if i < len(headers) else ""
                if h_label in {"pitcher", "metric"}:
                    c.drawString(xx+8, yy+10, cell_txt)
                else:
                    c.drawCentredString(xx + colw[i]/2, yy+10, cell_txt)
                xx += colw[i]
            c.setStrokeColor(line2); c.line(x, yy, x+w, yy)
        c.setStrokeColor(line); c.rect(x, yy, w, row_h*(len(rows)+1), fill=0, stroke=1)
        xx = x
        for cw in colw[:-1]:
            xx += cw
            c.line(xx, yy, xx, y_top)
        return yy

    def usage_rows():
        try:
            rows = build_pitch_usage_rows(target_agg, target_pitch_col)
        except Exception:
            rows = [["Pitch", "P", "Usage"]]
        clean = [rows[0]]
        for r in rows[1:]:
            try:
                pct = float(str(r[2]).replace('%',''))
                p = float(str(r[1]).replace(',',''))
            except Exception:
                pct = 0; p = 0
            if pct > 0 or p > 0:
                clean.append(r)
        return clean[:8]

    def best_score():
        if overall is not None and not overall.empty:
            r = overall.iloc[0]
            return r.get("Similarity Score", r.get("Avg_Similarity", np.nan)), safe(r.get(comp_pitcher_col, r.get("Pitcher", ""))), safe(r.get("Matched_Pitches", ""))
        return np.nan, "", ""

    def gauge_png(score):
        try:
            import matplotlib.pyplot as plt
            sc = 0 if pd.isna(score) else max(0, min(100, float(score)))
            fig, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw={'aspect':'equal'})
            ax.pie([sc, 100-sc], startangle=90, counterclock=False, colors=["#d99a00", "#d7dce4"], wedgeprops={'width':.16, 'edgecolor':'white'})
            ax.text(0, 0.10, f"{sc:.1f}", ha='center', va='center', fontsize=38, fontweight='bold', color=primary_color)
            ax.text(0, -0.28, "SIMILARITY", ha='center', va='center', fontsize=10, fontweight='bold', color=primary_color)
            ax.axis('off')
            out = io.BytesIO(); fig.savefig(out, format='png', dpi=220, bbox_inches='tight', transparent=True); plt.close(fig); out.seek(0); return out
        except Exception:
            return None

    def donut_png(rows):
        try:
            import matplotlib.pyplot as plt
            labels=[]; vals=[]; cols=[]
            for r in rows[1:]:
                try: v=float(str(r[2]).replace('%',''))
                except Exception: v=0
                if v > 0:
                    labels.append(safe(r[0])); vals.append(v); cols.append(pitch_colors.get(safe(r[0]), "#9ca3af"))
            if not vals:
                return None
            fig, ax = plt.subplots(figsize=(3.6,3.6), subplot_kw={'aspect':'equal'})
            ax.pie(vals, startangle=90, counterclock=False, colors=cols, wedgeprops={'width':.52, 'edgecolor':'white'}, autopct=lambda p:f"{p:.0f}%" if p>=9 else "", textprops={'fontsize':12, 'weight':'bold', 'color':'white'})
            ax.axis('off')
            out=io.BytesIO(); fig.savefig(out, format='png', dpi=220, bbox_inches='tight', transparent=True); plt.close(fig); out.seek(0); return out
        except Exception:
            return None

    def pitch_row_for(pch):
        tr = target_agg[target_agg[target_pitch_col].astype(str).str.lower() == str(pch).lower()]
        return None if tr.empty else tr.iloc[0]

    def move_png():
        try:
            import matplotlib.pyplot as plt
            hx_t = metric_col(target_agg, ["HorzBrk","HB","HorizontalBreak","Horz Break"])
            vy_t = metric_col(target_agg, ["IndVertBrk","IVB","InducedVertBreak","VerticalBreak","Induced Vertical Break"])
            if not hx_t or not vy_t:
                return None
            rows = []
            for pch in PITCH_ORDER:
                res = pitch_results.get(pch)
                trow = pitch_row_for(pch)
                if res is None or res.empty or trow is None:
                    continue
                hx_c = metric_col(res, ["HorzBrk","HB","HorizontalBreak","Horz Break"])
                vy_c = metric_col(res, ["IndVertBrk","IVB","InducedVertBreak","VerticalBreak","Induced Vertical Break"])
                if not hx_c or not vy_c:
                    continue
                crow = res.iloc[0]
                vals = pd.to_numeric(pd.Series([trow.get(hx_t), trow.get(vy_t), crow.get(hx_c), crow.get(vy_c)]), errors='coerce')
                if vals.isna().any():
                    continue
                rows.append((pch, float(vals.iloc[0]), float(vals.iloc[1]), float(vals.iloc[2]), float(vals.iloc[3]), safe(crow.get(comp_pitcher_col))))
            if not rows:
                return None
            fig, ax = plt.subplots(figsize=(6.2,3.8))
            ax.axhline(0, color="#c4ccd8", lw=1.1, ls='--')
            ax.axvline(0, color="#c4ccd8", lw=1.1, ls='--')
            xs=[]; ys=[]
            first=True
            for pch, tx, ty, cx, cy, cname in rows:
                xs += [tx,cx]; ys += [ty,cy]
                col = pitch_colors.get(pch, "#9ca3af")
                ax.scatter(tx, ty, s=125, facecolors='white', edgecolors=col, linewidths=2.6, label='Target' if first else None, zorder=3)
                ax.scatter(cx, cy, s=95, color=primary_color, label=f'Closest Comp' if first else None, zorder=3)
                ax.plot([tx,cx], [ty,cy], color="#cbd5e1", lw=1.2, zorder=2)
                ax.text(tx+0.6, ty+0.7, pitch_abbr.get(pch,pch[:2].upper()), fontsize=10, color=primary_color, fontweight='bold')
                first=False
            ax.set_xlim(min(-22, min(xs)-5), max(22, max(xs)+5))
            ax.set_ylim(min(-20, min(ys)-5), max(25, max(ys)+5))
            ax.set_xlabel("HORIZONTAL BREAK (IN.)", fontsize=9, color=primary_color, labelpad=9)
            ax.set_ylabel("INDUCED VERTICAL BREAK (IN.)", fontsize=9, color=primary_color, labelpad=9)
            ax.tick_params(labelsize=8, colors=primary_color)
            ax.grid(True, color="#e5e7eb", lw=.7)
            ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
            for spine in ax.spines.values(): spine.set_color("#d7dee8")
            fig.tight_layout(pad=.8)
            out=io.BytesIO(); fig.savefig(out, format='png', dpi=220, bbox_inches='tight', transparent=False); plt.close(fig); out.seek(0); return out
        except Exception:
            return None

    def pitch_percentiles():
        """Return percentile rows separated by pitch type.

        Each target pitch is compared only against the comparison pool for that same pitch type.
        This prevents a single combined percentile card where fastball velocity, slider shape,
        etc. are blended together.
        """
        aliases=[
            ("Velocity","Vel",["Vel"]),
            ("Spin","Spin",["Spin"]),
            ("Extension","Extension",["Extension"]),
            ("IVB","IndVertBrk",["IndVertBrk","IVB"]),
            ("HB","HorzBrk",["HorzBrk","HB"]),
            ("Rel Height","Rel. Height",["Rel. Height","RelHt"]),
            ("Rel Side","RSd",["RSd"]),
        ]
        out=[]
        # Prefer pitch types that actually appear in the similarity results so the card mirrors the report.
        pitch_list=[p for p in PITCH_ORDER if p in pitch_results and pitch_results.get(p) is not None and not pitch_results.get(p).empty]
        if not pitch_list:
            pitch_list=[]
            if target_pitch_col in target_agg.columns:
                for p in target_agg[target_pitch_col].dropna().astype(str).tolist():
                    canon=canonical_pitch_for_reports(p) or p
                    if canon not in pitch_list:
                        pitch_list.append(canon)
        for pch in pitch_list:
            tr = target_agg[target_agg[target_pitch_col].astype(str).str.lower()==str(pch).lower()]
            pool = comp_agg[comp_agg[comp_pitch_col].astype(str).str.lower()==str(pch).lower()] if comp_pitch_col in comp_agg.columns else comp_agg
            if tr.empty or pool.empty:
                continue
            trow=tr.iloc[0]
            rows=[]
            for label,std,names in aliases:
                tc=metric_col(target_agg,names); cc=metric_col(pool,names)
                if not tc or not cc:
                    continue
                tv=num(trow.get(tc))
                pv=pd.to_numeric(pool[cc], errors='coerce').dropna()
                if pd.isna(tv) or pv.empty:
                    continue
                pct=float((pv <= tv).mean()*100)
                unit = " mph" if label == "Velocity" else (" rpm" if label == "Spin" else (" ft" if label in {"Extension","Rel Height","Rel Side"} else " in"))
                rows.append((label, max(0,min(100,round(pct))), fmt(tv,1)+unit))
            if rows:
                out.append((pch, rows[:5]))
        return out

    def ordinal(n):
        try:
            n=int(round(float(n)))
        except Exception:
            return ""
        if 10 <= n % 100 <= 20:
            suffix="th"
        else:
            suffix={1:"st",2:"nd",3:"rd"}.get(n%10,"th")
        return f"{n}{suffix}"

    def reason_lines():
        """Explain the selected pitcher vs the OVERALL top comp only.

        Earlier versions pulled the best per-pitch comp, which made this section
        mention pitchers other than the Top Comp. This version finds the Top Comp
        pitcher inside the comparison aggregate and compares the same pitch types
        directly against the selected pitcher.
        """
        if not top_name:
            return []
        metric_defs = [
            ("Vel", ["Vel"], "velocity", " mph"),
            ("VelMax", ["VelMax"], "max velo", " mph"),
            ("Spin", ["Spin"], "spin", " rpm"),
            ("Extension", ["Extension"], "extension", " ft"),
            ("IVB", ["IndVertBrk", "IVB"], "IVB", '"'),
            ("HB", ["HorzBrk", "HB"], "HB", '"'),
            ("RelHt", ["Rel. Height", "RelHt"], "release height", " ft"),
            ("RSd", ["RSd"], "release side", " ft"),
        ]
        # Use the matched pitches reported in the overall leaderboard when available.
        top_pitch_list = []
        for raw in safe(top_pitches).replace("…", "").split(','):
            cp = canonical_pitch_for_reports(raw.strip())
            if cp and cp not in top_pitch_list:
                top_pitch_list.append(cp)
        if not top_pitch_list:
            top_pitch_list = [p for p in PITCH_ORDER if p in pitch_results and pitch_results.get(p) is not None and not pitch_results.get(p).empty]

        out = []
        for pch in top_pitch_list[:4]:
            tr = target_agg[target_agg[target_pitch_col].astype(str).str.lower() == str(pch).lower()]
            cr = comp_agg[
                (comp_agg[comp_pitch_col].astype(str).str.lower() == str(pch).lower()) &
                (comp_agg[comp_pitcher_col].astype(str).str.lower() == str(top_name).lower())
            ]
            if tr.empty or cr.empty:
                continue
            trow = tr.iloc[0]
            crow = cr.iloc[0]
            diffs = []
            for _, aliases, label, unit in metric_defs:
                tc = metric_col(target_agg, aliases)
                cc = metric_col(comp_agg, aliases)
                if not tc or not cc:
                    continue
                tv = num(trow.get(tc))
                cv = num(crow.get(cc))
                if pd.isna(tv) or pd.isna(cv):
                    continue
                gap = abs(float(tv) - float(cv))
                # Spin gaps read too large next to shape/slot gaps; still allow if it is best.
                diffs.append((gap, label, unit))
            if diffs:
                diffs.sort(key=lambda x: x[0])
                best = diffs[:2]
                parts = []
                for gap, label, unit in best:
                    if unit == " rpm":
                        parts.append(f"{label} gap {gap:.0f}{unit}")
                    else:
                        parts.append(f"{label} gap {gap:.1f}{unit}")
                out.append(f"{pch}: {top_name} — " + "; ".join(parts))
        if out:
            return out[:4]
        return [f"Closest overall match: {top_name}", f"Matched pitch types: {safe(top_pitches) or 'available arsenal'}"]

    def draw_percentile_card(x,y,w,h):
        card(x,y,w,h,"Pitch percentile rankings", "(vs Same Pitch Type)")
        groups = pitch_percentiles()
        if not groups:
            draw_text(x+18, y+h/2, "No percentile data available.", 12, muted, maxw=w-36)
            return

        # Wide percentile card: colored pitch headers, centered pitch names, and no bars.
        # Use fixed sub-columns inside each pitch group so the value and percentile never bleed
        # into the neighboring pitch group. This was the main cause of the crowded look.
        n = min(len(groups), 4)
        gap = 22
        left_pad = 18
        right_pad = 18
        col_w = (w - left_pad - right_pad - gap*(n-1)) / n
        base_x = x + left_pad
        top_y = y + h - 58
        row_h = 26
        for idx, (pch, rows) in enumerate(groups[:n]):
            px0 = base_x + idx*(col_w+gap)
            col = C(pitch_colors.get(pch,"#9ca3af"),"#9ca3af")

            # Pitch header bar: centered label with a little breathing room on each side.
            c.setFillColor(col)
            c.roundRect(px0, top_y-10, col_w, 28, 5, fill=1, stroke=0)
            draw_text(px0 + col_w/2, top_y, pch.upper(), 10.2, colors.white, True, align="center", maxw=col_w-14)

            metric_x = px0 + 8
            # Keep the percentile tucked inside its pitch card, not riding the edge.
            pct_x = px0 + col_w - 14
            # Put value safely between the metric and percentile columns.
            value_x = px0 + col_w * 0.52
            value_max = max(38, col_w * 0.25)
            yy = top_y - 32
            for ridx, (lab,pct,val) in enumerate(rows[:5]):
                if ridx % 2 == 1:
                    c.setFillColor(colors.HexColor("#fbfdff"))
                    c.rect(px0, yy-5, col_w, row_h, fill=1, stroke=0)

                # Smaller, tighter typography in this card so four pitches can fit cleanly.
                draw_text(metric_x, yy+4, lab, 7.4, textc, maxw=col_w*0.32)
                draw_text(value_x, yy+4, val, 7.3, navy, align="center", maxw=value_max)
                draw_text(pct_x, yy+4, ordinal(pct), 7.8, blue, True, align="right")

                c.setStrokeColor(line2); c.setLineWidth(.35); c.line(px0+3, yy-7, px0+col_w-3, yy-7)
                yy -= row_h
        draw_text(x+18, y+18, "Each pitch is ranked against only pitchers with that same pitch type in the comparison pool.", 7.4, muted, maxw=w-36)

    # --- Data -----------------------------------------------------------------
    bio = target_bio or {}
    score, top_name, top_pitches = best_score()
    usage = usage_rows()
    top_name = top_name or "No comp"

    # --- Page background -------------------------------------------------------
    c.setFillColor(colors.white); c.rect(0,0,W,H,fill=1,stroke=0)
    c.setStrokeColor(line); c.setLineWidth(1.2); c.roundRect(1.5,1.5,W-3,H-3,7,fill=0,stroke=1)

    # --- Top identity + summary band -----------------------------------------
    draw_text(28, 984, "PITCHER SIMILARITY REPORT", 15, navy, True)

    # Player photo / placeholder
    photo_x, photo_y, photo_w, photo_h = 30, 770, 170, 190
    if headshot_bytes and draw_img(headshot_bytes, photo_x, photo_y, photo_w, photo_h):
        c.setStrokeColor(line); c.roundRect(photo_x, photo_y, photo_w, photo_h, 8, fill=0, stroke=1)
    else:
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#b7c2d0")); c.setDash(4,3); c.roundRect(photo_x, photo_y, photo_w, photo_h, 9, fill=0, stroke=1); c.setDash()
        c.setStrokeColor(colors.HexColor("#aab3c2")); c.setLineWidth(3)
        c.circle(photo_x+photo_w/2, photo_y+118, 15, fill=0, stroke=1)
        c.arc(photo_x+58, photo_y+68, photo_x+112, photo_y+122, 25, 155)
        draw_text(photo_x+photo_w/2, photo_y+65, "ADD PHOTO", 13, colors.HexColor("#7b8797"), True, align="center")

    # Player block — intentionally wider and less crowded than prior versions.
    px = 238
    draw_text(px, 920, target_name, 48, navy, True, maxw=380, leading=50)
    throws=safe(bio.get("Throws")); pos=safe(bio.get("Position"),"Pitcher")
    handed = f"{throws}HP" if throws in {"R","L"} else (throws or "")
    raw_age = safe(bio.get("Age", ""), "")
    age_val = raw_age if raw_age and raw_age != "—" else ""
    subtitle = f"{handed}   |   {pos if pos!='P' else 'Pitcher'}" + (f"   |   Age {age_val}" if age_val else "")
    draw_text(px, 878, subtitle, 17, navy, False, maxw=390)

    # Optional logo, then a clean two-row bio grid so Team/Level never run outside the card.
    if logo_bytes:
        draw_img(logo_bytes, px, 800, 44, 44)
    detail_x = px + (62 if logo_bytes else 0)
    detail_y = 806
    details = [
        ("ORG", safe(bio.get("Org", "TEX"), "TEX").upper(), 62),
        ("TEAM", safe(bio.get("Team", ""), "—"), 172),
        ("LEVEL", safe(bio.get("Level", ""), "—").upper(), 64),
        ("THROWS", safe(bio.get("Throws", ""), "—").upper(), 70),
    ]
    if age_val:
        details.append(("AGE", age_val, 54))
    cx = detail_x
    for lab, val, bw in details:
        c.setStrokeColor(line); c.setLineWidth(.8); c.line(cx-8, detail_y-5, cx-8, detail_y+52)
        draw_text(cx, detail_y+34, lab, 8.5, muted, True)
        max_chars = 21 if lab == "TEAM" else 8
        draw_text(cx, detail_y+9, shorten(val, max_chars), 13, navy, False, maxw=bw, leading=14)
        cx += bw

    draw_text(px, 770, f"Generated {datetime.now().strftime('%b %d, %Y')}  •  One-page pitcher comp report", 10, muted)

    # Vertical separators in top band
    for sx in [665, 848, 1110]:
        c.setStrokeColor(line); c.setLineWidth(1.1); c.line(sx, 764, sx, 996)

    # Score gauge - drawn natively so the score never disappears if matplotlib is unavailable
    def draw_native_gauge(cx, cy, r, score_value):
        sc = 0 if pd.isna(score_value) else max(0, min(100, float(score_value)))
        c.setLineCap(1)
        c.setStrokeColor(line)
        c.setLineWidth(16)
        c.circle(cx, cy, r, fill=0, stroke=1)
        c.setStrokeColor(gold)
        # draw arc clockwise from top; reportlab arc uses degrees on bounding box
        c.arc(cx-r, cy-r, cx+r, cy+r, 90, -360*sc/100)
        c.setLineWidth(1)
        draw_text(cx, cy+14, f"{sc:.1f}", 38, navy, True, align="center")
        draw_text(cx, cy-24, "SIMILARITY", 10, navy, True, align="center")

    draw_text(686, 984, "TOP COMP SCORE", 15, navy, True)
    draw_native_gauge(756, 870, 64, score)

    # Top comp + why
    draw_text(878, 984, "TOP COMP", 15, navy, True)
    draw_text(878, 938, top_name, 22, navy, True, maxw=210, leading=24)
    draw_text(878, 910, top_pitches, 12, textc, maxw=210, leading=14)
    draw_text(878, 862, "WHY THIS COMP?", 13, navy, True)
    yy=832
    for r in reason_lines()[:5]:
        c.setFillColor(green); c.circle(882, yy+5, 5, fill=1, stroke=0)
        draw_text(896, yy, r, 10.2, textc, maxw=190, leading=14)
        yy -= 28 if len(safe(r)) < 50 else 42
        if yy < 775:
            break

    # Pitch usage donut. Draw directly with ReportLab so it never disappears and
    # always completes a full 360-degree donut. Percentages are actual usage pct.
    def draw_usage_donut(cx, cy, radius, rows):
        data=[]
        for rr in rows[1:]:
            pitch=safe(rr[0])
            try:
                count=float(str(rr[1]).replace(',', '').strip())
            except Exception:
                count=0.0
            try:
                pct=float(str(rr[2]).replace('%','').strip())
            except Exception:
                pct=0.0
            if count > 0 and pitch in PITCH_ORDER:
                data.append((pitch, count, pct))
        if not data:
            return False
        total=sum(v for _,v,_ in data) or 1.0
        # Draw every non-zero pitch using its pitch-specific color. Tiny pitches
        # remain thin slivers, but labels only show for readable slices.
        start=90.0
        for pitch,count,pct in data:
            extent = -360.0 * count / total
            c.setFillColor(C(pitch_colors.get(pitch, '#9ca3af'), '#9ca3af'))
            c.setStrokeColor(colors.white)
            c.setLineWidth(1.5)
            c.wedge(cx-radius, cy-radius, cx+radius, cy+radius, start, extent, fill=1, stroke=1)
            # Label the slice only if it is large enough to be legible.
            true_pct = count / total * 100
            if true_pct >= 5:
                import math
                mid = math.radians(start + extent/2.0)
                tx = cx + math.cos(mid) * radius * 0.63
                ty = cy + math.sin(mid) * radius * 0.63
                draw_text(tx, ty-5, f"{true_pct:.0f}%", 13, colors.white, True, align="center")
            start += extent
        # Donut hole
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.white)
        c.circle(cx, cy, radius*0.38, fill=1, stroke=0)
        return True

    draw_text(1142, 984, "PITCH USAGE", 15, navy, True)
    draw_usage_donut(1206, 861, 70, usage)
    draw_text(1312, 948, "PITCH", 8, muted, True); draw_text(1405, 948, "P", 8, muted, True); draw_text(1488, 948, "USAGE", 8, muted, True, align="right")
    yy=923
    for r in usage[1:8]:
        p=safe(r[0]); col=C(pitch_colors.get(p,"#9ca3af"),"#9ca3af")
        c.setFillColor(col); c.circle(1316, yy+5, 5, fill=1, stroke=0)
        draw_text(1330, yy, shorten(p,14), 11, textc)
        draw_text(1405, yy, safe(r[1]), 11, textc)
        draw_text(1488, yy, safe(r[2]), 11, textc, align="right")
        yy -= 27

    # --- Middle cards ---------------------------------------------------------
    card(28, 405, 418, 315, "Overall top 5 similar pitchers")
    o_rows=[]
    if overall is not None and not overall.empty:
        for i,(_,r) in enumerate(overall.head(5).iterrows(),1):
            sc=r.get("Similarity Score",r.get("Avg_Similarity",np.nan))
            o_rows.append([i,shorten(r.get(comp_pitcher_col,r.get("Pitcher","")),22),fmt(sc,1),shorten(r.get("Matched_Pitches",""),18)])
    draw_table_lines(46, 666, 382, ["Rank","Pitcher","Similarity","Matched Pitches"], o_rows, [.14,.42,.22,.22], row_h=42, fs=12, header_fs=9, header_fill=navy, blue_col=2)

    # Narrower movement card so pitch percentile rankings have more room, matching the dashboard mockup.
    card(458, 405, 280, 315, "Movement plot", "(IVB vs HB)")

    def draw_native_movement(x, y, w, h):
        hx_t = metric_col(target_agg, ["HorzBrk","HB","HorizontalBreak","Horz Break","Horizontal Break"])
        vy_t = metric_col(target_agg, ["IndVertBrk","IVB","InducedVertBreak","VerticalBreak","Induced Vertical Break"])
        hx_c = metric_col(comp_agg, ["HorzBrk","HB","HorizontalBreak","Horz Break","Horizontal Break"])
        vy_c = metric_col(comp_agg, ["IndVertBrk","IVB","InducedVertBreak","VerticalBreak","Induced Vertical Break"])
        if not (hx_t and vy_t and hx_c and vy_c):
            return False
        rows=[]
        for pch in PITCH_ORDER:
            tr = target_agg[target_agg[target_pitch_col].astype(str).str.lower()==str(pch).lower()]
            if tr.empty:
                continue
            # Prefer the closest pitch comp row. Fall back to the overall top comp for that pitch.
            res = pitch_results.get(pch)
            crow = None
            if res is not None and not res.empty:
                crow = res.iloc[0]
            if crow is None and top_name:
                pool = comp_agg[(comp_agg[comp_pitch_col].astype(str).str.lower()==str(pch).lower()) & (comp_agg[comp_pitcher_col].astype(str)==str(top_name))]
                if not pool.empty:
                    crow = pool.iloc[0]
            if crow is None:
                continue
            vals = pd.to_numeric(pd.Series([tr.iloc[0].get(hx_t), tr.iloc[0].get(vy_t), crow.get(hx_c), crow.get(vy_c)]), errors='coerce')
            if vals.isna().any():
                continue
            rows.append((pch, float(vals.iloc[0]), float(vals.iloc[1]), float(vals.iloc[2]), float(vals.iloc[3])))
        if not rows:
            return False
        xs=[v for row in rows for v in (row[1],row[3])]; ys=[v for row in rows for v in (row[2],row[4])]
        xmin=min(-22, min(xs)-5); xmax=max(22, max(xs)+5); ymin=min(-20, min(ys)-5); ymax=max(25, max(ys)+5)
        def sx(v): return x + (v-xmin)/(xmax-xmin)*w
        def sy(v): return y + (v-ymin)/(ymax-ymin)*h
        c.setStrokeColor(line); c.setLineWidth(1); c.rect(x,y,w,h,fill=0,stroke=1)
        c.setStrokeColor(colors.HexColor("#cbd5e1")); c.setDash(4,3)
        if xmin < 0 < xmax: c.line(sx(0), y, sx(0), y+h)
        if ymin < 0 < ymax: c.line(x, sy(0), x+w, sy(0))
        c.setDash()
        # Axis tick labels so the plot reads like the mockup.
        c.setStrokeColor(line); c.setFillColor(muted); c.setFont("Helvetica", 8)
        for tick in [-20, -10, 0, 10, 20]:
            if xmin <= tick <= xmax:
                txp = sx(tick)
                c.setStrokeColor(line2); c.line(txp, y, txp, y-4)
                c.drawCentredString(txp, y-16, str(tick))
        for tick in [-20, -10, 0, 10, 20]:
            if ymin <= tick <= ymax:
                typ = sy(tick)
                c.setStrokeColor(line2); c.line(x-4, typ, x, typ)
                c.drawRightString(x-8, typ-3, str(tick))
        for pch, tx, ty, cxp, cyp in rows:
            col=C(pitch_colors.get(pch,"#9ca3af"),"#9ca3af")
            c.setStrokeColor(line); c.setLineWidth(1); c.line(sx(tx), sy(ty), sx(cxp), sy(cyp))
            c.setFillColor(colors.white); c.setStrokeColor(col); c.setLineWidth(2.2); c.circle(sx(tx), sy(ty), 7, fill=1, stroke=1)
            c.setFillColor(navy); c.setStrokeColor(navy); c.circle(sx(cxp), sy(cyp), 6, fill=1, stroke=0)
            draw_text(sx(tx)+10, sy(ty)+8, pitch_abbr.get(pch,pch[:2].upper()), 10, navy, True)
        draw_text(x, y+h+24, "○ Target", 10, accent)
        draw_text(x+82, y+h+24, f"● Closest Comp ({shorten(top_name,22)})", 10, navy)
        draw_text(x+w/2, y-28, "HORIZONTAL BREAK (IN.)", 9, navy, align="center")
        draw_text(x-35, y+h/2, "IVB", 9, navy)
        return True

    if draw_native_movement(492, 484, 200, 166):
        draw_text(474, 430, "Plot shows average movement. More toward top = more rise. More right = more arm-side run.", 7.1, muted, maxw=250)
    else:
        draw_text(598, 560, "Needs HB/IVB movement columns.", 12, muted, align="center")

    draw_percentile_card(752, 405, 756, 315)

    # --- Bottom pitch cards ---------------------------------------------------
    # Keep EVERYTHING on one polished page: up to four pitch cards fit across
    # the bottom row. Each card includes top comps + compact metric comparison.
    matched={p:r for p,r in pitch_results.items() if r is not None and not r.empty}
    shown=[p for p in PITCH_ORDER if p in matched][:4]
    bottom_y=88; bottom_h=280
    gap=14
    card_w=(W-56-gap*(max(1,len(shown))-1))/max(1,len(shown)) if shown else 0
    if card_w > 360:
        card_w = 360
    for idx,pch in enumerate(shown):
        x=28+idx*(card_w+gap)
        card(x,bottom_y,card_w,bottom_h,None)
        col=C(pitch_colors.get(pch,accent_color),accent_color)
        c.setFillColor(col); c.circle(x+22,bottom_y+bottom_h-28,10,fill=1,stroke=0)
        draw_text(x+40,bottom_y+bottom_h-34,f"{pch.upper()} SIMILARITY",13,navy,True, maxw=card_w-50)
        res=matched[pch]
        tr=target_agg[target_agg[target_pitch_col].astype(str).str.lower()==str(pch).lower()]
        comp_name = safe(res.iloc[0].get(comp_pitcher_col,"Comp")) if not res.empty else "Comp"

        # Left mini-table: top 5 comps
        left_x=x+16
        left_w=card_w*0.43
        right_x=x+left_w+30
        right_w=card_w-left_w-46
        draw_text(left_x,bottom_y+bottom_h-58,f"TOP 5 MOST SIMILAR",9.2,navy,True, maxw=left_w)
        rows=[]
        for i,(_,r) in enumerate(res.head(5).iterrows(),1):
            rows.append([i,shorten(r.get(comp_pitcher_col,""),16),fmt(r.get("Similarity Score"),1)])
        draw_table_lines(left_x,bottom_y+bottom_h-78,left_w,["#","Pitcher","Sim"],rows,[.16,.58,.26],row_h=27,fs=8.5,header_fs=7.2,header_fill=col,blue_col=2)

        # Divider
        c.setStrokeColor(line); c.setLineWidth(.8); c.line(right_x-12,bottom_y+28,right_x-12,bottom_y+bottom_h-62)

        # Right mini-table: metric comparison
        draw_text(right_x,bottom_y+bottom_h-58,f"METRIC COMPARISON",9.2,navy,True, maxw=right_w)
        if not tr.empty and not res.empty:
            trow=tr.iloc[0]; crow=res.iloc[0]
            preferred=[("Vel","Velo"),("VelMax","Max"),("Spin","Spin"),("Extension","Ext"),("IndVertBrk","IVB"),("HorzBrk","HB"),("Rel. Height","RelHt")]
            mrows=[]
            for m,label in preferred:
                if m in trow.index and m in crow.index:
                    tv=num(trow.get(m)); cv=num(crow.get(m))
                    diff="" if pd.isna(tv) or pd.isna(cv) else f"{tv-cv:+.1f}"
                    mrows.append([label,fmt(tv,1),fmt(cv,1),diff])
            draw_table_lines(right_x,bottom_y+bottom_h-78,right_w,["Metric","Tgt","Comp","Diff"],mrows[:7],[.33,.21,.24,.22],row_h=22,fs=7.2,header_fs=6.5,header_fill=col,blue_col=None)

    draw_text(26, 48, "Method: pitch types with no matches are excluded. Metrics are averaged by pitcher/pitch type, standardized by comparison pool, and ranked by normalized distance.", 8.5, muted)
    draw_text(26, 30, "Similarity = 100 / (1 + distance).", 8.5, muted)

    c.save(); buf.seek(0); return buf.getvalue()

st.write("Upload one target-pitcher CSV and one or more comparison CSVs. The app compares pitch characteristics by pitch type and returns the closest matches.")

with st.sidebar:
    st.header("Uploads")
    target_file = st.file_uploader("Target pitcher CSV", type=["csv"], key="target")
    comp_files = st.file_uploader("Comparison CSVs", type=["csv"], accept_multiple_files=True, key="comparison")
    top_n = st.number_input("Number of matches to show", min_value=1, max_value=25, value=5, step=1)
    min_overlap = st.number_input("Minimum pitch types for overall ranking", min_value=1, max_value=8, value=1, step=1)
    level_filter = st.selectbox("Comparison pool", ["All Players", "MLB Players Only", "MiLB Players Only"], index=0)
    min_pitch_count = st.selectbox("Minimum pitches thrown", [1, 20, 50, 100], index=0)
    logo_file = st.file_uploader("Optional team logo for reports", type=["png", "jpg", "jpeg"], key="logo")
    headshot_file = st.file_uploader("Optional pitcher headshot", type=["png", "jpg", "jpeg"], key="headshot")
    headshot_url_input = st.text_input("Optional headshot URL", value="", help="Paste a MiLB/MLB image URL if auto-fetch is blocked on your network.")
    auto_headshot = st.checkbox("Try MiLB/MLB headshot from playerId", value=True)
    st.markdown("**Report style**")
    report_primary_color = st.color_picker("Primary report color", "#0b1f3a")
    report_accent_color = st.color_picker("Accent report color", "#c1121f")
    force_one_page_report = st.checkbox("Keep PDF to 1 page", value=True, help="When checked, the PDF shrinks content to stay on one page.")

if not target_file or not comp_files:
    st.info("Upload your target pitcher CSV and at least one comparison CSV to begin.")
    st.stop()

target_df = read_csv(target_file)
comp_frames = []
unknown_pitch_files = []
for f in comp_files:
    pitch_from_file = infer_pitch_from_filename(f.name)
    if pitch_from_file is None:
        unknown_pitch_files.append(f.name)
    comp_frames.append(
        read_csv(f).assign(
            SourceFile=f.name,
            PitchType_From_File=pitch_from_file,
        )
    )
comp_df = pd.concat(comp_frames, ignore_index=True)
logo_bytes = logo_file.getvalue() if logo_file is not None else None
comp_df = filter_comp_pool(comp_df, level_filter, int(min_pitch_count))
if comp_df.empty:
    st.error("No comparison rows remain after applying filters. Try lowering the minimum pitch count or using All Players.")
    st.stop()

st.subheader("Column Mapping")
all_target_cols = list(target_df.columns)
all_comp_cols = list(comp_df.columns)

def metric_default(df_cols, metric):
    return guess_column(df_cols, [metric, metric.replace(".", ""), metric.replace(" ", ""), metric.replace(".", "").replace(" ", "")])

t_guess_pitcher = guess_column(all_target_cols, ["playerFullName", "playerId", "Pitcher", "Pitcher Name", "Player", "Player Name", "Name", "pitcher_name", "splitByName"])
c_guess_pitcher = guess_column(all_comp_cols, ["playerFullName", "Pitcher", "Pitcher Name", "Player", "Player Name", "Name", "pitcher_name", "playerId"])
t_guess_pitch = guess_column(all_target_cols, ["Pitch Type", "PitchType", "TaggedPitchType", "AutoPitchType", "Pitch", "pitch_type"])
c_guess_pitch = "PitchType_From_File" if "PitchType_From_File" in all_comp_cols else guess_column(all_comp_cols, ["Pitch Type", "PitchType", "TaggedPitchType", "AutoPitchType", "Pitch", "pitch_type"])

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Target CSV**")
    target_pitcher_col = st.selectbox("Target pitcher name column", all_target_cols, index=all_target_cols.index(t_guess_pitcher) if t_guess_pitcher in all_target_cols else 0)
    target_pitch_col = st.selectbox("Target pitch type column", all_target_cols, index=all_target_cols.index(t_guess_pitch) if t_guess_pitch in all_target_cols else 0)
with col2:
    st.markdown("**Comparison CSVs**")
    comp_pitcher_col = st.selectbox("Comparison pitcher name column", all_comp_cols, index=all_comp_cols.index(c_guess_pitcher) if c_guess_pitcher in all_comp_cols else 0)
    comp_pitch_col = st.selectbox(
        "Comparison pitch type source",
        all_comp_cols,
        index=all_comp_cols.index(c_guess_pitch) if c_guess_pitch in all_comp_cols else 0,
        help="Use PitchType_From_File when each comparison file is named like 'Fastball Pitch Characteristics.csv'.",
    )

if unknown_pitch_files and comp_pitch_col == "PitchType_From_File":
    st.warning(
        "I could not detect a pitch type from these comparison file names: "
        + ", ".join(unknown_pitch_files)
        + ". Rename them with Fastball, Sinker, Cutter, Slider, Curveball, Change, Splitter, or Sweeper in the file name."
    )

st.markdown("**Metric columns**")
metric_map = {}
metric_cols = st.columns(4)
for i, metric in enumerate(DEFAULT_METRICS):
    with metric_cols[i % 4]:
        guess = metric_default(all_target_cols, metric)
        options = ["-- skip --"] + all_target_cols
        default_index = options.index(guess) if guess in options else 0
        selected = st.selectbox(f"{metric}", options, index=default_index, key=f"metric_{metric}")
        if selected != "-- skip --":
            metric_map[metric] = selected

if not metric_map:
    st.error("Select at least one metric column.")
    st.stop()

# Rename selected target metrics to standard metric names and find matching comp columns by same standard/default guesses.
target_use = target_df.rename(columns={v: k for k, v in metric_map.items()}).copy()
comp_metric_renames = {}
missing_comp = []
for metric in metric_map.keys():
    guess = metric_default(all_comp_cols, metric)
    if guess:
        comp_metric_renames[guess] = metric
    else:
        missing_comp.append(metric)

if missing_comp:
    st.error("Could not find these metric columns in the comparison CSVs: " + ", ".join(missing_comp))
    st.stop()

comp_use = comp_df.rename(columns=comp_metric_renames).copy()
metrics = list(metric_map.keys())


def is_good_display_name(value) -> bool:
    """True when a value looks like an actual person name, not an ID/header/group label."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    bad = {
        "nan", "none", "null", "total", "average", "rank", "playerid",
        "player id", "splitby", "split by", "pitch type", "pitcher"
    }
    if text.lower() in bad:
        return False
    if re.fullmatch(r"\d+(\.0)?", text):
        return False
    # A real name should normally contain letters.
    return any(ch.isalpha() for ch in text)


def first_good_name_from_rows(rows: pd.DataFrame) -> Optional[str]:
    preferred = [
        "playerFullName", "Pitcher", "Pitcher Name", "Player", "Player Name",
        "Name", "playerName", "fullName", "abbrevName", "splitByName"
    ]
    for cand in preferred:
        col = guess_column(list(rows.columns), [cand])
        if not col:
            continue
        vals = rows[col].dropna().astype(str).str.strip()
        vals = vals[vals.apply(is_good_display_name)]
        if not vals.empty:
            return str(vals.iloc[0])
    return None


def get_target_display_name(
    df: pd.DataFrame,
    selected_value: str,
    selected_col: str,
    comp_df_for_lookup: Optional[pd.DataFrame] = None,
    target_id_keys_for_lookup: Optional[set] = None,
) -> str:
    """Return the best human-readable name for report headers.

    Some target exports have playerId but no full name, and some have a bogus
    splitByName value like "SplitBy". In that case, look up the same playerId
    in the comparison CSVs and pull playerFullName before falling back.
    """
    try:
        selected_text = str(selected_value).strip()
        rows = df[df[selected_col].astype(str).str.strip() == selected_text].copy()
        if rows.empty:
            rows = df.copy()
        found = first_good_name_from_rows(rows)
        if found:
            return found

        if comp_df_for_lookup is not None and target_id_keys_for_lookup:
            id_mask = pd.Series(False, index=comp_df_for_lookup.index)
            for col in possible_id_columns(list(comp_df_for_lookup.columns)):
                if col in comp_df_for_lookup.columns:
                    id_mask = id_mask | comp_df_for_lookup[col].apply(normalize_identifier).isin(target_id_keys_for_lookup)
            comp_rows = comp_df_for_lookup.loc[id_mask].copy()
            found = first_good_name_from_rows(comp_rows)
            if found:
                return found
    except Exception:
        pass
    return str(selected_value) if is_good_display_name(selected_value) else "Target Pitcher"



# -----------------------------------------------------------------------------
# Aggregation + similarity helpers
# -----------------------------------------------------------------------------
def prepare_agg(df: pd.DataFrame, pitcher_col: str, pitch_col: str, metrics: List[str]) -> pd.DataFrame:
    """Aggregate rows by pitcher and pitch type.

    Supports both pitch-level CSVs and already-aggregated pitch characteristic
    CSVs. If a pitch-count column exists, it is preserved as PitchCount;
    otherwise each row counts as one pitch.
    """
    count_col = guess_column(list(df.columns), ["P", "PitchCount", "pitchCount", "Pitches", "pitches", "Count", "pitch_count"])
    base_cols = [pitcher_col, pitch_col] + [m for m in metrics if m in df.columns]
    keep_cols = base_cols.copy()
    if count_col and count_col not in keep_cols:
        keep_cols.append(count_col)

    work = df[keep_cols].copy()
    work[pitch_col] = work[pitch_col].apply(canonical_pitch_for_reports)
    work = work[work[pitch_col].isin(PITCH_ORDER)].copy()
    work = coerce_numeric(work, [m for m in metrics if m in work.columns])

    if count_col and count_col in work.columns:
        work["PitchCount"] = pd.to_numeric(work[count_col], errors="coerce").fillna(0)
        work.loc[work["PitchCount"] <= 0, "PitchCount"] = 1
    else:
        work["PitchCount"] = 1

    work = work.dropna(subset=[pitcher_col, pitch_col])
    work = work[work[pitch_col].astype(str).str.strip().ne("")]
    agg_dict = {m: "mean" for m in metrics if m in work.columns}
    agg_dict["PitchCount"] = "sum"
    return work.groupby([pitcher_col, pitch_col], as_index=False).agg(agg_dict)


def compute_pitch_similarity(
    target_agg: pd.DataFrame,
    comp_agg: pd.DataFrame,
    target_pitcher_col: str,
    comp_pitcher_col: str,
    target_pitch_col: str,
    comp_pitch_col: str,
    metrics: List[str],
    target_name: str,
    pitch_type: str,
    top_n: int = 5,
) -> pd.DataFrame:
    target_rows = target_agg[target_agg[target_pitch_col] == pitch_type]
    comp_rows = comp_agg[comp_agg[comp_pitch_col] == pitch_type].copy()
    if target_rows.empty or comp_rows.empty:
        return pd.DataFrame()

    t = target_rows.iloc[0]
    metric_cols = [m for m in metrics if m in target_rows.columns and m in comp_rows.columns and not pd.isna(t.get(m))]
    if not metric_cols:
        return pd.DataFrame()

    comp_rows = comp_rows.dropna(subset=metric_cols, how="all").copy()
    if comp_rows.empty:
        return pd.DataFrame()

    means = comp_rows[metric_cols].mean(skipna=True)
    stds = comp_rows[metric_cols].std(skipna=True).replace(0, np.nan).fillna(1)
    target_z = (t[metric_cols] - means) / stds
    comp_z = (comp_rows[metric_cols] - means) / stds
    diffs = comp_z.subtract(target_z, axis=1)

    squared = np.square(diffs.to_numpy(dtype=float))
    valid_counts = np.sum(~np.isnan(squared), axis=1)
    sum_squared = np.nansum(squared, axis=1)
    distances = np.full(len(comp_rows), np.nan, dtype=float)
    valid_rows = valid_counts > 0
    distances[valid_rows] = np.sqrt(sum_squared[valid_rows] / valid_counts[valid_rows])

    comp_rows["Distance"] = distances
    comp_rows["Similarity Score"] = 100 / (1 + comp_rows["Distance"])
    comp_rows["Pitch Type"] = pitch_type
    comp_rows["Metrics Used"] = valid_counts
    comp_rows = comp_rows.dropna(subset=["Distance"])
    if comp_rows.empty:
        return pd.DataFrame()
    return comp_rows.sort_values("Distance").head(top_n)


def compute_all(target_agg, comp_agg, target_cols, comp_cols, metrics, target_name, top_n, min_overlap):
    t_pitcher_col, t_pitch_col = target_cols
    c_pitcher_col, c_pitch_col = comp_cols
    pitch_types = [p for p in PITCH_ORDER if p in set(target_agg[t_pitch_col])]
    pitch_results = {}
    all_pitch_distances = []

    for pitch in pitch_types:
        res = compute_pitch_similarity(target_agg, comp_agg, t_pitcher_col, c_pitcher_col, t_pitch_col, c_pitch_col, metrics, target_name, pitch, top_n)
        pitch_results[pitch] = res
        full = compute_pitch_similarity(target_agg, comp_agg, t_pitcher_col, c_pitcher_col, t_pitch_col, c_pitch_col, metrics, target_name, pitch, top_n=10000)
        if not full.empty:
            cols = [c_pitcher_col, "Pitch Type", "Distance", "Similarity Score"]
            if "PitchCount" in full.columns:
                cols.append("PitchCount")
            all_pitch_distances.append(full[cols])

    if not all_pitch_distances:
        return pitch_results, pd.DataFrame()

    stacked = pd.concat(all_pitch_distances, ignore_index=True)
    overall = (
        stacked.groupby(c_pitcher_col)
        .agg(
            Avg_Distance=("Distance", "mean"),
            Median_Distance=("Distance", "median"),
            Avg_Similarity=("Similarity Score", "mean"),
            Pitch_Types_Matched=("Pitch Type", "nunique"),
            Matched_Pitches=("Pitch Type", lambda x: ", ".join([p for p in PITCH_ORDER if p in set(x)])),
        )
        .reset_index()
    )
    overall = overall[overall["Pitch_Types_Matched"] >= min_overlap]
    if overall.empty:
        return pitch_results, overall
    overall = overall.sort_values(["Avg_Distance", "Pitch_Types_Matched"], ascending=[True, False]).head(top_n)
    overall["Similarity Score"] = 100 / (1 + overall["Avg_Distance"])
    return pitch_results, overall

# Target pitcher identity and self-removal
target_names = sorted(target_use[target_pitcher_col].dropna().astype(str).unique())
selected_target_name = st.selectbox("Target pitcher to exclude from comparison", target_names, index=0 if target_names else None)

# Build a stronger set of target identifiers. This removes self-comparisons by
# playerId first, then by every available target name field. This matters when
# the target CSV only has an ID/name abbreviation while the comparison CSV has
# playerFullName.
target_identity_values = set()
for col in [target_pitcher_col] + possible_name_columns(list(target_use.columns)):
    if col in target_use.columns:
        target_identity_values.update(target_use[col].dropna().astype(str).tolist())
target_identity_values.add(str(selected_target_name))
target_identity_keys = {normalize_person_name(v) for v in target_identity_values if normalize_person_name(v)}

target_id_values = set()
for col in possible_id_columns(list(target_use.columns)):
    if col in target_use.columns:
        target_id_values.update(target_use[col].dropna().astype(str).tolist())
target_id_values.add(str(selected_target_name))
target_id_keys = {normalize_identifier(v) for v in target_id_values if normalize_identifier(v)}

# Report header name: use target CSV full name when present; otherwise use
# comparison CSV playerFullName matched by playerId. This is also used to build
# the MiLB page URL for automatic headshot lookup.
target_display_guess = get_target_display_name(
    target_use,
    selected_target_name,
    target_pitcher_col,
    comp_df_for_lookup=comp_use,
    target_id_keys_for_lookup=target_id_keys,
)

headshot_name_candidates = [target_display_guess, selected_target_name]
for col in possible_name_columns(list(target_use.columns)):
    if col in target_use.columns:
        headshot_name_candidates.extend(target_use[col].dropna().astype(str).head(5).tolist())

headshot_bytes = headshot_file.getvalue() if headshot_file is not None else None
if headshot_bytes is None and str(headshot_url_input or "").strip():
    HEADSHOT_DEBUG.clear()
    headshot_bytes = _download_image_bytes(
        str(headshot_url_input).strip(),
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": "image/png,image/jpeg,image/*;q=0.8,*/*;q=0.5",
            "Referer": "https://www.milb.com/",
        },
    )
    if headshot_bytes:
        st.caption("Headshot loaded from the URL you entered.")
    else:
        st.warning("Could not load that headshot URL. Try opening it in your browser first, then paste the final image URL.")
if headshot_bytes is None and auto_headshot and target_id_keys:
    HEADSHOT_DEBUG.clear()
    with st.spinner("Trying to fetch pitcher headshot from MiLB player page / MLB image service..."):
        headshot_bytes = get_auto_headshot_bytes(target_id_keys, headshot_name_candidates)
    if headshot_bytes:
        st.caption("Headshot detected from the player page/playerId and will be used in the report.")
    else:
        st.caption("No auto headshot found. Your network may be blocking mlbstatic.com image downloads, or the player may not have a public headshot.")
        if target_id_keys:
            sample_id = sorted(target_id_keys)[0]
            st.code(f"https://img.mlbstatic.com/mlb-photos/image/upload/c_fill%2Cg_auto/w_180/v1/people/{sample_id}/headshot/milb/current")
        with st.expander("Headshot fetch diagnostics"):
            st.write("Player IDs tried:", sorted(target_id_keys))
            st.write("Name slugs tried:", [slugify_player_name_for_milb(x) for x in headshot_name_candidates if str(x).strip()][:5])
            st.write("Cache folder:", str(_headshot_cache_dir()))
            st.write("Attempts:")
            st.code("\n".join(HEADSHOT_DEBUG[-35:]) or "No attempts recorded")

# If the user uploads/pastes a headshot, cache it under the playerId so future
# reports can auto-use it even if the remote service is intermittent.
if headshot_bytes and target_id_keys:
    try:
        _save_cached_headshot(sorted(target_id_keys)[0], headshot_bytes)
    except Exception:
        pass

target_display_name = st.text_input("Report pitcher name", value=target_display_guess).strip() or "Target Pitcher"
st.caption(f"PDF header will use: {target_display_name}")
target_bio = lookup_target_bio(target_display_name, comp_use, target_id_keys)
# If age/hand/position are not in the CSVs, try the public MLB Stats API using playerId.
try:
    api_bio = fetch_mlb_person_bio(target_id_keys)
    for k, v in api_bio.items():
        if v and not str(target_bio.get(k, "")).strip():
            target_bio[k] = v
except Exception:
    pass
report_age_value = st.text_input("Report age (optional)", value=target_bio.get("Age", "")).strip()
if report_age_value:
    target_bio["Age"] = report_age_value

# Display polish: keep short bio values in report-style uppercase.
# This turns values like mlb/rok/dsl/r into MLB/ROK/DSL/R while leaving the team
# name in normal title case.
for _bio_key in ["Org", "Level", "Throws"]:
    if str(target_bio.get(_bio_key, "")).strip():
        target_bio[_bio_key] = str(target_bio[_bio_key]).strip().upper()

if target_bio:
    st.caption("Report player info detected: " + " • ".join([f"{k}: {v}" for k, v in target_bio.items()]))

comp_before = len(comp_use)
self_mask = pd.Series(False, index=comp_use.index)

# ID-based removal is the most reliable. Example: Eddy Peralta = playerId 807842.
for col in possible_id_columns(list(comp_use.columns)):
    if col in comp_use.columns and target_id_keys:
        self_mask = self_mask | comp_use[col].apply(normalize_identifier).isin(target_id_keys)

# Name-based fallback across all likely name columns.
for col in [comp_pitcher_col] + possible_name_columns(list(comp_use.columns)):
    if col in comp_use.columns and target_identity_keys:
        self_mask = self_mask | comp_use[col].apply(normalize_person_name).isin(target_identity_keys)

removed_examples = []
if self_mask.any():
    example_cols = [c for c in [comp_pitcher_col, "playerFullName", "playerId", "abbrevName", "player"] if c in comp_use.columns]
    removed_examples = comp_use.loc[self_mask, example_cols].drop_duplicates().head(5).astype(str).to_dict("records")

comp_use = comp_use[~self_mask].copy()
removed = comp_before - len(comp_use)

if removed == 0:
    st.warning(
        "No self-comparison rows were removed. If the target pitcher still appears in the results, "
        "check that both files include playerId or that the target name appears in a name column."
    )
else:
    with st.expander("Self-comparison rows removed"):
        st.write(f"Removed {removed:,} rows using target IDs: {', '.join(sorted(target_id_keys)) or 'none'}")
        if removed_examples:
            st.dataframe(pd.DataFrame(removed_examples), use_container_width=True)

try:
    target_agg = prepare_agg(target_use, target_pitcher_col, target_pitch_col, metrics)
    target_agg = target_agg[target_agg[target_pitcher_col].astype(str).str.strip().str.lower() == str(selected_target_name).strip().lower()]
    comp_agg = prepare_agg(comp_use, comp_pitcher_col, comp_pitch_col, metrics)
    if int(min_pitch_count) > 1 and "PitchCount" in comp_agg.columns:
        comp_agg = comp_agg[comp_agg["PitchCount"] >= int(min_pitch_count)].copy()
except Exception as exc:
    st.error(f"Could not process the files: {exc}")
    st.stop()

if target_agg.empty:
    st.error("No target pitcher rows were found after column mapping. Check the pitcher and pitch type columns.")
    st.stop()

# Show which target pitch types were detected so missing pitches are easier to diagnose.
detected_target_pitches = [p for p in PITCH_ORDER if p in set(target_agg[target_pitch_col])]
if detected_target_pitches:
    st.caption("Target pitch types detected: " + ", ".join(detected_target_pitches))

pitch_results, overall = compute_all(
    target_agg,
    comp_agg,
    (target_pitcher_col, target_pitch_col),
    (comp_pitcher_col, comp_pitch_col),
    metrics,
    selected_target_name,
    int(top_n),
    int(min_overlap),
)

st.success(f"Processed {len(target_df):,} target rows and {len(comp_df):,} comparison rows. Removed {removed:,} self-comparison rows for {target_display_name}.")

st.subheader("Top Similar Pitchers Overall")
if overall.empty:
    st.warning("No overall matches found. Try lowering the minimum pitch type overlap or check pitch type names.")
else:
    display_overall = overall.rename(columns={comp_pitcher_col: "Pitcher"})
    st.dataframe(display_overall, use_container_width=True, hide_index=True)

st.subheader("Top Similar Pitchers by Pitch Type")
for pitch in [p for p in PITCH_ORDER if p in pitch_results]:
    res = pitch_results[pitch]
    with st.expander(pitch, expanded=True):
        if res.empty:
            st.write("No matches found for this pitch type.")
        else:
            display = res.rename(columns={comp_pitcher_col: "Pitcher"})
            cols = ["Pitcher", "Pitch Type", "Similarity Score", "Distance", "PitchCount", "Metrics Used"] + metrics
            cols = [c for c in cols if c in display.columns]
            st.dataframe(display[cols], use_container_width=True, hide_index=True)

# Download workbook-style CSV bundle
output = io.StringIO()
if not overall.empty:
    output.write("OVERALL\n")
    overall.rename(columns={comp_pitcher_col: "Pitcher"}).to_csv(output, index=False)
    output.write("\n")
for pitch, res in pitch_results.items():
    output.write(f"{pitch}\n")
    if not res.empty:
        res.rename(columns={comp_pitcher_col: "Pitcher"}).to_csv(output, index=False)
    output.write("\n")

st.download_button(
    "Download similarity results CSV",
    data=output.getvalue(),
    file_name=f"pitcher_similarity_{target_display_name}.csv".replace(" ", "_"),
    mime="text/csv",
)

try:
    pdf_bytes = make_pdf_report(
        target_display_name,
        target_agg,
        comp_agg,
        pitch_results,
        overall,
        target_pitcher_col,
        target_pitch_col,
        comp_pitcher_col,
        comp_pitch_col,
        metrics,
        logo_bytes=logo_bytes,
        headshot_bytes=headshot_bytes,
        primary_color=report_primary_color,
        accent_color=report_accent_color,
        target_bio=target_bio,
    )
    st.download_button(
        "Export PDF",
        data=pdf_bytes,
        file_name=f"pitcher_similarity_report_{safe_filename(target_display_name)}.pdf",
        mime="application/pdf",
    )
except Exception as exc:
    st.warning(f"PDF report could not be generated: {exc}")

# PowerPoint export removed in v18. PDF is the primary report output.

with st.expander("How similarity is calculated"):
    st.write(
        "The app averages each pitcher by pitch type, standardizes every selected metric within the comparison pool for that pitch type, "
        "then calculates distance between the target pitch and every comparison pitcher pitch. Smaller distance means more similar. "
        "The similarity score is `100 / (1 + distance)`, so higher is better. Overall similarity is the average distance across matched pitch types."
    )
