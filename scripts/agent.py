"""
Habit Tracker Guides Agent
Runs weekly to discover, validate, and update guides.json.

Pipeline:
  1. Discover  — broad open-ended search per habit × language
  2. Validate  — URL liveness check + value judgement by Claude
  3. Audit     — fast HEAD/GET liveness check on existing guides only
  4. Enrich    — assign constrained tags via Claude
  5. Save      — merge, validate schema (with auto-repair), persist JSON
"""

import os
import json
import httpx
import anthropic
from datetime import datetime, timezone
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

GUIDES_FILE = "guides.json"
VALIDATION_THRESHOLD = 7.0

HABITS: list[str] = [
    "Tobacco", "Alcohol", "Weed", "Pills", "Opioids",
    "Cocaine", "Ecstasy", "Mushrooms", "Amphetamines",
    "LSD", "GHB", "Heroin", "Gambling",
]

LANGUAGES: list[dict] = [
    {"code": "en", "name": "English",    "locale": "English-language"},
    {"code": "fr", "name": "French",     "locale": "French-language"},
    {"code": "de", "name": "German",     "locale": "German-language"},
    {"code": "es", "name": "Spanish",    "locale": "Spanish-language"},
    {"code": "pt", "name": "Portuguese", "locale": "Portuguese-language"},
]
VALID_LANG_CODES: set[str] = {l["code"] for l in LANGUAGES}

ALLOWED_TAGS: set[str] = {
    "addiction", "effects", "global", "medical",
    "overview", "prevention", "risks", "treatment", "trends",
}

TRUSTED_DOMAINS: list[str] = [
    "apa.org", "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "healthline.com", "mayoclinic.org", "psychologytoday.com",
    "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov",
    "inserm.fr", "ameli.fr", "has-sante.fr",
    "bzga.de", "dhs.de", "gesundheitsinformation.de",
    "mscbs.gob.es", "fundacionsalud.org",
    "sns.gov.pt", "portaldasaude.pt", "fiocruz.br",
]

# ─── Anthropic client ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ─── Model config ──────────────────────────────────────────────────────────────

# Discovery uses web_search tool — must be a model that supports it
MODEL_SEARCH = "claude-sonnet-4-5"   # or "claude-haiku-4-5" if available on your tier

# Scoring + tagging + repair are plain text inference — Haiku is fast and cheap
MODEL_FAST   = "claude-haiku-4-5"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_guides() -> list[dict]:
    path = Path(GUIDES_FILE)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("guides", data) if isinstance(data, dict) else data


def save_guides(guides: list[dict]) -> None:
    path = Path(GUIDES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated": utc_now().isoformat(),
        "guides": guides,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(guides)} guides to {GUIDES_FILE}")


def next_id(existing: list[dict]) -> str:
    if not existing:
        return "1"
    numeric = [int(g["id"]) for g in existing if str(g.get("id", "")).isdigit()]
    return str(max(numeric) + 1) if numeric else "1"


def is_url_live(url: str) -> bool:
    try:
        r = httpx.head(url, follow_redirects=True, timeout=8)
        if r.status_code < 400:
            return True
    except Exception:
        pass
    try:
        r = httpx.get(url, follow_redirects=True, timeout=8)
        return r.status_code < 400
    except Exception:
        return False


def is_trusted(url: str) -> bool:
    return any(domain in url for domain in TRUSTED_DOMAINS)


def strip_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)
    return raw.strip()


def claude(prompt: str, system: str = "") -> str:
    kwargs: dict = {
        "model": MODEL_FAST,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content.text


def claude_with_search(prompt: str, system: str = "") -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": MODEL_SEARCH,
        "max_tokens": 4096,
        "messages": messages,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    if system:
        kwargs["system"] = system

    while True:
        resp = client.messages.create(**kwargs)
        if resp.stop_reason == "end_turn":
            return "\n".join(b.text for b in resp.content if hasattr(b, "text"))
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in resp.content
                if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages
        else:
            break
    return ""

# ─── Step 1: Discover ──────────────────────────────────────────────────────────

def discover_new_articles(existing_urls: set[str]) -> list[dict]:
    """
    One broad search per habit × language.
    Claude freely decides what articles are worth surfacing — no topic constraints.
    """
    print("\n🔍 Step 1: Discovering new articles...")
    all_found: list[dict] = []

    for lang in LANGUAGES:
        for habit in HABITS:
            print(f"   [{lang['code']}] {habit}")
            prompt = f"""Search the web for high-quality {lang['locale']} articles about "{habit}" 
as a substance or behaviour — covering any relevant angle such as health effects, 
addiction, risks, treatment, prevention, or science.

The article MUST be written entirely in {lang['name']} (language code: {lang['code']}).
Prefer reputable health, medical, academic, or government sources.
Find up to 3 credible articles. Only include an article if it genuinely provides 
value to someone trying to understand or manage this habit.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {{
    "url": "https://...",
    "title": "Article title in {lang['name']}",
    "description": "1–2 sentence summary in {lang['name']}",
    "source": "Publisher name",
    "author": null,
    "language": "{lang['code']}",
    "habits": ["{habit}"],
    "estimatedReadingMinutes": 5
  }}
]

Skip any of these already-known URLs: {list(existing_urls)[:20]}
If nothing genuinely valuable is found, return an empty array: []
"""
            try:
                raw = claude_with_search(prompt)
                found = json.loads(strip_fence(raw))
                if isinstance(found, list):
                    all_found.extend(found)
            except Exception as e:
                print(f"   ⚠️  Search failed [{lang['code']}] {habit}: {e}")

    seen = set(existing_urls)
    unique: list[dict] = []
    for item in all_found:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    print(f"   Found {len(unique)} new candidate articles")
    return unique

# ─── Step 2: Validate new articles ────────────────────────────────────────────

def _score_article(article: dict) -> dict:
    url = article["url"]
    trusted = is_trusted(url)

    prompt = f"""You are a health content quality reviewer for a habit-tracking app.

Evaluate this article:
URL: {url}
Title: {article.get('title', '')}
Description: {article.get('description', '')}
Language: {article.get('language', 'en')}
Habit: {(article.get('habits') or ['?'])}
Trusted source: {trusted}

Score 1–10 on each dimension:
- credibility : Reputable source, author credentials
- accuracy    : Evidence-based, no misinformation
- relevance   : Useful to someone managing this habit
- recency     : Prefer last 3 years

Return ONLY JSON:
{{
  "credibility": 0,
  "accuracy": 0,
  "relevance": 0,
  "recency": 0,
  "rejection_reason": ""
}}
"""
    try:
        scores = json.loads(strip_fence(claude(prompt)))
        overall = (
            scores["credibility"] + scores["accuracy"] +
            scores["relevance"] + scores["recency"]
        ) / 4.0
        scores["overall"] = round(min(10.0, overall + (0.5 if trusted else 0.0)), 2)
        scores["verdict"] = "approved" if scores["overall"] >= VALIDATION_THRESHOLD else "rejected"
        return scores
    except Exception as e:
        return {
            "credibility": 0, "accuracy": 0, "relevance": 0, "recency": 0,
            "overall": 0.0, "verdict": "rejected",
            "rejection_reason": f"Validation error: {e}",
        }


def validate_new_articles(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    print("\n🧪 Step 2: Validating new articles...")
    approved, rejected = [], []

    for article in candidates:
        url = article["url"]

        if not is_url_live(url):
            print(f"   ❌ Dead link: {url}")
            rejected.append({**article, "_rejection_reason": "URL not accessible"})
            continue

        scores = _score_article(article)
        article["_validation"] = scores

        if scores["verdict"] == "approved":
            print(f"   ✅ ({scores['overall']:.1f}): {article.get('title', url)}")
            approved.append(article)
        else:
            reason = scores.get("rejection_reason") or "Below quality threshold"
            print(f"   ❌ ({scores['overall']:.1f}): {article.get('title', url)} — {reason}")
            rejected.append(article)

    print(f"   Approved: {len(approved)} | Rejected: {len(rejected)}")
    return approved, rejected

# ─── Step 3: Audit existing guides (liveness only) ────────────────────────────

def audit_existing_guides(guides: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Fast liveness check only — no deep content review.
    A guide is removed only if its URL is no longer reachable.
    """
    print("\n✅ Step 3: Auditing existing guides (liveness check)...")
    kept, removed = [], []

    for guide in guides:
        url = guide.get("url", "")
        if is_url_live(url):
            kept.append(guide)
        else:
            print(f"   🗑️  Dead link removed: {url}")
            guide["_removal_reason"] = "URL no longer accessible"
            removed.append(guide)

    print(f"   Kept: {len(kept)} | Removed: {len(removed)}")
    return kept, removed

# ─── Step 4: Enrich tags ──────────────────────────────────────────────────────

def enrich_tags(article: dict) -> list[str]:
    prompt = f"""Select 1 to 4 tags for this article from EXACTLY this allowed set:
{sorted(ALLOWED_TAGS)}

Title: {article.get('title', '')}
Description: {article.get('description', '')}
Habits: {article.get('habits', [])}

Return ONLY a JSON array of strings, no markdown:
["tag1", "tag2"]
"""
    try:
        tags = json.loads(strip_fence(claude(prompt)))
        valid = [t for t in tags if t in ALLOWED_TAGS]
        return valid if valid else ["overview"]
    except Exception:
        return ["overview"]

# ─── Step 5: Build guide entry ────────────────────────────────────────────────

def build_guide_entry(article: dict, guide_id: str) -> dict:
    reading_minutes: int | None = None
    try:
        val = article.get("estimatedReadingMinutes")
        if val is not None:
            reading_minutes = int(val)
    except (TypeError, ValueError):
        pass

    return {
        "id": guide_id,
        "title": article.get("title", ""),
        "description": article.get("description") or None,
        "language": article.get("language", "en"),
        "estimatedReadingMinutes": reading_minutes,
        "source": article.get("source", ""),
        "url": article["url"],
        "author": article.get("author") or None,
        "habits": article.get("habits") or None,
        "tags": enrich_tags(article),
    }

# ─── Schema validation & repair ───────────────────────────────────────────────

REQUIRED_KEYS: set[str] = {"id", "title", "description", "language", "source", "url", "habits"}


def validate_json_schema(guides: list[dict]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for i, guide in enumerate(guides):
        gid = guide.get("id", "?")
        missing = REQUIRED_KEYS - guide.keys()
        if missing:
            errors.append(f"Guide[{i}] id={gid} missing keys: {missing}")
            continue
        if not str(guide.get("url", "")).startswith("http"):
            errors.append(f"Guide[{i}] id={gid} invalid URL: '{guide.get('url')}'")
        if guide.get("language") not in VALID_LANG_CODES:
            errors.append(f"Guide[{i}] id={gid} invalid language: '{guide.get('language')}'")
        bad_tags = [t for t in (guide.get("tags") or []) if t not in ALLOWED_TAGS]
        if bad_tags:
            errors.append(f"Guide[{i}] id={gid} invalid tags: {bad_tags}")
    return len(errors) == 0, errors


def repair_guides_with_claude(guides: list[dict], errors: list[str]) -> list[dict]:
    MAX_ATTEMPTS = 2
    current = guides

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n🔧 Repair attempt {attempt}/{MAX_ATTEMPTS}...")
        prompt = f"""Fix this JSON array of guide objects to pass schema validation.

Rules:
- Required keys per entry: {sorted(REQUIRED_KEYS)}
- "url" must start with "http"
- "language" must be one of: {sorted(VALID_LANG_CODES)}
- "tags" values must each be one of: {sorted(ALLOWED_TAGS)}
- Preserve "id" unchanged on every entry

Errors to fix:
{json.dumps(errors, indent=2)}

Data:
{json.dumps(current, indent=2, ensure_ascii=False)}

Return ONLY the corrected JSON array — no markdown, no explanation.
Use "" for missing string fields. Remove entries with unfixable URLs.
"""
        try:
            repaired = json.loads(strip_fence(claude(prompt)))
            valid, new_errors = validate_json_schema(repaired)
            if valid:
                print(f"   ✅ Repair succeeded on attempt {attempt}")
                return repaired
            print(f"   ⚠️  Still invalid after attempt {attempt}: {new_errors}")
            errors, current = new_errors, repaired
        except Exception as e:
            print(f"   ❌ Repair attempt {attempt} raised exception: {e}")

    print("   ❌ All repair attempts exhausted — saving last best effort.")
    return current

# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("🚀 Starting Habit Guides Agent")
    print("=" * 52)

    existing_guides = load_guides()
    existing_urls: set[str] = {g["url"] for g in existing_guides}
    print(f"📚 Loaded {len(existing_guides)} existing guides")

    # ── 1. Discover ───────────────────────────────────────────────────────────
    candidates = discover_new_articles(existing_urls)

    # ── 2. Validate new ───────────────────────────────────────────────────────
    approved_new, rejected_new = validate_new_articles(candidates)

    # ── 3. Audit existing (liveness only) ─────────────────────────────────────
    kept_existing, removed_existing = audit_existing_guides(existing_guides)

    # ── 4. Build new entries ──────────────────────────────────────────────────
    new_entries: list[dict] = []
    id_pool = kept_existing.copy()
    for article in approved_new:
        guide_id = next_id(id_pool)
        entry = build_guide_entry(article, guide_id)
        new_entries.append(entry)
        id_pool.append(entry)

    final_guides = kept_existing + new_entries

    # ── 5. Schema validation with auto-repair ─────────────────────────────────
    valid, errors = validate_json_schema(final_guides)
    if not valid:
        print(f"\n⚠️  Schema validation failed ({len(errors)} error(s)):")
        for err in errors:
            print(f"   • {err}")
        final_guides = repair_guides_with_claude(final_guides, errors)
        valid, errors = validate_json_schema(final_guides)
        if not valid:
            raise ValueError(
                "❌ JSON schema still invalid after repair!\n" + "\n".join(errors)
            )

    save_guides(final_guides)

    # ── Summary ───────────────────────────────────────────────────────────────
    lang_counts = {l["code"]: 0 for l in LANGUAGES}
    habit_counts = {h: 0 for h in HABITS}
    for g in final_guides:
        lc = g.get("language", "")
        if lc in lang_counts:
            lang_counts[lc] += 1
        for h in (g.get("habits") or []):
            if h in habit_counts:
                habit_counts[h] += 1

    summary = {
        "date": utc_now().strftime("%Y-%m-%d"),
        "total_guides": len(final_guides),
        "new_added": len(new_entries),
        "removed": len(removed_existing),
        "rejected_new": len(rejected_new),
        "guides_by_language": lang_counts,
        "guides_by_habit": habit_counts,
        "removed_urls": [g["url"] for g in removed_existing],
        "new_details": [
            {
                "id": g["id"],
                "url": g["url"],
                "title": g.get("title", ""),
                "language": g.get("language", ""),
                "habits": g.get("habits", []),
                "tags": g.get("tags", []),
            }
            for g in new_entries
        ],
    }

    print("\n📊 Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()