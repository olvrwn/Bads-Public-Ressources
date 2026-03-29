"""
Habit Tracker Guides Agent
Runs weekly to discover, validate, audit and update guides.json.

Pipeline:
  1. Discover  — find native-language articles per habit × language
  2. Validate  — score quality, check liveness, reject low-quality
  3. Audit     — re-check existing guides for staleness / dead links
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

# All habits matching your Swift Habit enum raw values
HABITS: list[str] = [
    "Tobacco", "Alcohol", "Weed", "Pills", "Opioids",
    "Cocaine", "Ecstasy", "Mushrooms", "Amphetamines",
    "LSD", "GHB", "Heroin", "Gambling",
]

# Target languages — code must be a valid ISO 639-1 value
LANGUAGES: list[dict] = [
    {"code": "en", "name": "English",    "locale": "English-language"},
    {"code": "fr", "name": "French",     "locale": "French-language"},
    {"code": "de", "name": "German",     "locale": "German-language"},
    {"code": "es", "name": "Spanish",    "locale": "Spanish-language"},
    {"code": "pt", "name": "Portuguese", "locale": "Portuguese-language"},
]
VALID_LANG_CODES: set[str] = {l["code"] for l in LANGUAGES}

# Exactly the tags your app supports
ALLOWED_TAGS: set[str] = {
    "addiction", "effects", "global", "medical",
    "overview", "prevention", "risks", "treatment", "trends",
}

# 2 search topics per habit — kept lean to avoid excessive API calls
HABIT_SEARCH_TOPICS: dict[str, list[str]] = {
    "Tobacco":      ["cigarette smoking health risks", "tobacco addiction science"],
    "Alcohol":      ["alcohol use disorder risks", "alcohol addiction effects"],
    "Weed":         ["cannabis use health effects", "marijuana addiction risks"],
    "Pills":        ["prescription pill abuse risks", "benzodiazepine addiction"],
    "Opioids":      ["opioid addiction crisis", "opioid overdose risks"],
    "Cocaine":      ["cocaine health effects", "cocaine addiction science"],
    "Ecstasy":      ["MDMA ecstasy health risks", "ecstasy addiction effects"],
    "Mushrooms":    ["psilocybin mushrooms risks", "magic mushrooms health effects"],
    "Amphetamines": ["amphetamine addiction risks", "methamphetamine health effects"],
    "LSD":          ["LSD acid health risks", "LSD psychological effects"],
    "GHB":          ["GHB drug health risks", "GHB addiction effects"],
    "Heroin":       ["heroin addiction health effects", "heroin overdose risks"],
    "Gambling":     ["gambling addiction psychology", "problem gambling health effects"],
}

# Domains that earn a +0.5 quality-score bonus
TRUSTED_DOMAINS: list[str] = [
    # Global / English
    "apa.org", "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "healthline.com", "mayoclinic.org", "psychologytoday.com",
    "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov",
    # French
    "inserm.fr", "ameli.fr", "has-sante.fr",
    # German
    "bzga.de", "dhs.de", "gesundheitsinformation.de",
    # Spanish
    "mscbs.gob.es", "fundacionsalud.org",
    # Portuguese
    "sns.gov.pt", "portaldasaude.pt", "fiocruz.br",
]

# ─── Anthropic client ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-opus-4-5"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    """Single source of truth for current UTC time (timezone-aware)."""
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
    """Return the next sequential integer ID as a string (e.g. '1', '2', …)."""
    if not existing:
        return "1"
    numeric = [int(g["id"]) for g in existing if str(g.get("id", "")).isdigit()]
    return str(max(numeric) + 1) if numeric else "1"


def is_url_live(url: str) -> bool:
    try:
        r = httpx.head(url, follow_redirects=True, timeout=10)
        return r.status_code < 400
    except Exception:
        try:
            r = httpx.get(url, follow_redirects=True, timeout=10)
            return r.status_code < 400
        except Exception:
            return False


def is_trusted(url: str) -> bool:
    return any(domain in url for domain in TRUSTED_DOMAINS)


def strip_fence(raw: str) -> str:
    """Remove markdown code fences from a Claude response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)
    return raw.strip()


def claude(prompt: str, system: str = "") -> str:
    """Plain Claude call — returns the first text block."""
    kwargs: dict = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content.text


def claude_with_search(prompt: str, system: str = "") -> str:
    """Claude call with web_search tool; drives the agentic loop automatically."""
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": MODEL,
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
    Search for native-language articles for every habit × language combination.
    Each candidate dict carries: url, title, description, source, author,
    language, habits, estimatedReadingMinutes.
    """
    print("\n🔍 Step 1: Discovering new articles...")
    all_found: list[dict] = []

    for lang in LANGUAGES:
        for habit in HABITS:
            for topic in HABIT_SEARCH_TOPICS[habit]:
                print(f"   [{lang['code']}] {habit} — {topic}")
                prompt = f"""Search the web for high-quality {lang['locale']} articles about: "{topic}"

The article MUST be written entirely in {lang['name']} (language code: {lang['code']}).
Prefer reputable health, medical, or academic sources.
Find 2–3 relevant, credible articles published in the last 3 years.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {{
    "url": "https://...",
    "title": "Article title in {lang['name']}",
    "description": "1–2 sentence summary written in {lang['name']}",
    "source": "Publisher or organisation name",
    "author": null,
    "language": "{lang['code']}",
    "habits": ["{habit}"],
    "estimatedReadingMinutes": 5
  }}
]

Exclude these already-known URLs: {list(existing_urls)[:20]}
"""
                try:
                    raw = claude_with_search(prompt)
                    found = json.loads(strip_fence(raw))
                    if isinstance(found, list):
                        all_found.extend(found)
                except Exception as e:
                    print(f"   ⚠️  Search failed [{lang['code']}] '{topic}': {e}")

    # Deduplicate by URL
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
    """Ask Claude to score quality on 4 dimensions; returns the parsed scores dict."""
    url = article["url"]
    habit = (article.get("habits") or ["Unknown"])
    trusted = is_trusted(url)

    prompt = f"""You are a health content quality reviewer for a habit-tracking app.

Evaluate this article for inclusion in the app's resource library:
URL: {url}
Title: {article.get('title', 'Unknown')}
Description: {article.get('description', '')}
Language: {article.get('language', 'en')}
Habit category: {habit}
Trusted source: {trusted}

Score 1–10 on each dimension (10 = best):
- credibility : Is the source reputable? (publisher, author credentials)
- accuracy    : Are claims evidence-based? No misinformation?
- relevance   : Does it relate to the habit, its risks, or behaviour change?
- recency     : Is it current (prefer last 3 years)?

Return ONLY JSON, no markdown:
{{
  "credibility": 0,
  "accuracy": 0,
  "relevance": 0,
  "recency": 0,
  "flags": [],
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
            "flags": [], "rejection_reason": f"Validation error: {e}",
        }


def validate_new_articles(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    print("\n🧪 Step 2: Validating new articles...")
    approved, rejected = [], []

    for article in candidates:
        url = article["url"]
        if not is_url_live(url):
            print(f"   ❌ Dead link: {url}")
            rejected.append({**article, "rejection_reason": "URL not accessible"})
            continue

        scores = _score_article(article)
        article["_validation"] = scores  # internal use only — not written to JSON

        if scores["verdict"] == "approved":
            print(f"   ✅ Approved ({scores['overall']:.1f}): {article.get('title', url)}")
            approved.append(article)
        else:
            reason = scores.get("rejection_reason") or "Below quality threshold"
            print(f"   ❌ Rejected ({scores['overall']:.1f}): {article.get('title', url)} — {reason}")
            rejected.append(article)

    print(f"   Approved: {len(approved)} | Rejected: {len(rejected)}")
    return approved, rejected

# ─── Step 3: Audit existing guides ────────────────────────────────────────────

def audit_existing_guides(guides: list[dict]) -> tuple[list[dict], list[dict]]:
    print("\n✅ Step 3: Auditing existing guides...")
    kept, removed = [], []

    for guide in guides:
        url = guide.get("url", "")

        if not is_url_live(url):
            print(f"   🗑️  Dead link removed: {url}")
            guide["_removal_reason"] = "URL no longer accessible"
            removed.append(guide)
            continue

        prompt = f"""Review this existing article in a habit-tracking app's resource library.
URL: {url}
Title: {guide.get('title', '')}
Language: {guide.get('language', 'en')}
Habits: {guide.get('habits', [])}

Is this article still accurate, relevant, and up to date as of {utc_now().strftime('%B %Y')}?
Consider: outdated research, retracted studies, superseded guidelines, defunct domains.

Return ONLY JSON:
{{"keep": true, "reason": "brief explanation"}}
"""
        try:
            result = json.loads(strip_fence(claude(prompt)))
            if result.get("keep", True):
                kept.append(guide)
            else:
                guide["_removal_reason"] = result.get("reason", "Outdated or irrelevant")
                print(f"   🗑️  Removed ({guide['_removal_reason']}): {url}")
                removed.append(guide)
        except Exception:
            kept.append(guide)  # keep on parse failure — safe default

    print(f"   Kept: {len(kept)} | Removed: {len(removed)}")
    return kept, removed

# ─── Step 4: Enrich tags ──────────────────────────────────────────────────────

def enrich_tags(article: dict) -> list[str]:
    """
    Assign 1–4 tags from ALLOWED_TAGS for an article.
    Falls back to ["overview"] on any error.
    """
    prompt = f"""Select 1 to 4 relevant tags for this article from EXACTLY this set:
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
    """
    Construct a Guide dict matching the Swift Guide model exactly.
    Internal keys prefixed with _ are stripped here.
    """
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
        prompt = f"""The following JSON array of guide objects failed schema validation.

Rules:
- Required keys for every entry: {sorted(REQUIRED_KEYS)}
- Every "url" must start with "http"
- "language" must be one of: {sorted(VALID_LANG_CODES)}
- Every value in "tags" must be one of: {sorted(ALLOWED_TAGS)}
- Preserve the "id" field of every entry unchanged

Errors:
{json.dumps(errors, indent=2)}

Data to fix:
{json.dumps(current, indent=2, ensure_ascii=False)}

Return ONLY the corrected JSON array — no markdown, no explanation.
Use "" for missing string fields. Remove entries whose URL cannot be fixed.
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

    # ── 3. Audit existing ─────────────────────────────────────────────────────
    kept_existing, removed_existing = audit_existing_guides(existing_guides)

    # ── 4. Build new entries with sequential IDs ──────────────────────────────
    new_entries: list[dict] = []
    id_pool = kept_existing.copy()          # grow alongside new entries for correct next_id
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
                "❌ JSON schema still invalid after repair attempts!\n" + "\n".join(errors)
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
        "removed_details": [
            {"url": g["url"], "reason": g.get("_removal_reason", "")}
            for g in removed_existing
        ],
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