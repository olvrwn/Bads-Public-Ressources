"""
Habit Tracker Guides Agent
Triggered every other week by an external scheduler (e.g. GitHub Actions cron).
Run directly with: python habit_guides_agent.py

Pipeline:
  1. Discover  -- broad open-ended search per habit x language
  2. Coverage  -- for any habit x language with <MIN_GUIDES_PER_COMBO guides, run a
                 targeted top-up search so every combo has baseline content
  3. Validate  -- URL liveness check + value judgement by Claude (with page snippet)
  4. Audit     -- moderate liveness check: remove only on GET 404/410
  5. Enrich    -- assign tags via Claude; can propose new tags (capped at MAX_TAGS total)
  6. Tag hygiene -- normalise plural/singular variants; merge near-synonym tags
  7. Save      -- merge, validate schema (with auto-repair), persist JSON

Key settings:
  - VALIDATION_THRESHOLD  : 6.0
  - TRUSTED_DOMAIN_BONUS  : +1.0
  - Dead-link rule        : moderate -- GET 404 or 410 only
  - MIN_GUIDES_PER_COMBO  : 3    (top-up search fires when below this)
  - MAX_TAGS              : 15   (hard cap on total distinct tags in the registry)
  - Snippet fetch         : yes, before scoring
  - Near-duplicate dedup  : URL normalisation + Jaccard title similarity
"""

import os
import re
import json
import httpx
import anthropic
from datetime import datetime, timezone
from pathlib import Path

# --- Config -------------------------------------------------------------------

GUIDES_FILE = "guides.json"

VALIDATION_THRESHOLD = 6.0
TRUSTED_DOMAIN_BONUS  = 1.0

MIN_GUIDES_PER_COMBO = 3

MAX_TAGS = 15

DEAD_STATUS_CODES      = {404, 410}
TRANSIENT_STATUS_CODES = {429, 451, 500, 502, 503, 504}

DEFAULT_TAGS: set[str] = {
    "addiction", "effects", "global", "medical",
    "overview", "prevention", "risks", "treatment", "trends",
}

TAG_MERGE_MAP: dict[str, str] = {
    "risk":          "risks",
    "effect":        "effects",
    "trend":         "trends",
    "health":        "medical",
    "medicine":      "medical",
    "therapy":       "treatment",
    "rehab":         "treatment",
    "recovery":      "treatment",
    "abuse":         "addiction",
    "dependency":    "addiction",
    "dependence":    "addiction",
    "drug":          "overview",
    "substance":     "overview",
    "education":     "prevention",
    "awareness":     "prevention",
    "statistics":    "trends",
    "research":      "trends",
    "science":       "medical",
    "worldwide":     "global",
    "international": "global",
}

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

TRUSTED_DOMAINS: list[str] = [
    "apa.org", "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "healthline.com", "mayoclinic.org", "psychologytoday.com",
    "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov",
    "inserm.fr", "ameli.fr", "has-sante.fr",
    "bzga.de", "dhs.de", "gesundheitsinformation.de",
    "mscbs.gob.es", "fundacionsalud.org",
    "sns.gov.pt", "portaldasaude.pt", "fiocruz.br",
]

# --- Anthropic client ---------------------------------------------------------

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# --- Model config -------------------------------------------------------------

MODEL_SEARCH = "claude-sonnet-4-5"
MODEL_FAST   = "claude-haiku-4-5"

# --- Helpers ------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_guides() -> tuple[list[dict], set[str]]:
    """Returns (guides, tag_registry). Registry falls back to DEFAULT_TAGS if absent."""
    path = Path(GUIDES_FILE)
    if not path.exists():
        return [], set(DEFAULT_TAGS)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        guides = data.get("guides", [])
        registry = set(data.get("tag_registry", DEFAULT_TAGS))
    else:
        guides = data
        registry = set(DEFAULT_TAGS)
    return guides, registry


def save_guides(guides: list[dict], tag_registry: set[str]) -> None:
    path = Path(GUIDES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated": utc_now().isoformat(),
        "tag_registry": sorted(tag_registry),
        "guides": guides,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(guides)} guides | tag registry: {sorted(tag_registry)}")


def next_id(existing: list[dict]) -> str:
    if not existing:
        return "1"
    numeric = [int(g["id"]) for g in existing if str(g.get("id", "")).isdigit()]
    return str(max(numeric) + 1) if numeric else "1"


def normalise_url(url: str) -> str:
    """Strip scheme, trailing slash, and query params for dedup comparison."""
    url = re.sub(r'^https?://', '', url.lower())
    url = url.rstrip('/')
    url = re.sub(r'\?.*$', '', url)   # FIX: was r'\\?.*$' (double-escaped, never stripped query params)
    return url


def titles_are_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """Jaccard similarity on word sets -- catches reworded duplicates."""
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def is_duplicate(candidate: dict, existing_guides: list[dict]) -> bool:
    """Return True if candidate is a near-duplicate of any existing guide."""
    norm_url = normalise_url(candidate.get("url", ""))
    cand_title = candidate.get("title", "")
    for g in existing_guides:
        if normalise_url(g.get("url", "")) == norm_url:
            return True
        if titles_are_similar(cand_title, g.get("title", "")):
            return True
    return False


def liveness_result(url: str) -> str:
    """
    Moderate dead-link detection.

    Returns one of: "alive" | "dead" | "transient"

    "dead"      -> GET returns 404 or 410 specifically
    "transient" -> any other non-2xx/3xx (5xx, 429, 451, network error, timeout)
    "alive"     -> GET returns < 400

    HEAD is skipped entirely -- it is unreliable for dead-link detection
    (many servers return 405 or wrong codes on HEAD).
    """
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=10)
        if resp.status_code < 400:
            return "alive"
        if resp.status_code in DEAD_STATUS_CODES:
            return "dead"
        return "transient"
    except Exception:
        return "transient"


def is_trusted(url: str) -> bool:
    return any(domain in url for domain in TRUSTED_DOMAINS)


def strip_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)
    return raw.strip()


def fetch_page_snippet(url: str, max_chars: int = 800) -> str:
    """Grab a short plaintext snippet from the page for richer scoring."""
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=10,
                         headers={"Accept": "text/html"})
        if resp.status_code >= 400:
            return ""
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return ""


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

# --- Step 1: Discover ---------------------------------------------------------

def discover_new_articles(existing_guides: list[dict]) -> list[dict]:
    """
    One broad search per habit x language.
    Existing URLs and normalised titles are passed so Claude avoids duplicates.
    """
    print("\n🔎 Step 1: Discovering new articles...")
    existing_urls: set[str] = {g["url"] for g in existing_guides}
    existing_norm_urls: set[str] = {normalise_url(u) for u in existing_urls}
    all_found: list[dict] = []

    for lang in LANGUAGES:
        for habit in HABITS:
            print(f"   [{lang['code']}] {habit}")
            known_sample = list(existing_urls)[:30]
            prompt = f"""Search the web for high-quality {lang['locale']} articles about "{habit}"
as a substance or behaviour -- covering any relevant angle such as health effects,
addiction, risks, treatment, prevention, or science.

The article MUST be written entirely in {lang['name']} (language code: {lang['code']}).
Sources do NOT need to be from official or government bodies -- any trustworthy,
accurate, and informative health or science source is acceptable (blogs, clinics,
NGOs, reputable news outlets, universities, etc.).

Find up to 3 credible articles. Include an article if it genuinely provides
value to someone trying to understand or manage this habit. Be generous -- prefer
to include borderline-good articles rather than returning an empty list.

Return ONLY a valid JSON array -- no markdown, no explanation:
[
  {{
    "url": "https://...",
    "title": "Article title in {lang['name']}",
    "description": "1-2 sentence summary in {lang['name']}",
    "source": "Publisher name",
    "author": null,
    "language": "{lang['code']}",
    "habits": ["{habit}"],
    "estimatedReadingMinutes": 5
  }}
]

Do NOT include any of these already-known URLs: {known_sample}
If nothing genuinely valuable is found, return an empty array: []
"""
            try:
                raw = claude_with_search(prompt)
                found = json.loads(strip_fence(raw))
                if isinstance(found, list):
                    fresh = [
                        item for item in found
                        if item.get("url")
                        and normalise_url(item["url"]) not in existing_norm_urls
                    ]
                    all_found.extend(fresh)
            except Exception as e:
                print(f"   ⚠️  Search failed [{lang['code']}] {habit}: {e}")

    seen_norm: set[str] = set(existing_norm_urls)
    unique: list[dict] = []
    for item in all_found:
        norm = normalise_url(item.get("url", ""))
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            unique.append(item)

    print(f"   Found {len(unique)} new candidate articles")
    return unique

# --- Step 2: Coverage top-up --------------------------------------------------

def find_coverage_gaps(all_guides: list[dict]) -> list[tuple[str, dict]]:
    """
    Return (habit, lang_dict) pairs that have fewer than MIN_GUIDES_PER_COMBO guides.
    Counts guides in `all_guides` (existing kept + newly approved candidates).
    """
    counts: dict[tuple[str, str], int] = {}
    for g in all_guides:
        lang = g.get("language", "")
        for h in (g.get("habits") or []):
            counts[(h, lang)] = counts.get((h, lang), 0) + 1

    gaps = []
    for lang in LANGUAGES:
        for habit in HABITS:
            if counts.get((habit, lang["code"]), 0) < MIN_GUIDES_PER_COMBO:
                gaps.append((habit, lang))
    return gaps


def topup_search(
    gaps: list[tuple[str, dict]],
    existing_guides: list[dict],
    already_found: list[dict],
) -> list[dict]:
    """
    Targeted searches for each under-covered habit x language combo.
    Returns new candidate articles (not already in existing_guides or already_found).
    """
    if not gaps:
        return []

    print(f"\n🎯 Step 2: Coverage top-up -- {len(gaps)} gap(s) to fill...")
    all_known = existing_guides + already_found
    known_norm: set[str] = {normalise_url(g.get("url", "")) for g in all_known}
    topup_candidates: list[dict] = []

    for habit, lang in gaps:
        current_count = sum(
            1 for g in all_known
            if lang["code"] == g.get("language")
            and habit in (g.get("habits") or [])
        )
        needed = MIN_GUIDES_PER_COMBO - current_count
        print(f"   [{lang['code']}] {habit} -- need {needed} more article(s)")

        prompt = f"""I urgently need {needed} {lang['locale']} article(s) about "{habit}".

This is a coverage gap: we have almost no content for this combination.
Please find ANY trustworthy, accurate {lang['name']}-language source -- government,
NGO, clinic, university, reputable news outlet, health blog -- that covers
"{habit}" in terms of health effects, risks, addiction, treatment, or prevention.

The article MUST be primarily written in {lang['name']} (language code: {lang['code']}).
Be flexible with source type -- accuracy and usefulness matter more than prestige.

Return ONLY a valid JSON array (no markdown):
[
  {{
    "url": "https://...",
    "title": "Title in {lang['name']}",
    "description": "1-2 sentence summary in {lang['name']}",
    "source": "Publisher name",
    "author": null,
    "language": "{lang['code']}",
    "habits": ["{habit}"],
    "estimatedReadingMinutes": 5
  }}
]

Do NOT include any of these known URLs: {list(known_norm)[:20]}
If truly nothing exists, return [].
"""
        try:
            raw = claude_with_search(prompt)
            found = json.loads(strip_fence(raw))
            if isinstance(found, list):
                fresh = [
                    item for item in found
                    if item.get("url")
                    and normalise_url(item["url"]) not in known_norm
                ]
                for item in fresh:
                    known_norm.add(normalise_url(item["url"]))
                topup_candidates.extend(fresh)
        except Exception as e:
            print(f"   ⚠️  Top-up search failed [{lang['code']}] {habit}: {e}")

    print(f"   Top-up found {len(topup_candidates)} additional candidate(s)")
    return topup_candidates


# --- Step 3: Validate new articles --------------------------------------------

def _score_article(article: dict) -> dict:
    url = article["url"]
    trusted = is_trusted(url)
    snippet = fetch_page_snippet(url)

    prompt = f"""You are a health content quality reviewer for a habit-tracking app.

Evaluate this article:
URL: {url}
Title: {article.get('title', '')}
Description: {article.get('description', '')}
Language: {article.get('language', 'en')}
Habit: {(article.get('habits') or ['?'])}
Trusted domain: {trusted}
Page snippet: {snippet or '(could not fetch)'}

Score 1-10 on each dimension:
- credibility : Reputable source, author credentials, cites evidence
- accuracy    : Evidence-based, factually correct, no misinformation
- relevance   : Useful to someone managing or understanding this habit
- recency     : Prefer last 3 years; older is fine if still accurate

Be generous on borderline cases -- a score of 6 is acceptable for a solid
but not exceptional article from a non-official source.

Return ONLY JSON (no markdown):
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
        bonus = TRUSTED_DOMAIN_BONUS if trusted else 0.0
        scores["overall"] = round(min(10.0, overall + bonus), 2)
        scores["verdict"] = "approved" if scores["overall"] >= VALIDATION_THRESHOLD else "rejected"
        return scores
    except Exception as e:
        return {
            "credibility": 0, "accuracy": 0, "relevance": 0, "recency": 0,
            "overall": 0.0, "verdict": "rejected",
            "rejection_reason": f"Validation error: {e}",
        }


def validate_new_articles(
    candidates: list[dict],
    existing_guides: list[dict],
) -> tuple[list[dict], list[dict]]:
    print("\n🧪 Step 3: Validating new articles...")
    approved, rejected = [], []

    for article in candidates:
        url = article["url"]

        if is_duplicate(article, existing_guides):
            print(f"   ♻️  Duplicate skipped: {url}")
            rejected.append({**article, "_rejection_reason": "Near-duplicate of existing guide"})
            continue

        result = liveness_result(url)
        if result == "dead":
            print(f"   ✗ Dead link: {url}")
            rejected.append({**article, "_rejection_reason": "URL not accessible"})
            continue
        if result == "transient":
            print(f"   ⏳ Transient error (skipping, not rejecting): {url}")
            continue

        scores = _score_article(article)
        article["_validation"] = scores

        if scores["verdict"] == "approved":
            print(f"   ✅ ({scores['overall']:.1f}): {article.get('title', url)}")
            approved.append(article)
        else:
            reason = scores.get("rejection_reason") or "Below quality threshold"
            print(f"   ✗ ({scores['overall']:.1f}): {article.get('title', url)} -- {reason}")
            rejected.append(article)

    print(f"   Approved: {len(approved)} | Rejected: {len(rejected)}")
    return approved, rejected

# --- Step 4: Audit existing guides (conservative liveness check) --------------

def audit_existing_guides(guides: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Conservative liveness check.
    A guide is removed ONLY when we get a definitive 404/410.
    Timeouts, 5xx, 429, and pure network errors -> keep the guide.
    """
    print("\n✅ Step 4: Auditing existing guides (conservative liveness check)...")
    kept, removed = [], []

    for guide in guides:
        url = guide.get("url", "")
        result = liveness_result(url)

        if result == "dead":
            print(f"   🗑️  Dead link removed: {url}")
            guide["_removal_reason"] = "URL returned definitive 404/410"
            removed.append(guide)
        else:
            if result == "transient":
                print(f"   ⚠️  Transient error (keeping): {url}")
            kept.append(guide)

    print(f"   Kept: {len(kept)} | Removed: {len(removed)}")
    return kept, removed

# --- Step 5: Enrich tags ------------------------------------------------------

def normalise_tag(tag: str) -> str:
    """Apply the merge map to a single tag string."""
    return TAG_MERGE_MAP.get(tag.lower().strip(), tag.lower().strip())


def apply_tag_merge_map(guides: list[dict]) -> tuple[list[dict], int]:
    """
    Walk every guide and rewrite any tag that has a canonical alias.
    Returns (updated_guides, number_of_tags_rewritten).
    """
    rewrites = 0
    for guide in guides:
        old_tags = guide.get("tags") or []
        new_tags_set: list[str] = []
        seen: set[str] = set()
        for t in old_tags:
            canonical = normalise_tag(t)
            if canonical not in seen:
                seen.add(canonical)
                new_tags_set.append(canonical)
            if canonical != t:
                rewrites += 1
        guide["tags"] = new_tags_set
    return guides, rewrites


def prune_registry(registry: set[str], guides: list[dict]) -> set[str]:
    """
    If the registry exceeds MAX_TAGS, drop the least-used tags until it fits.
    Tags used by at least one guide are preferred; truly unused ones go first.
    """
    if len(registry) <= MAX_TAGS:
        return registry

    usage: dict[str, int] = {t: 0 for t in registry}
    for g in guides:
        for t in (g.get("tags") or []):
            if t in usage:
                usage[t] += 1

    sorted_tags = sorted(registry, key=lambda t: usage.get(t, 0))
    to_drop = len(registry) - MAX_TAGS
    dropped = set(sorted_tags[:to_drop])
    print(f"   🏷️  Registry over cap -- dropping {dropped}")
    return registry - dropped


def enrich_tags(article: dict, tag_registry: set[str]) -> tuple[list[str], set[str]]:
    """
    Ask Claude to assign 1-4 tags from the current registry, or propose a new
    one if none fit well. Returns (tags_for_this_article, updated_registry).
    """
    prompt = f"""You are tagging a health article for a habit-tracking app.

Current tag registry (prefer these):
{sorted(tag_registry)}

Article:
  Title      : {article.get('title', '')}
  Description: {article.get('description', '')}
  Habits     : {article.get('habits', [])}

Instructions:
1. Pick 1-4 tags from the registry that best describe this article.
2. If and ONLY IF no existing tag fits at all, you may propose ONE new tag.
   - New tags must be a single lowercase word (no spaces or hyphens).
   - Do not propose a new tag if an existing one is even a rough fit.
3. Apply these merge rules before returning -- rewrite any of these to their canonical form:
{json.dumps(TAG_MERGE_MAP, indent=2)}

Return ONLY JSON -- no markdown:
{{
  "tags": ["tag1", "tag2"],
  "new_tag": null
}}
(set "new_tag" to null if you did not propose one)
"""
    try:
        result = json.loads(strip_fence(claude(prompt)))
        raw_tags: list[str] = result.get("tags") or []
        new_tag: str | None = result.get("new_tag") or None

        tags = list(dict.fromkeys(normalise_tag(t) for t in raw_tags))
        tags = [t for t in tags if t in tag_registry]

        updated_registry = set(tag_registry)
        if new_tag:
            new_tag = normalise_tag(new_tag)
            if new_tag not in updated_registry:
                print(f"   🏷️  New tag proposed: '{new_tag}'")
                updated_registry.add(new_tag)

        return tags if tags else ["overview"], updated_registry
    except Exception:
        return ["overview"], tag_registry

# --- Step 6: Build guide entry ------------------------------------------------

def build_guide_entry(
    article: dict, guide_id: str, tag_registry: set[str]
) -> tuple[dict, set[str]]:
    reading_minutes: int | None = None
    try:
        val = article.get("estimatedReadingMinutes")
        if val is not None:
            reading_minutes = int(val)
    except (TypeError, ValueError):
        pass

    tags, updated_registry = enrich_tags(article, tag_registry)
    entry = {
        "id": guide_id,
        "title": article.get("title", ""),
        "description": article.get("description") or None,
        "language": article.get("language", "en"),
        "estimatedReadingMinutes": reading_minutes,
        "source": article.get("source", ""),
        "url": article["url"],
        "author": article.get("author") or None,
        "habits": article.get("habits") or None,
        "tags": tags,
        # last_checked removed intentionally
    }
    return entry, updated_registry

# --- Schema validation & repair -----------------------------------------------

REQUIRED_KEYS: set[str] = {"id", "title", "description", "language", "source", "url", "habits"}


def validate_json_schema(
    guides: list[dict], tag_registry: set[str]
) -> tuple[bool, list[str]]:
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
        bad_tags = [t for t in (guide.get("tags") or []) if t not in tag_registry]
        if bad_tags:
            errors.append(f"Guide[{i}] id={gid} unknown tags: {bad_tags}")
    return len(errors) == 0, errors


def repair_guides_with_claude(
    guides: list[dict], errors: list[str], tag_registry: set[str]
) -> list[dict]:
    MAX_ATTEMPTS = 2
    current = guides

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n🔧 Repair attempt {attempt}/{MAX_ATTEMPTS}...")
        prompt = f"""Fix this JSON array of guide objects to pass schema validation.

Rules:
- Required keys per entry: {sorted(REQUIRED_KEYS)}
- "url" must start with "http"
- "language" must be one of: {sorted(VALID_LANG_CODES)}
- "tags" values must each be one of: {sorted(tag_registry)}
- Preserve "id" unchanged on every entry

Errors to fix:
{json.dumps(errors, indent=2)}

Data:
{json.dumps(current, indent=2, ensure_ascii=False)}

Return ONLY the corrected JSON array -- no markdown, no explanation.
Use "" for missing string fields. Remove entries with unfixable URLs.
"""
        try:
            repaired = json.loads(strip_fence(claude(prompt)))
            valid, new_errors = validate_json_schema(repaired, tag_registry)
            if valid:
                print(f"   ✅ Repair succeeded on attempt {attempt}")
                return repaired
            print(f"   ⚠️  Still invalid after attempt {attempt}: {new_errors}")
            errors, current = new_errors, repaired
        except Exception as e:
            print(f"   ✗ Repair attempt {attempt} raised exception: {e}")

    print("   ✗ All repair attempts exhausted -- saving last best effort.")
    return current

# --- Main ---------------------------------------------------------------------

def run() -> dict:
    print("🚀 Starting Habit Guides Agent")
    print("=" * 52)

    existing_guides, tag_registry = load_guides()
    print(f"📑 Loaded {len(existing_guides)} existing guides | "
          f"{len(tag_registry)} tags in registry: {sorted(tag_registry)}")

    # -- 0. Tag hygiene on existing guides (merge map pass) --------------------
    existing_guides, rewrites = apply_tag_merge_map(existing_guides)
    if rewrites:
        print(f"   🏷️  Rewrote {rewrites} tag(s) on existing guides via merge map")

    # -- 1. Discover -----------------------------------------------------------
    candidates = discover_new_articles(existing_guides)

    # -- 2. Coverage top-up ----------------------------------------------------
    gaps = find_coverage_gaps(existing_guides + candidates)
    topup = topup_search(gaps, existing_guides, candidates)
    all_candidates = candidates + topup

    # -- 3. Validate new -------------------------------------------------------
    approved_new, rejected_new = validate_new_articles(all_candidates, existing_guides)

    # -- 4. Audit existing (moderate liveness: GET 404/410 only) ---------------
    kept_existing, removed_existing = audit_existing_guides(existing_guides)

    # -- 5. Build new entries (tags enriched, registry updated per article) ----
    new_entries: list[dict] = []
    id_pool = kept_existing.copy()
    for article in approved_new:
        guide_id = next_id(id_pool)
        entry, tag_registry = build_guide_entry(article, guide_id, tag_registry)
        new_entries.append(entry)
        id_pool.append(entry)

    # -- 6. Tag hygiene -- prune registry if over cap --------------------------
    final_guides = kept_existing + new_entries
    tag_registry = prune_registry(tag_registry, final_guides)

    final_guides, _ = apply_tag_merge_map(final_guides)

    for g in final_guides:
        g["tags"] = [t for t in (g.get("tags") or []) if t in tag_registry] or ["overview"]

    # -- 7. Schema validation with auto-repair ---------------------------------
    valid, errors = validate_json_schema(final_guides, tag_registry)
    if not valid:
        print(f"\n⚠️  Schema validation failed ({len(errors)} error(s)):")
        for err in errors:
            print(f"   • {err}")
        final_guides = repair_guides_with_claude(final_guides, errors, tag_registry)
        valid, errors = validate_json_schema(final_guides, tag_registry)
        if not valid:
            raise ValueError(
                "✗ JSON schema still invalid after repair!\n" + "\n".join(errors)
            )

    # -- 8. Save ---------------------------------------------------------------
    save_guides(final_guides, tag_registry)

    # -- Summary (reporting only - guides.json already written above) ----------
    lang_counts = {l["code"]: 0 for l in LANGUAGES}
    habit_counts = {h: 0 for h in HABITS}
    for g in final_guides:
        lc = g.get("language", "")
        if lc in lang_counts:
            lang_counts[lc] += 1
        for h in (g.get("habits") or []):
            if h in habit_counts:
                habit_counts[h] += 1

    coverage: dict[str, dict[str, int]] = {h: {} for h in HABITS}
    for g in final_guides:
        lc = g.get("language", "")
        for h in (g.get("habits") or []):
            if h in coverage:
                coverage[h][lc] = coverage[h].get(lc, 0) + 1

    still_gaps = [
        f"{h}/{lc}"
        for h in HABITS
        for lang in LANGUAGES
        for lc in [lang["code"]]
        if coverage.get(h, {}).get(lc, 0) < MIN_GUIDES_PER_COMBO
    ]

    summary = {
        "date": utc_now().strftime("%Y-%m-%d"),
        "total_guides": len(final_guides),
        "new_added": len(new_entries),
        "topup_candidates_searched": len(topup),
        "coverage_gaps_filled": len(gaps),
        "removed": len(removed_existing),
        "rejected_new": len(rejected_new),
        "tag_registry": sorted(tag_registry),
        "guides_by_language": lang_counts,
        "guides_by_habit": habit_counts,
        "coverage_below_minimum": still_gaps,
        "new_details": [
            {
                "id": g["id"],
                "url": g["url"],
                "title": g.get("title", ""),
                "description": g.get("description") or "",
                "language": g.get("language", ""),
                "habits": g.get("habits") or [],
                "tags": g.get("tags") or [],
            }
            for g in new_entries
        ],
        "removed_details": [
            {
                "url": g["url"],
                "title": g.get("title", ""),
                "reason": g.get("_removal_reason", "Unknown"),
            }
            for g in removed_existing
        ],
    }
    
    # Write summary.json for the GitHub Actions workflow to consume
    with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n📊 Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

# --- Entrypoint ---------------------------------------------------------------

if __name__ == "__main__":
    run()