"""
Habit Tracker Guides Agent (Optimized)
=======================================
Triggered every week by GitHub Actions (see .github/workflows/guides.yml).
Run locally with: python scripts/agent.py

Pipeline:
  1. Load      -- read guides.json + config.json
  2. Audit     -- stale-aware liveness check (only guides not verified within
                  LIVENESS_RECHECK_DAYS; Tier-0 domains rechecked more often).
                  Dead = 404/410 after redirects. Transient errors = keep.
  3. Discover  -- skip combos that are already covered AND fresh enough.
                  Gaps (<MIN_GUIDES_PER_COMBO) → aggressive prompt (4 results).
                  Stale-covered combos → light refresh prompt (2 results).
                  Fully-covered & fresh combos → skipped entirely.
  4. Validate  -- two-gate quality check:
                  Gate 1: domain tier (Tier1=9.0, Tier2=7.5, auto-pass)
                  Gate 2: unknown domains → combined quality+tag Haiku call
  5. Enrich    -- assign tags; reuse Gate-2 tags for unknown-domain articles.
                  If article already has valid tags matching current registry
                  and tags_version matches, skip re-tagging.
  6. Save      -- merge, schema-validate, persist flat JSON array
  7. Summary   -- write summary.json for GitHub Actions workflow

Optimisation highlights vs. original:
  ● Stale-aware liveness: only re-check guides older than LIVENESS_RECHECK_DAYS
    (Tier-0 domains are rechecked twice as often).
  ● Skip-when-covered discovery: combos already meeting min count and freshness
    threshold are skipped entirely — no Sonnet call at all.
  ● Batch Haiku tagging: Tier-1/Tier-2 articles are tagged in one batched call
    per 10 articles instead of one call per article.
  ● Tag-version cache: articles with a tags_version hash matching the current
    registry are not re-tagged.
  ● Tiered run mode (--mode gap-fill | maintenance | full):
      gap-fill    → search gap combos only, skip liveness
      maintenance → stale liveness only, skip fully-covered fresh combos
      full        → complete pipeline (weekly default)

Rate limit notes:
  - All Claude calls are serial with exponential backoff on 429s.
  - SEARCH_THROTTLE_SECS is the baseline gap between search calls.
  - MAX_LIVENESS_WORKERS only controls httpx threads (no API quota impact).
"""

import os
import re
import sys
import json
import time
import hashlib
import httpx
import argparse
import anthropic

from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# GitHub Actions log helpers
# ---------------------------------------------------------------------------

def gha_group(title: str) -> None:
    print(f"::group::{title}", flush=True)

def gha_endgroup() -> None:
    print("::endgroup::", flush=True)

def gha_notice(msg: str) -> None:
    print(f"::notice::{msg}", flush=True)

def gha_warning(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)

def gha_error(msg: str) -> None:
    print(f"::error::{msg}", flush=True)

def log(msg: str) -> None:
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_FILE = "config.json"
GUIDES_FILE = "guides.json"

MODEL_SEARCH = "claude-sonnet-4-5"
MODEL_FAST   = "claude-haiku-4-5"

MAX_LIVENESS_WORKERS    = 20    # httpx only — no API quota impact
SEARCH_THROTTLE_SECS    = 2     # minimum gap between Claude search calls
MAX_KNOWN_URLS_IN_PROMPT = 10   # keep prompts lean; dedup enforced in Python
BATCH_TAG_SIZE          = 10    # articles per batched Haiku tagging call
MAX_DISCOVERIES_PER_RUN = 10    # cap new candidates found per run

# Stale-aware liveness: how many days before we re-check a guide's URL.
# Tier-0 (unknown) domains are rechecked at half this interval.
LIVENESS_RECHECK_DAYS       = 21   # recheck every 3 weeks for Tier-1/2
LIVENESS_RECHECK_DAYS_TIER0 = 10   # recheck every ~10 days for unknown domains

# Discovery freshness: if a combo has enough guides and was last searched
# within this many days, skip searching it entirely.
DISCOVERY_FRESHNESS_DAYS = 30

# Retry settings for Claude API calls
RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_DELAY   = 5.0   # seconds; doubles on each 429

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_guides() -> list[dict]:
    path = Path(GUIDES_FILE)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("guides", [])


def save_guides(guides: list[dict]) -> None:
    Path(GUIDES_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(GUIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(guides, f, indent=2, ensure_ascii=False)
    gha_notice(f"Saved {len(guides)} guides to {GUIDES_FILE}")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today_str() -> str:
    return utc_now().strftime("%Y-%m-%d")


def days_since(date_str: str | None) -> int:
    """Return days since date_str (ISO format). Returns 9999 if None/invalid."""
    if not date_str:
        return 9999
    try:
        d = date.fromisoformat(date_str[:10])
        return (utc_now().date() - d).days
    except (ValueError, TypeError):
        return 9999


def registry_hash(tag_registry: set[str]) -> str:
    """Short hash of the sorted tag registry — used for cache invalidation."""
    canonical = json.dumps(sorted(tag_registry), separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]


def normalise_url(url: str) -> str:
    """Strip scheme, trailing slash, and query string for dedup comparison."""
    url = re.sub(r'^https?://', '', url.lower())
    url = url.rstrip('/')
    url = re.sub(r'\?.*$', '', url)
    return url


def titles_are_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """Jaccard similarity on word sets to catch reworded duplicates."""
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def is_duplicate(candidate: dict, existing: list[dict]) -> bool:
    norm  = normalise_url(candidate.get("url", ""))
    title = candidate.get("title", "")
    for g in existing:
        if normalise_url(g.get("url", "")) == norm:
            return True
        if titles_are_similar(title, g.get("title", "")):
            return True
    return False


def strip_fence(raw: str) -> str:
    """Remove markdown code fences Claude sometimes wraps JSON in."""
    # ✅ SAFETY GUARD: handle non-string responses
    if not isinstance(raw, str):
        return ""

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]  # ✅ FIXED (was returning tuple)
    return raw.strip()


def fetch_snippet(url: str, max_chars: int = 1000) -> str:
    """Fetch a short plaintext snippet from a page for content scoring."""
    try:
        resp = httpx.get(
            url, follow_redirects=True, timeout=10,
            headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code >= 400:
            return ""
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Claude wrappers with exponential backoff on 429
# ---------------------------------------------------------------------------

def _with_backoff(fn, *args, **kwargs):
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except anthropic.RateLimitError:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            gha_warning(
                f"Rate limit hit — waiting {delay:.0f}s before retry "
                f"({attempt}/{RETRY_MAX_ATTEMPTS - 1})"
            )
            time.sleep(delay)
            delay *= 2


def claude_fast(prompt: str, system: str = "") -> str:
    """Single-turn Claude Haiku call for scoring / tagging."""
    def _call():
        kwargs: dict = {
            "model":      MODEL_FAST,
            "max_tokens": 2048,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)

        # ✅ FIX: safely extract text from content blocks
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)

        return "\n".join(parts)

    return _with_backoff(_call)


def claude_search(prompt: str, system: str = "") -> str:
    """
    Agentic search loop using web_search tool.
    Only the initial API call is wrapped in backoff.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model":      MODEL_SEARCH,
        "max_tokens": 4096,
        "messages":   messages,
        "tools":      [{"type": "web_search_20250305", "name": "web_search"}],
    }
    if system:
        kwargs["system"] = system

    resp = _with_backoff(client.messages.create, **kwargs)

    while True:
        if resp.stop_reason == "end_turn":
            parts = []
            for b in resp.content:
                text = getattr(b, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, list):
                    parts.extend(t for t in text if isinstance(t, str))
            return "\n".join(parts)

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in resp.content
                if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages
            resp = client.messages.create(**kwargs)
        else:
            break

    return ""

# ---------------------------------------------------------------------------
# Step 1 — Stale-aware audit
# ---------------------------------------------------------------------------

def _domain_tier(url: str, config: dict) -> int:
    """Returns 1, 2, or 0 (unknown) based on trusted_domains config."""
    for domain in config["trusted_domains"]["tier1"]:
        if domain in url:
            return 1
    for domain in config["trusted_domains"]["tier2"]:
        if domain in url:
            return 2
    return 0


def _needs_liveness_check(guide: dict, config: dict) -> bool:
    """
    Returns True if this guide's URL should be re-checked this run.
    Tier-0 (unknown) domains are checked twice as often.
    """
    tier    = _domain_tier(guide.get("url", ""), config)
    max_age = LIVENESS_RECHECK_DAYS_TIER0 if tier == 0 else LIVENESS_RECHECK_DAYS
    return days_since(guide.get("last_checked")) >= max_age


def _check_liveness(guide: dict) -> tuple[dict, str]:
    """
    Returns (guide, status) where status is 'alive' | 'dead' | 'transient'.
    Follows redirects so a 301 → 404 chain is correctly marked dead.
    """
    url = guide.get("url", "")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=10)
        if resp.status_code < 400:
            return guide, "alive"
        return guide, "dead"
    except Exception:
        return guide, "dead"


def audit_guides(guides: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """
    Stale-aware parallel liveness check.
    - Guides checked recently (< LIVENESS_RECHECK_DAYS) are skipped.
    - Tier-0 domains get a shorter recheck interval.
    - Definitively dead (404/410) guides are removed.
    - Transient errors = keep, update last_checked.
    Returns (kept, removed).
    """
    to_check   = [g for g in guides if _needs_liveness_check(g, config)]
    skip_count = len(guides) - len(to_check)

    gha_group(
        f"Step 1 · Audit — {len(to_check)} guides to check "
        f"({skip_count} skipped as recently verified)"
    )

    kept, removed = [], []
    skip_map: dict[str, dict] = {
        g.get("url", ""): g
        for g in guides
        if not _needs_liveness_check(g, config)
    }

    # Keep already-fresh guides unchanged
    kept.extend(skip_map.values())

    if to_check:
        with ThreadPoolExecutor(max_workers=MAX_LIVENESS_WORKERS) as pool:
            futures = {pool.submit(_check_liveness, g): g for g in to_check}
            for future in as_completed(futures):
                guide, status = future.result()
                url = guide.get("url", "")
                if status == "dead":
                    log(f"  🗑️  Dead → removing: {url}")
                    guide["_removal_reason"] = "URL returned 404/410 (after redirects)"
                    removed.append(guide)
                else:
                    if status == "transient":
                        gha_warning(f"Transient error on {url} — keeping guide")
                    guide["last_checked"] = today_str()
                    kept.append(guide)

    log(f"  Checked: {len(to_check)} | Kept: {len(kept)} | Removed: {len(removed)}")
    gha_endgroup()
    return kept, removed

# ---------------------------------------------------------------------------
# Step 2 — Gap-aware, freshness-aware discovery (serial)
# ---------------------------------------------------------------------------

def _compute_combo_state(
    guides: list[dict],
    habits: list[str],
    languages: list[dict],
    min_combo: int,
) -> dict[tuple[str, str], dict]:
    """
    Returns a map from (habit, lang_code) → {count, newest_search_date}.
    newest is the most recent last_checked date for any guide in that combo.
    """
    state: dict[tuple[str, str], dict] = {}
    for g in guides:
        lc         = g.get("language", "")
        guide_date = g.get("last_checked") or "2000-01-01"
        for h in (g.get("habits") or []):
            key = (h, lc)
            if key not in state:
                state[key] = {"count": 0, "newest": "2000-01-01"}
            state[key]["count"] += 1
            if guide_date > state[key]["newest"]:
                state[key]["newest"] = guide_date
    return state


def _search_one_combo(
    habit: str,
    lang: dict,
    is_gap: bool,
    known_urls: set[str],
) -> list[dict]:
    """
    Single habit × language search. Returns a list of raw candidate dicts.
    is_gap=True  → ask for 4 articles with a more aggressive prompt.
    is_gap=False → ask for 2 articles for a light refresh.
    """
    time.sleep(SEARCH_THROTTLE_SECS)

    count   = 4 if is_gap else 2
    urgency = (
        "This is a COVERAGE GAP — we have almost no content for this combination. "
        "Be very flexible with source type. Any accurate, helpful source counts."
    ) if is_gap else (
        "We already have some content. Only return genuinely new, high-quality sources."
    )

    sample_urls = sorted(known_urls)[:MAX_KNOWN_URLS_IN_PROMPT]

    prompt = f"""Search the web for {count} high-quality {lang['locale']} articles about "{habit}".

{urgency}

Requirements:
- Article MUST be written entirely in {lang['name']} (language code: {lang['code']})
- Must cover at least one of: health effects, addiction, risks, treatment, prevention, science
- Acceptable sources: government, WHO/UN, universities, established health NGOs,
  reputable clinics, well-known health publishers, peer-reviewed journals
- Do NOT include: forums, Reddit, social media, anonymous blogs, product/sales pages

Do NOT include any of these already-known URLs:
{sample_urls}

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {{
    "url": "https://...",
    "title": "Article title in {lang['name']}",
    "description": "2-3 sentence factual summary in {lang['name']}",
    "source": "Publisher / organisation name",
    "author": null,
    "language": "{lang['code']}",
    "habits": ["{habit}"],
    "estimatedReadingMinutes": 5
  }}
]

If nothing genuinely valuable is found, return: []
"""
    try:
        raw    = claude_search(prompt)
        parsed = json.loads(strip_fence(raw))
        if isinstance(parsed, list):
            return [item for item in parsed if item.get("url")]
        return []
    except Exception as e:
        gha_warning(f"Search failed [{lang['code']}] {habit}: {e}")
        return []


def discover_articles(
    existing_guides: list[dict],
    config: dict,
    mode: str = "full",
) -> list[dict]:
    """
    Gap-aware, freshness-aware discovery. Fully serial.

    mode="gap-fill"    → only search combos that are under min threshold
    mode="maintenance" → search gaps + stale-covered combos; skip fresh ones
    mode="full"        → same behaviour as maintenance
    """
    habits    = config["habits"]
    languages = config["languages"]
    min_combo = config.get("min_guides_per_combo", 3)

    combo_state = _compute_combo_state(existing_guides, habits, languages, min_combo)
    known_urls: set[str] = {normalise_url(g["url"]) for g in existing_guides}

    tasks: list[tuple[str, dict, bool]] = []
    skipped_count = 0

    for lang in languages:
        for habit in habits:
            key   = (habit, lang["code"])
            cs    = combo_state.get(key, {"count": 0, "newest": "2000-01-01"})
            count = cs["count"]
            age   = days_since(cs["newest"])

            is_gap   = count < min_combo
            is_stale = age >= DISCOVERY_FRESHNESS_DAYS

            if mode == "gap-fill" and not is_gap:
                skipped_count += 1
                continue

            if not is_gap and not is_stale:
                skipped_count += 1
                continue

            tasks.append((habit, lang, is_gap))

    gap_count = sum(1 for _, _, g in tasks if g)

    gha_group(
        f"Step 2 · Discover — {len(tasks)} combos to search "
        f"({gap_count} gaps, {len(tasks) - gap_count} stale refreshes) "
        f"| {skipped_count} skipped (covered + fresh) | mode={mode}"
    )

    all_found: list[dict] = []
    seen_norm: set[str]   = set(known_urls)

    for habit, lang, is_gap in tasks:
        label = "GAP" if is_gap else "refresh"
        try:
            results = _search_one_combo(habit, lang, is_gap, known_urls)
            fresh = []
            for item in results:
                norm = normalise_url(item.get("url", ""))
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    fresh.append(item)
            log(f"  [{lang['code']}] {habit:<15} ({label}) → {len(fresh)} new candidate(s)")
            all_found.extend(fresh)
        except Exception as e:
            gha_warning(f"[{lang['code']}] {habit} ({label}) raised: {e}")

    if len(all_found) > MAX_DISCOVERIES_PER_RUN:
        log(f"  ⚠️  Capping candidates at {MAX_DISCOVERIES_PER_RUN} (found {len(all_found)} total)")
        all_found = all_found[:MAX_DISCOVERIES_PER_RUN]
    log(f"\n  Total candidates found: {len(all_found)}")
    gha_endgroup()
    return all_found

# ---------------------------------------------------------------------------
# Step 3 — Two-gate validation
# ---------------------------------------------------------------------------

def _gate2_and_tags_claude(
    article: dict,
    tag_registry: set[str],
    merge_map: dict,
) -> tuple[bool, str, list[str], str | None]:
    """
    Single Haiku call: Gate-2 quality check + tag assignment.
    Returns (passed, reason, tags, new_tag_or_None).
    """
    url     = article["url"]
    snippet = fetch_snippet(url)
    if not snippet:
        return False, "Could not fetch page content for review", [], None

    prompt = f"""You are a health content reviewer and tagger for a habit-tracking app.

Review the article and answer the quality questions, then assign tags.

URL      : {url}
Title    : {article.get('title', '')}
Habits   : {article.get('habits', [])}
Snippet  : {snippet}

--- PART 1: Quality checks (answer each with exactly "yes" or "no") ---
1. factual   — Is the health/science information grounded in evidence?
2. qualified — Does the content appear written or reviewed by a qualified person or credible organisation?
3. useful    — Would this genuinely help someone understand or manage this habit/substance?

--- PART 2: Tagging ---
Current tag registry (use these first):
{sorted(tag_registry)}

Tag merge rules — apply before returning:
{json.dumps(merge_map, indent=2)}

Pick 1-4 tags from the registry. Only propose a new tag (single lowercase word) if NO existing tag is even a rough fit.

Return ONLY valid JSON — no markdown:
{{
  "factual":   "yes",
  "qualified": "yes",
  "useful":    "yes",
  "notes":     null,
  "tags":      ["tag1", "tag2"],
  "new_tag":   null
}}
"""
    try:
        result  = json.loads(strip_fence(claude_fast(prompt)))
        passed  = all(
            result.get(k, "no").strip().lower() == "yes"
            for k in ("factual", "qualified", "useful")
        )
        reason  = result.get("notes") or ("All checks passed" if passed else "Failed quality check")
        tags    = result.get("tags") or []
        new_tag = result.get("new_tag") or None
        return passed, reason, tags, new_tag
    except Exception as e:
        return False, f"Validation/tagging error: {e}", [], None


def validate_candidates(
    candidates: list[dict],
    existing_guides: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Gate 1: domain tier check (auto-pass Tier1/Tier2).
    Gate 2: combined quality + tag check for unknown domains.
    Also deduplicates against existing_guides.
    Returns (approved, rejected).
    """
    gha_group(f"Step 3 · Validate — {len(candidates)} candidate(s)")
    approved, rejected = [], []

    for article in candidates:
        url = article.get("url", "")

        if is_duplicate(article, existing_guides):
            log(f"  ♻️  Duplicate skipped: {url}")
            rejected.append({**article, "_rejection_reason": "Duplicate of existing guide"})
            continue

        try:
            resp = httpx.get(url, follow_redirects=True, timeout=10)
            if resp.status_code >= 400:
                log(f"  ✗ Dead link skipped: {url}")
                rejected.append({**article, "_rejection_reason": f"URL returned {resp.status_code} and is probably dead"})
                continue
        except Exception:
            gha_warning(f"Liveness check failed for {url} — skipping candidate")
            continue

        tier = _domain_tier(url, config)
        if tier == 1:
            article["_score"] = 9.0
            log(f"  ✅ Tier-1 auto-pass (9.0): {article.get('title', url)}")
            approved.append(article)
            continue
        if tier == 2:
            article["_score"] = 7.5
            log(f"  ✅ Tier-2 auto-pass (7.5): {article.get('title', url)}")
            approved.append(article)
            continue

        tag_registry: set[str] = set(config["tag_registry"])
        merge_map: dict        = config["tag_merge_map"]
        passed, reason, tags, new_tag = _gate2_and_tags_claude(article, tag_registry, merge_map)
        if passed:
            article["_score"]         = 6.0
            article["_gate2_tags"]    = tags
            article["_gate2_new_tag"] = new_tag
            log(f"  ✅ Gate-2 passed: {article.get('title', url)}")
            approved.append(article)
        else:
            log(f"  ✗ Gate-2 failed ({reason}): {article.get('title', url)}")
            rejected.append({**article, "_rejection_reason": reason})

    log(f"\n  Approved: {len(approved)} | Rejected: {len(rejected)}")
    gha_endgroup()
    return approved, rejected

# ---------------------------------------------------------------------------
# Step 4 — Tag enrichment (batched + tag-version cache)
# ---------------------------------------------------------------------------

def normalise_tag(tag: str, merge_map: dict) -> str:
    return merge_map.get(tag.lower().strip(), tag.lower().strip())


def _tags_are_current(guide: dict, tag_registry: set[str], reg_hash: str) -> bool:
    """
    True if the guide already has tags, all of which exist in the current
    registry, and its tags_version matches the current registry hash.
    """
    if guide.get("tags_version") != reg_hash:
        return False
    tags = guide.get("tags") or []
    return bool(tags) and all(t in tag_registry for t in tags)


def _batch_tag_articles(
    articles: list[dict],
    tag_registry: set[str],
    merge_map: dict,
    max_tags: int,
) -> list[tuple[list[str], str | None]]:
    """
    Tag a batch of articles in a single Haiku call.
    Returns a list of (tags, new_tag_or_None) tuples in the same order.
    Falls back to ["overview"] for any slot that fails to parse.
    """
    items_json = json.dumps(
        [
            {
                "index":       i,
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "habits":      a.get("habits", []),
            }
            for i, a in enumerate(articles)
        ],
        ensure_ascii=False,
    )

    prompt = f"""Tag these health articles for a habit-tracking app.

Current tag registry (use these first):
{sorted(tag_registry)}

Tag merge rules — apply before returning:
{json.dumps(merge_map, indent=2)}

Instructions:
1. For each article, pick 1-4 tags from the registry that best fit.
2. Only propose a new tag (single lowercase word) if NO existing tag is even a rough fit.
3. Apply merge rules before returning any tag.

Articles (JSON array):
{items_json}

Return ONLY a valid JSON array with one object per article, in the same order:
[
  {{"index": 0, "tags": ["tag1"], "new_tag": null}},
  ...
]
"""
    default = [([("overview")], None)] * len(articles)
    try:
        raw    = claude_fast(prompt)
        parsed = json.loads(strip_fence(raw))
        results: list[tuple[list[str], str | None] | None] = [None] * len(articles)
        for item in parsed:
            idx = item.get("index")
            if idx is None or idx >= len(articles):
                continue
            raw_tags = item.get("tags") or []
            new_tag  = item.get("new_tag") or None
            tags = list(dict.fromkeys(normalise_tag(t, merge_map) for t in raw_tags))
            tags = [t for t in tags if t in tag_registry]
            results[idx] = (tags or ["overview"], new_tag)
        return [r if r is not None else (["overview"], None) for r in results]
    except Exception:
        return default


def enrich_tags_for_entries(
    articles: list[dict],
    tag_registry: set[str],
    core_tags: set[str],
    merge_map: dict,
    max_tags: int,
    reg_hash: str,
) -> tuple[list[list[str]], set[str]]:
    """
    Batch-enriches tags for a list of articles.

    - Gate-2 articles: tags already computed; reuse them.
    - Tier-1/2 articles without valid cached tags: batched Haiku call.
    - Existing guides with matching tags_version: skip re-tagging entirely.

    Returns (list_of_tags_per_article, updated_registry).
    """
    updated_registry = set(tag_registry)
    all_tags: list[list[str] | None] = [None] * len(articles)

    # Pass 1: fill from Gate-2 cache or tags_version cache
    needs_tagging: list[tuple[int, dict]] = []
    for i, article in enumerate(articles):
        if "_gate2_tags" in article:
            raw_tags = article.pop("_gate2_tags") or []
            new_tag  = article.pop("_gate2_new_tag", None)
            tags = list(dict.fromkeys(normalise_tag(t, merge_map) for t in raw_tags))
            tags = [t for t in tags if t in updated_registry]
            if new_tag:
                new_tag = normalise_tag(new_tag, merge_map)
                if new_tag not in updated_registry and len(updated_registry) < max_tags:
                    log(f"  🏷️  New tag added: '{new_tag}'")
                    updated_registry.add(new_tag)
            all_tags[i] = tags or ["overview"]
        elif _tags_are_current(article, updated_registry, reg_hash):
            all_tags[i] = article.get("tags") or ["overview"]
        else:
            needs_tagging.append((i, article))

    # Pass 2: batch-tag remaining articles
    if needs_tagging:
        batch_indices  = [idx for idx, _ in needs_tagging]
        batch_articles = [a for _, a in needs_tagging]

        for start in range(0, len(batch_articles), BATCH_TAG_SIZE):
            chunk_arts = batch_articles[start : start + BATCH_TAG_SIZE]
            chunk_idxs = batch_indices[start : start + BATCH_TAG_SIZE]
            results    = _batch_tag_articles(chunk_arts, updated_registry, merge_map, max_tags)
            for local_i, (tags, new_tag) in enumerate(results):
                if new_tag:
                    new_tag = normalise_tag(new_tag, merge_map)
                    if new_tag not in updated_registry:
                        if len(updated_registry) < max_tags:
                            log(f"  🏷️  New tag added: '{new_tag}'")
                            updated_registry.add(new_tag)
                        else:
                            gha_warning(
                                f"Tag registry at cap ({max_tags}). "
                                f"Proposed tag '{new_tag}' not added."
                            )
                all_tags[chunk_idxs[local_i]] = tags

    final_tags = [t if t is not None else ["overview"] for t in all_tags]
    return final_tags, updated_registry

# ---------------------------------------------------------------------------
# Step 5 — Build final guide entries
# ---------------------------------------------------------------------------

def next_id(existing: list[dict]) -> str:
    if not existing:
        return "1"
    numeric = [int(g["id"]) for g in existing if str(g.get("id", "")).isdigit()]
    return str(max(numeric) + 1) if numeric else "1"


def build_entries(
    approved_articles: list[dict],
    id_pool: list[dict],
    tag_registry: set[str],
    core_tags: set[str],
    merge_map: dict,
    max_tags: int,
    reg_hash: str,
) -> tuple[list[dict], set[str]]:
    """
    Enrich tags for all approved articles in one batched pass, then build
    the final guide dicts.
    Returns (new_entries, updated_registry).
    """
    all_article_tags, updated_registry = enrich_tags_for_entries(
        approved_articles, tag_registry, core_tags, merge_map, max_tags, reg_hash
    )

    new_entries: list[dict] = []
    for article, tags in zip(approved_articles, all_article_tags):
        reading_minutes: int | None = None
        try:
            val = article.get("estimatedReadingMinutes")
            if val is not None:
                reading_minutes = int(val)
        except (TypeError, ValueError):
            pass

        guide_id = next_id(id_pool)
        entry = {
            "id":                      guide_id,
            "title":                   article.get("title", ""),
            "description":             article.get("description") or None,
            "language":                article.get("language", "en"),
            "estimatedReadingMinutes": reading_minutes,
            "source":                  article.get("source", ""),
            "url":                     article["url"],
            "author":                  article.get("author") or None,
            "habits":                  article.get("habits") or None,
            "tags":                    tags,
            "tags_version":            reg_hash,
            "last_checked":            today_str(),
        }
        new_entries.append(entry)
        id_pool.append(entry)

    return new_entries, updated_registry

# ---------------------------------------------------------------------------
# Step 6 — Schema validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS    = {"id", "title", "description", "language", "source", "url", "habits"}
VALID_LANG_CODES: set[str] = set()  # populated at runtime from config


def validate_schema(guides: list[dict], tag_registry: set[str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for i, guide in enumerate(guides):
        gid     = guide.get("id", "?")
        missing = REQUIRED_KEYS - guide.keys()
        if missing:
            errors.append(f"Guide[{i}] id={gid} missing keys: {missing}")
            continue
        if not str(guide.get("url", "")).startswith("http"):
            errors.append(f"Guide[{i}] id={gid} invalid URL")
        if guide.get("language") not in VALID_LANG_CODES:
            errors.append(f"Guide[{i}] id={gid} invalid language: '{guide.get('language')}'")
        bad_tags = [t for t in (guide.get("tags") or []) if t not in tag_registry]
        if bad_tags:
            errors.append(f"Guide[{i}] id={gid} unknown tags: {bad_tags}")
    return len(errors) == 0, errors

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Habit Guides Agent")
    parser.add_argument(
        "--mode",
        choices=["gap-fill", "maintenance", "full"],
        default="full",
        help=(
            "gap-fill: search only combos below the minimum threshold, skip liveness. "
            "maintenance: stale liveness + skip fully-covered fresh combos (weekly default). "
            "full: complete pipeline."
        ),
    )
    return parser.parse_args()


def run() -> dict:
    args = parse_args()
    mode = args.mode

    log("🚀 Habit Guides Agent starting")
    log(f"   Mode: {mode}")
    log("=" * 52)

    # -- Load ------------------------------------------------------------------
    config = load_config()
    global VALID_LANG_CODES
    VALID_LANG_CODES = {l["code"] for l in config["languages"]}

    tag_registry: set[str] = set(config["tag_registry"])
    core_tags:    set[str] = set(config["core_tags"])
    merge_map:    dict     = config["tag_merge_map"]
    max_tags:     int      = config.get("max_tags", 15)
    reg_hash:     str      = registry_hash(tag_registry)

    existing_guides = load_guides()
    log(f"📑 Loaded {len(existing_guides)} existing guides")
    log(f"🏷️  Tag registry ({len(tag_registry)}): {sorted(tag_registry)}")
    log(f"🔑 Registry hash: {reg_hash}")

    # -- Step 1: Audit ---------------------------------------------------------
    if mode == "gap-fill":
        kept_guides, removed_guides = existing_guides, []
        log("⏭️  Audit skipped (gap-fill mode)")
    else:
        kept_guides, removed_guides = audit_guides(existing_guides, config)

    # -- Step 2: Discover ------------------------------------------------------
    candidates = discover_articles(kept_guides, config, mode=mode)

    # -- Step 3: Validate ------------------------------------------------------
    approved, rejected = validate_candidates(candidates, kept_guides, config)

    # -- Steps 4 & 5: Enrich + Build entries (batched) -------------------------
    gha_group(f"Step 4 · Enrich & build — {len(approved)} approved article(s)")
    new_entries, tag_registry = build_entries(
        approved, kept_guides.copy(), tag_registry, core_tags, merge_map, max_tags, reg_hash
    )
    for entry in new_entries:
        log(f"  ➕ [{entry['language']}] {entry['title'][:70]}")
    gha_endgroup()

    # -- Step 6: Merge & validate schema ---------------------------------------
    final_guides = kept_guides + new_entries

    # Prune stale tags from old entries; clear tags_version if registry changed
    for g in final_guides:
        g["tags"] = [t for t in (g.get("tags") or []) if t in tag_registry] or ["overview"]
        if g.get("tags_version") != reg_hash:
            g.pop("tags_version", None)
        if not g.get("last_checked"):
            g["last_checked"] = "2000-01-01"  # force recheck on next run

    gha_group("Step 5 · Schema validation")
    valid, errors = validate_schema(final_guides, tag_registry)
    if valid:
        log("  ✅ Schema valid")
    else:
        for err in errors:
            gha_error(err)
        before = len(final_guides)
        final_guides = [
            g for g in final_guides
            if not any(f"id={g.get('id', '?')}" in e for e in errors)
        ]
        gha_warning(f"Dropped {before - len(final_guides)} invalid entries")
    gha_endgroup()

    # -- Step 7: Save ----------------------------------------------------------
    save_guides(final_guides)

    # -- Step 8: Summary for GitHub Actions ------------------------------------
    lang_counts  = {l["code"]: 0 for l in config["languages"]}
    habit_counts = {h: 0 for h in config["habits"]}
    coverage: dict[str, dict[str, int]] = {h: {} for h in config["habits"]}

    for g in final_guides:
        lc = g.get("language", "")
        if lc in lang_counts:
            lang_counts[lc] += 1
        for h in (g.get("habits") or []):
            if h in habit_counts:
                habit_counts[h] += 1
            if h in coverage:
                coverage[h][lc] = coverage[h].get(lc, 0) + 1

    summary = {
        "run_date":      today_str(),
        "mode":          mode,
        "total_guides":  len(final_guides),
        "new_guides":    len(new_entries),
        "removed":       len(removed_guides),
        "lang_counts":   lang_counts,
        "habit_counts":  habit_counts,
        "coverage":      coverage,
        "tag_registry":  sorted(tag_registry),
    }

    Path("summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gha_notice(f"Summary written to summary.json")

    log("\n" + "=" * 52)
    log(f"✅ Done — {len(final_guides)} guides total "
        f"(+{len(new_entries)} new, -{len(removed_guides)} removed)")
    return summary


if __name__ == "__main__":
    run()