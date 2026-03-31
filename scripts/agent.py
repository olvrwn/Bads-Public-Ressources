"""
Habit Tracker Guides Agent
==========================
Triggered every week by GitHub Actions (see .github/workflows/guides.yml).
Run locally with: python scripts/agent.py

Pipeline:
  1. Load      -- read guides.json + config.json
  2. Audit     -- parallel liveness check on all existing guides (GET, follow redirects)
                  Dead = 404/410 after redirects. Transient errors = keep.
  3. Discover  -- gap-aware serial search via Claude web_search tool
                  Gaps (<MIN_GUIDES_PER_COMBO) get a more aggressive prompt (4 results)
                  Covered combos get a light refresh prompt (2 results)
                  Serial with 2s throttle to respect Anthropic free tier limits
  4. Validate  -- two-gate quality check:
                  Gate 1: domain tier (Tier1=9.0, Tier2=7.5, auto-pass)
                  Gate 2: unknown domains → Claude 3-binary-check on fetched snippet
  5. Enrich    -- assign tags via Claude haiku; propose new tags (capped at max_tags)
  6. Save      -- merge, schema-validate, persist flat JSON array
  7. Summary   -- write summary.json for GitHub Actions workflow

Rate limits (Anthropic free tier):
  - 30,000 input tokens / minute
  - 50 requests / minute
  → MAX_SEARCH_WORKERS = 1 with 2s sleep between calls (~30 req/min)
"""

import os
import re
import json
import time
import httpx
import anthropic

from datetime import datetime, timezone
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

# Anthropic free tier: serial search with throttle to avoid 429 cascade.
# Each claude-sonnet search does multiple internal tool-use round trips,
# easily hitting the 30k token/min or 50 req/min ceiling when parallelised.
MAX_SEARCH_WORKERS   = 1
MAX_LIVENESS_WORKERS = 20  # httpx only — safe to parallelise heavily
SEARCH_THROTTLE_SECS = 2   # sleep between search calls

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
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)
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


def claude_fast(prompt: str, system: str = "") -> str:
    """Single-turn Claude haiku call for scoring / tagging."""
    kwargs: dict = {
        "model":     MODEL_FAST,
        "max_tokens": 1024,
        "messages":  [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content.text


def claude_search(prompt: str, system: str = "") -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model":      MODEL_SEARCH,
        "max_tokens": 4096,
        "messages":   messages,
        "tools":      [{"type": "web_search_20250305", "name": "web_search"}],
    }
    if system:
        kwargs["system"] = system

    while True:
        resp = client.messages.create(**kwargs)
        if resp.stop_reason == "end_turn":
            parts = []
            for b in resp.content:
                text = getattr(b, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, list):
                    # web_search blocks sometimes return text as a list of strings
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
        else:
            break
    return ""

# ---------------------------------------------------------------------------
# Step 1 — Audit existing guides
# ---------------------------------------------------------------------------

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
        if resp.status_code in {404, 410}:
            return guide, "dead"
        return guide, "transient"
    except Exception:
        return guide, "transient"


def audit_guides(guides: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Parallel liveness check. Removes guides that definitively return 404/410
    after following all redirects. Transient errors = keep.
    Returns (kept, removed).
    """
    gha_group(f"Step 1 · Audit — checking {len(guides)} existing guides")
    kept, removed = [], []

    with ThreadPoolExecutor(max_workers=MAX_LIVENESS_WORKERS) as pool:
        futures = {pool.submit(_check_liveness, g): g for g in guides}
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
                kept.append(guide)

    log(f"  Kept: {len(kept)} | Removed: {len(removed)}")
    gha_endgroup()
    return kept, removed

# ---------------------------------------------------------------------------
# Step 2 — Gap-aware discovery
# ---------------------------------------------------------------------------

def _compute_counts(
    guides: list[dict],
    habits: list[str],
    languages: list[dict],
) -> dict[tuple[str, str], int]:
    """Count existing guides per (habit, lang_code) pair."""
    counts: dict[tuple[str, str], int] = {}
    for g in guides:
        lc = g.get("language", "")
        for h in (g.get("habits") or []):
            counts[(h, lc)] = counts.get((h, lc), 0) + 1
    return counts


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
    Sleeps SEARCH_THROTTLE_SECS before the API call to respect free-tier limits.
    """
    time.sleep(SEARCH_THROTTLE_SECS)

    count   = 4 if is_gap else 2
    urgency = (
        "This is a COVERAGE GAP — we have almost no content for this combination. "
        "Be very flexible with source type. Any accurate, helpful source counts."
    ) if is_gap else (
        "We already have some content. Only return genuinely new, high-quality sources."
    )

    prompt = f"""Search the web for {count} high-quality {lang['locale']} articles about "{habit}".

{urgency}

Requirements:
- Article MUST be written entirely in {lang['name']} (language code: {lang['code']})
- Must cover at least one of: health effects, addiction, risks, treatment, prevention, science
- Acceptable sources: government, WHO/UN, universities, established health NGOs,
  reputable clinics, well-known health publishers, peer-reviewed journals
- Do NOT include: forums, Reddit, social media, anonymous blogs, product/sales pages

Do NOT include any of these already-known URLs:
{sorted(known_urls)[:25]}

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
) -> list[dict]:
    """
    Gap-aware discovery. One serial wave, no second pass.
    Gaps (below MIN_GUIDES_PER_COMBO) → aggressive prompt (4 results).
    Covered combos → light refresh (2 results).
    Serial with SEARCH_THROTTLE_SECS between calls to respect free-tier limits.
    """
    habits    = config["habits"]
    languages = config["languages"]
    min_combo = config.get("min_guides_per_combo", 3)

    counts     = _compute_counts(existing_guides, habits, languages)
    known_urls: set[str] = {normalise_url(g["url"]) for g in existing_guides}

    tasks: list[tuple[str, dict, bool]] = []
    gap_count = 0
    for lang in languages:
        for habit in habits:
            current = counts.get((habit, lang["code"]), 0)
            is_gap  = current < min_combo
            if is_gap:
                gap_count += 1
            tasks.append((habit, lang, is_gap))

    gha_group(
        f"Step 2 · Discover — {len(tasks)} combos "
        f"({gap_count} gaps, {len(tasks) - gap_count} refreshes) "
        f"| serial with {SEARCH_THROTTLE_SECS}s throttle"
    )

    all_found: list[dict] = []
    seen_norm: set[str]   = set(known_urls)

    # MAX_SEARCH_WORKERS=1 makes this effectively serial, but kept as a pool
    # so it's trivial to increase once you upgrade to a paid tier.
    with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as pool:
        future_map = {
            pool.submit(_search_one_combo, habit, lang, is_gap, known_urls): (habit, lang, is_gap)
            for habit, lang, is_gap in tasks
        }
        for future in as_completed(future_map):
            habit, lang, is_gap = future_map[future]
            label = "GAP" if is_gap else "refresh"
            try:
                results = future.result()
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

    log(f"\n  Total candidates found: {len(all_found)}")
    gha_endgroup()
    return all_found

# ---------------------------------------------------------------------------
# Step 3 — Two-gate validation
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


def _gate2_claude_check(article: dict) -> tuple[bool, str]:
    """
    Three binary checks via Claude on a fetched page snippet.
    All three must pass. Returns (passed, reason).
    """
    url     = article["url"]
    snippet = fetch_snippet(url)
    if not snippet:
        return False, "Could not fetch page content for review"

    prompt = f"""You are a health content safety reviewer for a habit-tracking app aimed at harm reduction.

Review this article and answer three yes/no questions honestly.

URL      : {url}
Title    : {article.get('title', '')}
Snippet  : {snippet}

Answer each question with exactly "yes" or "no":
1. factual   — Is the health/science information grounded in evidence? (no sensationalism, no miracle claims)
2. qualified — Does the content appear written or reviewed by a qualified person or credible organisation?
3. useful    — Would this genuinely help someone understand or manage this habit/substance?

Return ONLY valid JSON — no markdown:
{{
  "factual": "yes",
  "qualified": "yes",
  "useful": "yes",
  "notes": "one short sentence explaining any concern, or null"
}}
"""
    try:
        result = json.loads(strip_fence(claude_fast(prompt)))
        passed = all(
            result.get(k, "no").strip().lower() == "yes"
            for k in ("factual", "qualified", "useful")
        )
        reason = result.get("notes") or ("All checks passed" if passed else "Failed one or more quality checks")
        return passed, reason
    except Exception as e:
        return False, f"Validation error: {e}"


def validate_candidates(
    candidates: list[dict],
    existing_guides: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Gate 1: domain tier check (auto-pass Tier1/Tier2).
    Gate 2: Claude 3-binary-check for unknown domains.
    Also deduplicates against existing_guides.
    Returns (approved, rejected).
    """
    gha_group(f"Step 3 · Validate — {len(candidates)} candidate(s)")
    approved, rejected = [], []

    for article in candidates:
        url = article.get("url", "")

        # Dedup check
        if is_duplicate(article, existing_guides):
            log(f"  ♻️  Duplicate skipped: {url}")
            rejected.append({**article, "_rejection_reason": "Duplicate of existing guide"})
            continue

        # Liveness check before scoring
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=10)
            if resp.status_code in {404, 410}:
                log(f"  ✗ Dead link skipped: {url}")
                rejected.append({**article, "_rejection_reason": "URL is dead (404/410)"})
                continue
        except Exception:
            gha_warning(f"Liveness check failed for {url} — skipping candidate")
            continue

        # Gate 1: trusted domain tier
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

        # Gate 2: Claude content check for unknown domains
        passed, reason = _gate2_claude_check(article)
        if passed:
            article["_score"] = 6.0
            log(f"  ✅ Gate-2 passed: {article.get('title', url)}")
            approved.append(article)
        else:
            log(f"  ✗ Gate-2 failed ({reason}): {article.get('title', url)}")
            rejected.append({**article, "_rejection_reason": reason})

    log(f"\n  Approved: {len(approved)} | Rejected: {len(rejected)}")
    gha_endgroup()
    return approved, rejected

# ---------------------------------------------------------------------------
# Step 4 — Tag enrichment
# ---------------------------------------------------------------------------

def normalise_tag(tag: str, merge_map: dict) -> str:
    return merge_map.get(tag.lower().strip(), tag.lower().strip())


def enrich_tags(
    article: dict,
    tag_registry: set[str],
    core_tags: set[str],
    merge_map: dict,
    max_tags: int,
) -> tuple[list[str], set[str]]:
    """
    Ask Claude to assign 1-4 tags. It may propose one new tag if none fit.
    New tags are added to registry only if registry is below max_tags.
    Core tags are never dropped.
    Returns (tags_for_article, updated_registry).
    """
    prompt = f"""Tag this health article for a habit-tracking app.

Current tag registry (use these first):
{sorted(tag_registry)}

Core tags that always exist (never remove these from the registry):
{sorted(core_tags)}

Article:
  Title      : {article.get('title', '')}
  Description: {article.get('description', '')}
  Habits     : {article.get('habits', [])}

Tag merge rules — rewrite these before returning:
{json.dumps(merge_map, indent=2)}

Instructions:
1. Pick 1-4 tags from the registry that best fit this article.
2. Only propose a new tag (single lowercase word) if NO existing tag is even a rough fit.
3. Apply the merge rules above before returning any tag.

Return ONLY valid JSON — no markdown:
{{
  "tags": ["tag1", "tag2"],
  "new_tag": null
}}
"""
    try:
        result  = json.loads(strip_fence(claude_fast(prompt)))
        raw_tags: list[str] = result.get("tags") or []
        new_tag: str | None = result.get("new_tag") or None

        tags = list(dict.fromkeys(normalise_tag(t, merge_map) for t in raw_tags))
        tags = [t for t in tags if t in tag_registry]

        updated_registry = set(tag_registry)
        if new_tag:
            new_tag = normalise_tag(new_tag, merge_map)
            if new_tag not in updated_registry:
                if len(updated_registry) < max_tags:
                    log(f"  🏷️  New tag added to registry: '{new_tag}'")
                    updated_registry.add(new_tag)
                else:
                    gha_warning(
                        f"Tag registry at cap ({max_tags}). "
                        f"Proposed tag '{new_tag}' not added."
                    )

        return tags if tags else ["overview"], updated_registry
    except Exception:
        return ["overview"], tag_registry

# ---------------------------------------------------------------------------
# Step 5 — Build final guide entry
# ---------------------------------------------------------------------------

def next_id(existing: list[dict]) -> str:
    if not existing:
        return "1"
    numeric = [int(g["id"]) for g in existing if str(g.get("id", "")).isdigit()]
    return str(max(numeric) + 1) if numeric else "1"


def build_entry(
    article: dict,
    guide_id: str,
    tag_registry: set[str],
    core_tags: set[str],
    merge_map: dict,
    max_tags: int,
) -> tuple[dict, set[str]]:
    reading_minutes: int | None = None
    try:
        val = article.get("estimatedReadingMinutes")
        if val is not None:
            reading_minutes = int(val)
    except (TypeError, ValueError):
        pass

    tags, updated_registry = enrich_tags(
        article, tag_registry, core_tags, merge_map, max_tags
    )

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
    }
    return entry, updated_registry

# ---------------------------------------------------------------------------
# Step 6 — Schema validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS   = {"id", "title", "description", "language", "source", "url", "habits"}
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

def run() -> dict:
    log("🚀 Habit Guides Agent starting")
    log("=" * 52)

    # -- Load ------------------------------------------------------------------
    config = load_config()
    global VALID_LANG_CODES
    VALID_LANG_CODES = {l["code"] for l in config["languages"]}

    tag_registry: set[str] = set(config["tag_registry"])
    core_tags:    set[str] = set(config["core_tags"])
    merge_map:    dict     = config["tag_merge_map"]
    max_tags:     int      = config.get("max_tags", 15)

    existing_guides = load_guides()
    log(f"📑 Loaded {len(existing_guides)} existing guides")
    log(f"🏷️  Tag registry ({len(tag_registry)}): {sorted(tag_registry)}")

    # -- Step 1: Audit ---------------------------------------------------------
    kept_guides, removed_guides = audit_guides(existing_guides)

    # -- Step 2: Discover ------------------------------------------------------
    candidates = discover_articles(kept_guides, config)

    # -- Step 3: Validate ------------------------------------------------------
    approved, rejected = validate_candidates(candidates, kept_guides, config)

    # -- Steps 4 & 5: Enrich + Build entries -----------------------------------
    gha_group(f"Step 4 · Enrich & build — {len(approved)} approved article(s)")
    new_entries: list[dict] = []
    id_pool = kept_guides.copy()

    for article in approved:
        guide_id = next_id(id_pool)
        entry, tag_registry = build_entry(
            article, guide_id, tag_registry, core_tags, merge_map, max_tags
        )
        log(f"  ➕ [{entry['language']}] {entry['title'][:70]}")
        new_entries.append(entry)
        id_pool.append(entry)

    gha_endgroup()

    # -- Step 6: Merge & validate schema ---------------------------------------
    final_guides = kept_guides + new_entries

    # Ensure all guide tags are in registry (prune stale tags from old entries)
    for g in final_guides:
        g["tags"] = [t for t in (g.get("tags") or []) if t in tag_registry] or ["overview"]

    gha_group("Step 5 · Schema validation")
    valid, errors = validate_schema(final_guides, tag_registry)
    if valid:
        log("  ✅ Schema valid")
    else:
        for err in errors:
            gha_error(err)
        # Best-effort: drop invalid entries rather than saving corrupt data
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

    min_combo  = config.get("min_guides_per_combo", 3)
    still_gaps = [
        f"{h}/{lc}"
        for h in config["habits"]
        for lang in config["languages"]
        for lc in [lang["code"]]
        if coverage.get(h, {}).get(lc, 0) < min_combo
    ]

    summary = {
        "date":                   utc_now().strftime("%Y-%m-%d"),
        "total_guides":           len(final_guides),
        "new_added":              len(new_entries),
        "removed":                len(removed_guides),
        "rejected_new":           len(rejected),
        "guides_by_language":     lang_counts,
        "guides_by_habit":        habit_counts,
        "coverage_below_minimum": still_gaps,
        "new_details": [
            {
                "id":          g["id"],
                "url":         g["url"],
                "title":       g.get("title", ""),
                "description": g.get("description") or "",
                "language":    g.get("language", ""),
                "habits":      g.get("habits") or [],
                "tags":        g.get("tags") or [],
            }
            for g in new_entries
        ],
        "removed_details": [
            {
                "url":    g["url"],
                "title":  g.get("title", ""),
                "reason": g.get("_removal_reason", "Unknown"),
            }
            for g in removed_guides
        ],
    }

    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    gha_group("📊 Run Summary")
    log(f"  Total guides : {summary['total_guides']}")
    log(f"  New added    : {summary['new_added']}")
    log(f"  Removed      : {summary['removed']}")
    log(f"  Rejected new : {summary['rejected_new']}")
    log(f"  Coverage gaps: {len(still_gaps)}")
    if still_gaps:
        gha_warning(f"Combos still below minimum ({min_combo}): {still_gaps}")
    gha_endgroup()

    return summary


if __name__ == "__main__":
    run()