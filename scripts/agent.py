"""
Habit Tracker Guides Agent
Runs weekly to discover, validate, audit and update guides JSON.
"""

import os
import json
import httpx
import anthropic
from datetime import datetime
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

GUIDES_FILE = "guides.json"
VALIDATION_THRESHOLD = 7.0                # articles below this score are rejected
SEARCH_TOPICS = [                         # ← customise these to your app's topics
    "bad habits risks and side effects",
    "habit breaking science research",
    "dopamine addiction psychology",
    "behaviour change techniques",
    "habit loop neuroscience",
    "overcoming addictive habits",
    "screen time social media addiction risks",
    "sugar addiction health effects",
    "sleep habit improvement science",
]
TRUSTED_DOMAINS = [                       # ← articles from these skip strict checks
    "apa.org", "who.int", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "healthline.com", "mayoclinic.org", "psychologytoday.com",
    "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov",
]

# ─── Anthropic client ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-opus-4-5"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_guides() -> list[dict]:
    path = Path(GUIDES_FILE)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("guides", [])


def save_guides(guides: list[dict]) -> None:
    path = Path(GUIDES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "guides": guides,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved {len(guides)} guides to {GUIDES_FILE}")


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


def claude(prompt: str, system: str = "") -> str:
    """Simple Claude call, returns text."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": MODEL, "max_tokens": 4096, "messages": messages}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def claude_with_search(prompt: str, system: str = "") -> str:
    """Claude call with web search tool enabled."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": messages,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    if system:
        kwargs["system"] = system

    # Agentic loop — Claude may call the search tool multiple times
    while True:
        resp = client.messages.create(**kwargs)
        if resp.stop_reason == "end_turn":
            texts = [b.text for b in resp.content if hasattr(b, "text")]
            return "\n".join(texts)
        if resp.stop_reason == "tool_use":
            # Append assistant turn + tool results, then continue
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "",   # SDK handles actual search
                    })
            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages
        else:
            break
    return ""

# ─── Step 1: Discover ──────────────────────────────────────────────────────────

def discover_new_articles(existing_urls: set[str]) -> list[dict]:
    print("\n🔍 Step 1: Discovering new articles...")
    all_found = []

    for topic in SEARCH_TOPICS:
        print(f"   Searching: {topic}")
        prompt = f"""Search the web for high-quality articles about: "{topic}"
        
Find 3-5 relevant, credible articles published in the last 2 years.
Return ONLY a JSON array (no markdown, no explanation) like:
[
  {{"url": "https://...", "title": "...", "summary": "one sentence", "source": "domain.com"}},
  ...
]

Exclude these already-known URLs: {list(existing_urls)[:20]}
"""
        try:
            raw = claude_with_search(prompt)
            # Strip any accidental markdown fences
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            found = json.loads(raw)
            all_found.extend(found)
        except Exception as e:
            print(f"   ⚠️  Search failed for '{topic}': {e}")

    # Deduplicate by URL
    seen = set(existing_urls)
    unique = []
    for item in all_found:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    print(f"   Found {len(unique)} new candidate articles")
    return unique

# ─── Step 2: Validate ─────────────────────────────────────────────────────────

def validate_article(article: dict) -> dict:
    url = article["url"]
    trusted = is_trusted(url)

    prompt = f"""You are a medical/health content quality reviewer for a habit-tracking app.

Evaluate this article for inclusion in the app's resource library:
URL: {url}
Title: {article.get('title', 'Unknown')}
Summary: {article.get('summary', '')}
Trusted source: {trusted}

Score it 1-10 on each dimension (10 = best):
- credibility: Is the source reputable? (publisher, author credentials)
- accuracy: Are claims evidence-based? No misinformation?
- relevance: Does it relate to bad habits, risks, behaviour change, health?
- recency: Is it current (prefer last 2 years)?

Return ONLY JSON, no markdown:
{{
  "credibility": 0,
  "accuracy": 0,
  "relevance": 0,
  "recency": 0,
  "overall": 0.0,
  "verdict": "approved" or "rejected",
  "flags": ["list any concerns"],
  "rejection_reason": "only if rejected"
}}
"""
    try:
        raw = claude(prompt)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        scores = json.loads(raw)
        overall = (
            scores["credibility"] + scores["accuracy"] +
            scores["relevance"] + scores["recency"]
        ) / 4.0
        scores["overall"] = round(overall, 2)
        # Trusted domains get a +0.5 bonus
        if trusted:
            scores["overall"] = min(10.0, scores["overall"] + 0.5)
        scores["verdict"] = "approved" if scores["overall"] >= VALIDATION_THRESHOLD else "rejected"
        return scores
    except Exception as e:
        return {
            "credibility": 0, "accuracy": 0, "relevance": 0, "recency": 0,
            "overall": 0, "verdict": "rejected",
            "flags": [], "rejection_reason": f"Validation error: {e}"
        }


def validate_new_articles(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    print("\n🧪 Step 2: Validating new articles...")
    approved, rejected = [], []

    for article in candidates:
        url = article["url"]
        if not is_url_live(url):
            print(f"   ❌ Dead link skipped: {url}")
            rejected.append({**article, "rejection_reason": "URL not accessible"})
            continue

        scores = validate_article(article)
        article["validation"] = scores

        if scores["verdict"] == "approved":
            print(f"   ✅ Approved ({scores['overall']:.1f}): {article.get('title', url)}")
            approved.append(article)
        else:
            reason = scores.get("rejection_reason", "Below quality threshold")
            print(f"   ❌ Rejected ({scores['overall']:.1f}): {article.get('title', url)} — {reason}")
            rejected.append(article)

    print(f"   Approved: {len(approved)} | Rejected: {len(rejected)}")
    return approved, rejected

# ─── Step 3: Audit existing ───────────────────────────────────────────────────

def audit_existing_guides(guides: list[dict]) -> tuple[list[dict], list[dict]]:
    print("\n✅ Step 3: Auditing existing guides...")
    kept, removed = [], []

    for guide in guides:
        url = guide.get("url", "")

        # Check if still live
        if not is_url_live(url):
            print(f"   🗑️  Removed (dead link): {url}")
            guide["removal_reason"] = "URL no longer accessible"
            removed.append(guide)
            continue

        # Ask Claude if it's still relevant and accurate
        prompt = f"""Review this existing article in a habit-tracking app's resource library.
URL: {url}
Title: {guide.get('title', '')}
Added: {guide.get('added_date', 'unknown')}

Is this article still accurate, relevant, and up to date as of {datetime.utcnow().strftime('%B %Y')}?
Consider: outdated research, retracted studies, superseded guidelines, broken/redirected domains.

Return ONLY JSON:
{{"keep": true or false, "reason": "brief explanation"}}
"""
        try:
            raw = claude(prompt)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(raw)
            if result.get("keep", True):
                kept.append(guide)
            else:
                guide["removal_reason"] = result.get("reason", "Outdated or irrelevant")
                print(f"   🗑️  Removed ({result['reason']}): {url}")
                removed.append(guide)
        except Exception:
            kept.append(guide)  # Keep on error — safe default

    print(f"   Kept: {len(kept)} | Removed: {len(removed)}")
    return kept, removed

# ─── Step 4: Build & validate JSON ────────────────────────────────────────────

def build_guide_entry(article: dict) -> dict:
    return {
        "url": article["url"],
        "title": article.get("title", ""),
        "summary": article.get("summary", ""),
        "source": article.get("source", ""),
        "added_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "validation_score": article.get("validation", {}).get("overall", 0),
        "tags": [],   # You can extend Claude to generate tags too
    }


def validate_json_schema(guides: list[dict]) -> bool:
    required_keys = {"url", "title", "summary", "source", "added_date"}
    for guide in guides:
        if not required_keys.issubset(guide.keys()):
            return False
        if not guide["url"].startswith("http"):
            return False
    return True

# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("🚀 Starting Guides Agent")
    print("=" * 50)

    existing_guides = load_guides()
    existing_urls = {g["url"] for g in existing_guides}
    print(f"📚 Loaded {len(existing_guides)} existing guides")

    # Step 1 — Discover
    candidates = discover_new_articles(existing_urls)

    # Step 2 — Validate new
    approved_new, rejected_new = validate_new_articles(candidates)

    # Step 3 — Audit existing
    kept_existing, removed_existing = audit_existing_guides(existing_guides)

    # Step 4 — Merge & save
    new_entries = [build_guide_entry(a) for a in approved_new]
    final_guides = kept_existing + new_entries

    if not validate_json_schema(final_guides):
        raise ValueError("❌ JSON schema validation failed!")

    save_guides(final_guides)

    summary = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "total_guides": len(final_guides),
        "new_added": len(new_entries),
        "removed": len(removed_existing),
        "rejected_new": len(rejected_new),
        "removed_details": [
            {"url": g["url"], "reason": g.get("removal_reason", "")}
            for g in removed_existing
        ],
        "new_details": [
            {"url": g["url"], "title": g.get("title", ""), "score": g.get("validation", {}).get("overall", 0)}
            for g in new_entries
        ],
    }

    print("\n📊 Summary:")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
