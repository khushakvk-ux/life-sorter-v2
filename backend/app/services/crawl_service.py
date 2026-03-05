"""
═══════════════════════════════════════════════════════════════
CRAWL SERVICE — Async business website crawler & analyzer
═══════════════════════════════════════════════════════════════
Crawls a user's business website in the background:
  1. Fetches homepage → extracts title, meta description, H1s, nav links
  2. Crawls up to 5 internal pages (about, pricing, products, contact, blog)
  3. Extracts tech stack signals, CTA patterns, social links, schema markup
  4. Runs lightweight SEO check (meta tags, page speed signal, mobile viewport)
  5. Generates a compressed crawl summary (5 bullet points)
  6. Stores raw + summary data in the session

Designed to run in the background while the user answers Scale Questions.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
import structlog

from app.config import get_settings
from app.services import session_store
from app.services.openai_service import _get_client

logger = structlog.get_logger()

# Social media domains (for url_type detection)
SOCIAL_DOMAINS = {
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "linkedin.com", "tiktok.com", "youtube.com", "pinterest.com",
    "threads.net",
}

# Internal page path patterns to look for
INTERNAL_PAGE_PATTERNS = [
    (r"about|who-we-are|our-story|team", "about"),
    (r"pricing|plans|packages", "pricing"),
    (r"product|service|solution|features|what-we-do", "products"),
    (r"contact|get-in-touch|reach-us|support", "contact"),
    (r"blog|news|articles|insights|resources", "blog"),
]

# Max pages to crawl beyond homepage
MAX_INTERNAL_PAGES = 5

# HTTP client config
CRAWL_TIMEOUT = 15.0
CRAWL_USER_AGENT = "Mozilla/5.0 (compatible; IkshanBot/2.0; +https://ikshan.ai)"


def detect_url_type(url: str) -> str:
    """Detect if a URL is a social profile or regular website."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    for social in SOCIAL_DOMAINS:
        if domain == social or domain.endswith("." + social):
            return "social_profile"
    return "website"


def _extract_meta(html: str) -> dict:
    """Extract title, meta description, H1s, and viewport from HTML."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    meta_desc = ""
    meta_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        html, re.IGNORECASE | re.DOTALL,
    )
    if not meta_match:
        meta_match = re.search(
            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
            html, re.IGNORECASE | re.DOTALL,
        )
    if meta_match:
        meta_desc = meta_match.group(1).strip()

    h1s = re.findall(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    h1s = [re.sub(r"<[^>]+>", "", h).strip() for h in h1s]

    has_viewport = bool(re.search(r'<meta[^>]+name=["\']viewport["\']', html, re.IGNORECASE))
    has_meta = bool(title and meta_desc)

    return {
        "title": title[:200],
        "meta_desc": meta_desc[:500],
        "h1s": h1s[:5],
        "has_viewport": has_viewport,
        "has_meta": has_meta,
    }


def _extract_nav_links(html: str, base_url: str) -> list[str]:
    """Extract navigation links (internal) from HTML."""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.lower()

    # Find all href attributes
    hrefs = re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\']', html, re.IGNORECASE)
    internal_links = set()
    for href in hrefs:
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc.lower() == base_domain and parsed.path != "/" and parsed.path:
            # Clean the URL
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            internal_links.add(clean_url)

    return list(internal_links)[:30]  # Cap at 30 for processing


def _extract_social_links(html: str) -> list[str]:
    """Extract social media links from HTML."""
    hrefs = re.findall(r'<a[^>]+href=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
    socials = []
    for href in hrefs:
        parsed = urlparse(href)
        domain = parsed.netloc.lower().replace("www.", "")
        for social in SOCIAL_DOMAINS:
            if domain == social or domain.endswith("." + social):
                socials.append(href)
                break
    return list(set(socials))[:10]


def _extract_schema_markup(html: str) -> list[str]:
    """Extract JSON-LD schema types from HTML."""
    schemas = []
    ld_blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.IGNORECASE | re.DOTALL,
    )
    for block in ld_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                schema_type = data.get("@type", "")
                if schema_type:
                    schemas.append(schema_type if isinstance(schema_type, str) else str(schema_type))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type"):
                        schemas.append(str(item["@type"]))
        except (json.JSONDecodeError, Exception):
            pass
    return schemas[:10]


def _detect_tech_signals(html: str) -> list[str]:
    """Detect technology stack signals from HTML source."""
    signals = []
    tech_patterns = [
        (r"wp-content|wordpress", "WordPress"),
        (r"shopify\.com|cdn\.shopify", "Shopify"),
        (r"squarespace\.com|sqsp\.net", "Squarespace"),
        (r"wix\.com|wixstatic", "Wix"),
        (r"webflow\.com|webflow\.io", "Webflow"),
        (r"react|__next|_next/static", "React/Next.js"),
        (r"vue\.js|vuejs|__vue__", "Vue.js"),
        (r"angular|ng-version", "Angular"),
        (r"gatsby", "Gatsby"),
        (r"hubspot\.com|hs-scripts", "HubSpot"),
        (r"mailchimp\.com", "Mailchimp"),
        (r"intercom\.com|intercom-", "Intercom"),
        (r"drift\.com|drift-frame", "Drift"),
        (r"zendesk\.com|zdassets", "Zendesk"),
        (r"google-analytics|gtag|ga\.js|UA-", "Google Analytics"),
        (r"googletagmanager", "Google Tag Manager"),
        (r"hotjar\.com", "Hotjar"),
        (r"stripe\.com|stripe\.js", "Stripe"),
        (r"cloudflare", "Cloudflare"),
        (r"bootstrap|getbootstrap", "Bootstrap"),
        (r"tailwindcss|tailwind", "Tailwind CSS"),
        (r"calendly\.com", "Calendly"),
        (r"facebook\.net|fbq|fb-pixel", "Facebook Pixel"),
    ]
    for pattern, name in tech_patterns:
        if re.search(pattern, html, re.IGNORECASE):
            signals.append(name)
    return list(set(signals))


def _extract_cta_patterns(html: str) -> list[str]:
    """Extract CTA button text patterns from HTML."""
    # Look for button text and CTA-like link text
    buttons = re.findall(r"<button[^>]*>(.*?)</button>", html, re.IGNORECASE | re.DOTALL)
    cta_links = re.findall(
        r'<a[^>]+class=["\'][^"\']*(?:btn|cta|button)[^"\']*["\'][^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL,
    )
    all_ctas = buttons + cta_links
    cleaned = []
    for cta in all_ctas:
        text = re.sub(r"<[^>]+>", "", cta).strip()
        if text and len(text) < 50 and len(text) > 2:
            cleaned.append(text)
    return list(set(cleaned))[:10]


def _check_sitemap(html: str, base_url: str) -> bool:
    """Quick check if a sitemap reference exists."""
    return bool(re.search(r"sitemap\.xml", html, re.IGNORECASE))


def _select_pages_to_crawl(nav_links: list[str]) -> list[dict]:
    """Select the most relevant internal pages to crawl."""
    selected = []
    used_types = set()

    for link in nav_links:
        path = urlparse(link).path.lower()
        for pattern, page_type in INTERNAL_PAGE_PATTERNS:
            if page_type not in used_types and re.search(pattern, path):
                selected.append({"url": link, "type": page_type})
                used_types.add(page_type)
                break
        if len(selected) >= MAX_INTERNAL_PAGES:
            break

    return selected


def _html_to_text(html: str, max_chars: int = 2000) -> str:
    """Convert HTML to plain text, stripping scripts/styles/tags."""
    clean = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_chars]


async def _fetch_page(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Fetch a single page, returns HTML or None on failure."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.debug("Failed to fetch page", url=url, error=str(e))
        return None


async def crawl_website(website_url: str) -> dict:
    """
    Crawl a business website and extract structured data.

    Returns:
        {
            "homepage": { "title", "meta_desc", "h1s", "nav_links" },
            "pages_crawled": [ { "url", "type", "key_content" } ],
            "tech_signals": [],
            "cta_patterns": [],
            "social_links": [],
            "schema_markup": [],
            "seo_basics": { "has_meta", "has_viewport", "has_sitemap" }
        }
    """
    result = {
        "homepage": {"title": "", "meta_desc": "", "h1s": [], "nav_links": []},
        "pages_crawled": [],
        "tech_signals": [],
        "cta_patterns": [],
        "social_links": [],
        "schema_markup": [],
        "seo_basics": {"has_meta": False, "has_viewport": False, "has_sitemap": False},
    }

    async with httpx.AsyncClient(
        timeout=CRAWL_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": CRAWL_USER_AGENT},
    ) as client:
        # ── Step 1: Fetch homepage ─────────────────────────────
        homepage_html = await _fetch_page(client, website_url)
        if not homepage_html:
            logger.warning("Could not fetch homepage", url=website_url)
            return result

        # Extract homepage metadata
        meta = _extract_meta(homepage_html)
        nav_links = _extract_nav_links(homepage_html, website_url)
        social_links = _extract_social_links(homepage_html)
        schema_markup = _extract_schema_markup(homepage_html)
        tech_signals = _detect_tech_signals(homepage_html)
        cta_patterns = _extract_cta_patterns(homepage_html)
        has_sitemap = _check_sitemap(homepage_html, website_url)

        result["homepage"] = {
            "title": meta["title"],
            "meta_desc": meta["meta_desc"],
            "h1s": meta["h1s"],
            "nav_links": nav_links[:15],
        }
        result["tech_signals"] = tech_signals
        result["cta_patterns"] = cta_patterns
        result["social_links"] = social_links
        result["schema_markup"] = schema_markup
        result["seo_basics"] = {
            "has_meta": meta["has_meta"],
            "has_viewport": meta["has_viewport"],
            "has_sitemap": has_sitemap,
        }

        # ── Step 2: Crawl internal pages ───────────────────────
        pages_to_crawl = _select_pages_to_crawl(nav_links)

        if pages_to_crawl:
            # Fetch pages concurrently
            tasks = [_fetch_page(client, p["url"]) for p in pages_to_crawl]
            pages_html = await asyncio.gather(*tasks, return_exceptions=True)

            for page_info, html_or_err in zip(pages_to_crawl, pages_html):
                if isinstance(html_or_err, str) and html_or_err:
                    key_content = _html_to_text(html_or_err, max_chars=1500)
                    result["pages_crawled"].append({
                        "url": page_info["url"],
                        "type": page_info["type"],
                        "key_content": key_content,
                    })

                    # Also extract tech/social from sub-pages
                    result["tech_signals"] = list(
                        set(result["tech_signals"] + _detect_tech_signals(html_or_err))
                    )
                    sub_socials = _extract_social_links(html_or_err)
                    result["social_links"] = list(
                        set(result["social_links"] + sub_socials)
                    )

    logger.info(
        "Website crawl complete",
        url=website_url,
        pages_crawled=len(result["pages_crawled"]),
        tech_signals=len(result["tech_signals"]),
    )
    return result


async def generate_crawl_summary(crawl_raw: dict, website_url: str) -> dict:
    """
    Generate a compressed 5-bullet summary from raw crawl data using GPT.

    Returns:
        {
            "points": ["bullet 1", "bullet 2", ...],
            "crawl_status": "complete",
            "completed_at": "ISO-timestamp"
        }
    """
    settings = get_settings()
    if not settings.openai_api_key_active:
        # Fallback: generate basic summary without GPT
        return _generate_fallback_summary(crawl_raw)

    client = _get_client()

    # Build context from raw crawl
    context_parts = []
    hp = crawl_raw.get("homepage", {})
    if hp.get("title"):
        context_parts.append(f"Homepage Title: {hp['title']}")
    if hp.get("meta_desc"):
        context_parts.append(f"Meta Description: {hp['meta_desc']}")
    if hp.get("h1s"):
        context_parts.append(f"H1 Headlines: {', '.join(hp['h1s'][:3])}")

    tech = crawl_raw.get("tech_signals", [])
    if tech:
        context_parts.append(f"Tech Stack: {', '.join(tech[:8])}")

    ctas = crawl_raw.get("cta_patterns", [])
    if ctas:
        context_parts.append(f"CTAs Found: {', '.join(ctas[:5])}")

    socials = crawl_raw.get("social_links", [])
    if socials:
        context_parts.append(f"Social Profiles: {len(socials)} found")

    seo = crawl_raw.get("seo_basics", {})
    seo_notes = []
    if not seo.get("has_meta"):
        seo_notes.append("Missing meta tags")
    if not seo.get("has_viewport"):
        seo_notes.append("No mobile viewport")
    if not seo.get("has_sitemap"):
        seo_notes.append("No sitemap detected")
    if seo_notes:
        context_parts.append(f"SEO Issues: {', '.join(seo_notes)}")

    pages = crawl_raw.get("pages_crawled", [])
    if pages:
        context_parts.append(f"Pages Crawled: {len(pages)} ({', '.join(p.get('type', '') for p in pages)})")
        for p in pages[:3]:
            content_preview = p.get("key_content", "")[:300]
            if content_preview:
                context_parts.append(f"  [{p.get('type', 'page')}]: {content_preview}")

    crawl_context = "\n".join(context_parts)

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise business analyst. Given website crawl data, "
                        "produce exactly 5 bullet points (5-10 words each) summarizing "
                        "the business: what they do, who they target, their tech sophistication, "
                        "key strengths, and one notable gap or opportunity.\n\n"
                        "Return ONLY a JSON object: {\"points\": [\"...\", \"...\", \"...\", \"...\", \"...\"]}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Website: {website_url}\n\nCrawl Data:\n{crawl_context}",
                },
            ],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        points = parsed.get("points", [])[:5]

        return {
            "points": points,
            "crawl_status": "complete",
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.error("GPT crawl summary generation failed", error=str(e))
        return _generate_fallback_summary(crawl_raw)


def _generate_fallback_summary(crawl_raw: dict) -> dict:
    """Generate a basic summary without GPT."""
    points = []
    hp = crawl_raw.get("homepage", {})
    if hp.get("title"):
        points.append(f"Business: {hp['title'][:50]}")
    if hp.get("meta_desc"):
        points.append(hp["meta_desc"][:60])

    tech = crawl_raw.get("tech_signals", [])
    if tech:
        points.append(f"Uses: {', '.join(tech[:3])}")

    pages = crawl_raw.get("pages_crawled", [])
    if pages:
        points.append(f"{len(pages)} key pages identified")

    seo = crawl_raw.get("seo_basics", {})
    issues = []
    if not seo.get("has_meta"):
        issues.append("meta tags")
    if not seo.get("has_viewport"):
        issues.append("mobile viewport")
    if issues:
        points.append(f"Missing: {', '.join(issues)}")

    if not points:
        points = ["Website data collected for analysis"]

    return {
        "points": points[:5],
        "crawl_status": "complete",
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }


async def run_background_crawl(session_id: str, website_url: str):
    """
    Run the full crawl pipeline in the background.
    Called as an asyncio task — does not block the API response.

    Steps:
    1. Set crawl_status to "in_progress"
    2. Crawl the website (or social profile)
    3. Generate summary
    4. Store results in session
    5. Set crawl_status to "complete" (or "failed")
    """
    try:
        # Mark crawl as in progress
        session_store.set_crawl_status(session_id, "in_progress")
        logger.info("Background crawl started", session_id=session_id, url=website_url)

        # Determine URL type and run appropriate crawl
        url_type = detect_url_type(website_url)
        if url_type == "social_profile":
            crawl_raw = await crawl_social_profile(website_url)
        else:
            crawl_raw = await crawl_website(website_url)

        # Generate compressed summary
        crawl_summary = await generate_crawl_summary(crawl_raw, website_url)

        # Store in session
        session_store.set_crawl_data(session_id, crawl_raw, crawl_summary)

        logger.info(
            "Background crawl complete",
            session_id=session_id,
            url=website_url,
            url_type=url_type,
            pages=len(crawl_raw.get("pages_crawled", [])),
            tech_signals=len(crawl_raw.get("tech_signals", [])),
        )

    except Exception as e:
        logger.error(
            "Background crawl failed",
            session_id=session_id,
            url=website_url,
            error=str(e),
        )
        session_store.set_crawl_status(session_id, "failed")


async def crawl_social_profile(website_url: str) -> dict:
    """
    Lightweight crawl for social media profile URLs.
    Extracts bio, profile info from the page.
    """
    result = {
        "homepage": {"title": "", "meta_desc": "", "h1s": [], "nav_links": []},
        "pages_crawled": [],
        "tech_signals": [],
        "cta_patterns": [],
        "social_links": [website_url],
        "schema_markup": [],
        "seo_basics": {"has_meta": False, "has_viewport": True, "has_sitemap": False},
    }

    async with httpx.AsyncClient(
        timeout=CRAWL_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": CRAWL_USER_AGENT},
    ) as client:
        html = await _fetch_page(client, website_url)
        if html:
            meta = _extract_meta(html)
            result["homepage"] = {
                "title": meta["title"],
                "meta_desc": meta["meta_desc"],
                "h1s": meta["h1s"],
                "nav_links": [],
            }
            result["schema_markup"] = _extract_schema_markup(html)

    return result
