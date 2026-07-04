"""
crawler.py
==========

Company Research Assistant - Website Crawling Module.

This module provides an intelligent, reusable web crawler that, given a
company's base website URL, discovers the most business-relevant pages
(about, products, services, contact, etc.), extracts clean readable
content from them, and aggregates that content into structures suitable
for downstream AI analysis (e.g. LLM summarization).

Responsibilities:
    * Normalizing and validating URLs found on a site.
    * Discovering internally-linked, business-relevant pages.
    * Downloading and cleaning page HTML into plain text.
    * Crawling a bounded set of pages per site (max 10).
    * Combining all page content into a single bounded-length text blob.
    * Extracting contact information (phone, email, address) via regex.
    * Extracting likely product/service names from headings and content.

This module contains NO user interface code. It is intended to be
imported and used by higher-level application layers (e.g. app.py).

Example
-------
    from crawler import crawl_website, combine_content, extract_contact_information

    pages = crawl_website("https://tesla.com")
    full_text = combine_content(pages)
    contact_info = extract_contact_information(full_text)
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: int = 10
MAX_PAGES_TO_CRAWL: int = 10
MAX_COMBINED_CONTENT_LENGTH: int = 25_000
MAX_DISCOVERY_LINKS_SCANNED: int = 200

REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Keywords used to prioritize which internal pages are most valuable.
PRIORITY_KEYWORDS: List[str] = [
    "about",
    "company",
    "products",
    "services",
    "solutions",
    "pricing",
    "contact",
    "team",
]

# Path/keyword fragments that indicate a page should be excluded entirely.
EXCLUDED_PATH_KEYWORDS: List[str] = [
    "login",
    "signin",
    "sign-in",
    "signup",
    "sign-up",
    "register",
    "cart",
    "checkout",
    "privacy",
    "terms",
    "cookie",
    "logout",
]

# File extensions that should never be treated as crawlable HTML pages.
EXCLUDED_EXTENSIONS: List[str] = [
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".bmp",
    ".ico",
    ".zip",
    ".rar",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".css",
    ".js",
]

# HTML elements/selectors to strip before extracting clean text.
TAGS_TO_REMOVE: List[str] = ["script", "style", "svg", "header", "footer", "nav"]

# Common class/id substrings used by cookie consent popups and similar noise.
NOISE_SELECTORS_KEYWORDS: List[str] = [
    "cookie",
    "consent",
    "gdpr",
    "popup",
    "modal",
    "banner",
    "newsletter",
]

# Regex patterns for contact info extraction.
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(?\d{2,4}\)?[\s.-]?)"
    r"\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
)
ADDRESS_KEYWORDS: List[str] = [
    "street", "st.", "avenue", "ave.", "road", "rd.",
    "boulevard", "blvd", "suite", "floor", "building",
    "drive", "dr.", "lane", "ln.", "way", "plaza",
]

HEADING_TAGS: List[str] = ["h1", "h2", "h3", "h4"]

# --------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------

class CrawlerError(Exception):
    """Base exception for all crawler-related failures."""


class PageFetchError(CrawlerError):
    """Raised when a page could not be fetched (network, SSL, timeout, HTTP)."""


# --------------------------------------------------------------------------
# URL Helpers
# --------------------------------------------------------------------------

def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalize a URL: resolve relative URLs against a base, collapse
    duplicate slashes in the path, and strip trailing fragments.

    Args:
        url: The URL (absolute or relative) to normalize.
        base_url: The base URL to resolve relative URLs against. If not
            provided, `url` is assumed to already be absolute.

    Returns:
        A normalized, absolute URL string.

    Examples:
        >>> normalize_url("/about", "https://tesla.com")
        'https://tesla.com/about'
        >>> normalize_url("https://tesla.com//about//us")
        'https://tesla.com/about/us'
    """
    if not url:
        return ""

    absolute = urljoin(base_url, url) if base_url else url
    parsed = urlparse(absolute)

    # Collapse duplicate slashes in the path (but keep the leading '//' of
    # scheme intact, which urlparse already separates out).
    collapsed_path = re.sub(r"/{2,}", "/", parsed.path)

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            collapsed_path,
            parsed.params,
            parsed.query,
            "",  # Drop fragment identifiers.
        )
    )

    # Remove trailing slash for consistency, except for bare domain roots.
    if normalized.endswith("/") and collapsed_path != "/":
        normalized = normalized.rstrip("/")

    return normalized


def is_valid_internal_link(base_url: str, link: str) -> bool:
    """
    Determine whether `link` is a valid, crawlable internal page relative
    to `base_url`.

    A link is considered valid only if it:
        * Resolves to the same domain as base_url.
        * Uses http/https scheme (not javascript:, mailto:, tel:, etc.).
        * Is not a login, signup, cart, privacy, terms, or cookie page.
        * Does not point to a PDF, image, or other non-HTML asset.

    Args:
        base_url: The site's base/root URL.
        link: The candidate link (absolute or relative) to validate.

    Returns:
        True if the link should be crawled, False otherwise.
    """
    if not link:
        return False

    stripped = link.strip()

    # Reject obvious non-navigational / javascript links up front.
    if stripped.lower().startswith(
        ("javascript:", "mailto:", "tel:", "#", "data:")
    ):
        return False

    absolute = normalize_url(stripped, base_url)
    parsed = urlparse(absolute)

    if parsed.scheme not in ("http", "https"):
        return False

    base_domain = urlparse(base_url).netloc.lower().replace("www.", "")
    link_domain = parsed.netloc.lower().replace("www.", "")
    if link_domain != base_domain:
        return False

    path_lower = parsed.path.lower()

    if any(keyword in path_lower for keyword in EXCLUDED_PATH_KEYWORDS):
        return False

    if any(path_lower.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
        return False

    return True


# --------------------------------------------------------------------------
# Networking Helper
# --------------------------------------------------------------------------

def _fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Fetch raw HTML for a given URL, handling redirects, SSL errors,
    timeouts, and connection errors.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The raw HTML text of the page.

    Raises:
        PageFetchError: If the page could not be fetched for any reason.
    """
    try:
        response = requests.get(
    url,
    headers=REQUEST_HEADERS,
    timeout=timeout,
    allow_redirects=True,
    verify=False,
)
    except requests.exceptions.SSLError as exc:
        logger.warning("SSL error fetching %s: %s", url, exc)
        raise PageFetchError(f"SSL error while fetching {url}") from exc
    except requests.exceptions.Timeout as exc:
        logger.warning("Timeout fetching %s: %s", url, exc)
        raise PageFetchError(f"Timed out while fetching {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        logger.warning("Connection error fetching %s: %s", url, exc)
        raise PageFetchError(f"Connection error while fetching {url}") from exc
    except requests.exceptions.RequestException as exc:
        logger.warning("Request error fetching %s: %s", url, exc)
        raise PageFetchError(f"Error fetching {url}: {exc}") from exc

    if not response.ok:
        logger.warning(
            "Non-OK status %s fetching %s", response.status_code, url
        )
        raise PageFetchError(
            f"Received status code {response.status_code} for {url}"
        )

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        logger.warning(
            "Skipping non-HTML content type '%s' at %s", content_type, url
        )
        raise PageFetchError(f"Non-HTML content type at {url}")

    return response.text


# --------------------------------------------------------------------------
# Page Discovery
# --------------------------------------------------------------------------

def discover_pages(base_url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> List[str]:
    """
    Discover important, business-relevant internal pages linked from the
    base URL's homepage, prioritizing pages matching known keywords
    (about, company, products, services, solutions, pricing, contact, team).

    Args:
        base_url: The site's root URL (e.g. "https://tesla.com").
        timeout: Request timeout in seconds for the homepage fetch.

    Returns:
        A list of unique, absolute URLs, with priority-keyword pages
        ordered first, followed by other valid internal links.
    """
    try:
        html = _fetch_html(base_url, timeout=timeout)
    except PageFetchError as exc:
        logger.error("Could not fetch homepage %s: %s", base_url, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)

    priority_links: List[str] = []
    other_links: List[str] = []
    seen: Set[str] = set()

    for anchor in anchors[:MAX_DISCOVERY_LINKS_SCANNED]:
        href = anchor["href"]

        if not is_valid_internal_link(base_url, href):
            continue

        absolute = normalize_url(href, base_url)

        if absolute in seen:
            continue
        seen.add(absolute)

        path_lower = urlparse(absolute).path.lower()
        anchor_text = anchor.get_text(strip=True).lower()

        matched_priority = any(
            keyword in path_lower or keyword in anchor_text
            for keyword in PRIORITY_KEYWORDS
        )

        if matched_priority:
            priority_links.append(absolute)
        else:
            other_links.append(absolute)

    ordered_unique = list(dict.fromkeys(priority_links + other_links))
    logger.info(
        "Discovered %d candidate pages for %s (%d priority matches)",
        len(ordered_unique),
        base_url,
        len(priority_links),
    )
    return ordered_unique


# --------------------------------------------------------------------------
# Content Extraction
# --------------------------------------------------------------------------

def _strip_noise(soup: BeautifulSoup) -> None:
    """
    Remove non-content elements (scripts, styles, svgs, header/footer/nav,
    and likely cookie/consent popups) from a BeautifulSoup document
    in-place.

    Args:
        soup: The BeautifulSoup document to clean.
    """
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements that look like cookie banners / popups / modals
    # based on their class or id attributes.
    for tag in soup.find_all(True):

    # Skip broken tags
     if tag is None or getattr(tag, "attrs", None) is None:
        continue

     classes = tag.get("class") or []
     if not isinstance(classes, list):
        classes = [str(classes)]

     attr_blob = (
        " ".join(classes) + " " + str(tag.get("id") or "")
    ).lower()

     if any(keyword in attr_blob for keyword in NOISE_SELECTORS_KEYWORDS):
        tag.decompose()


def extract_page_content(
    url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> Dict[str, str]:
    """
    Download a page and extract its title, meta description, and clean
    body text, with non-content elements removed.

    Args:
        url: The page URL to fetch and parse.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary with keys: "url", "title", "description", "content".
        If the page could not be fetched, "content" will be empty and
        an "error" key will describe the failure.
    """
    result: Dict[str, str] = {
        "url": url,
        "title": "",
        "description": "",
        "content": "",
    }

    try:
        html = _fetch_html(url, timeout=timeout)
    except PageFetchError as exc:
        logger.warning("Failed to extract content from %s: %s", url, exc)
        result["error"] = str(exc)
        return result

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        result["title"] = title_tag.get_text(strip=True)

    meta_description = soup.find("meta", attrs={"name": "description"})
    if meta_description and meta_description.get("content"):
        result["description"] = meta_description["content"].strip()
    else:
        og_description = soup.find("meta", attrs={"property": "og:description"})
        if og_description and og_description.get("content"):
            result["description"] = og_description["content"].strip()

    _strip_noise(soup)

    body = soup.find("body") or soup
    text = body.get_text(separator=" ", strip=True)
    clean_text = re.sub(r"\s+", " ", text).strip()

    result["content"] = clean_text
    logger.info(
        "Extracted content from %s (%d chars)", url, len(clean_text)
    )
    return result


# --------------------------------------------------------------------------
# Crawling Orchestration
# --------------------------------------------------------------------------
def crawl_website(
    base_url: str,
    max_pages: int = MAX_PAGES_TO_CRAWL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, str]]:
    """
    Crawl a company website: discover important pages and extract clean
    content from each, up to a maximum number of pages.
    """

    normalized_base = normalize_url(base_url)
    logger.info(
        "Starting crawl of %s (max_pages=%d)",
        normalized_base,
        max_pages,
    )

    discovered = discover_pages(normalized_base, timeout=timeout)

    # Always crawl homepage first
    queue: deque[str] = deque([normalized_base] + discovered)

    visited: Set[str] = set()
    pages: List[Dict[str, str]] = []

    while queue and len(pages) < max_pages:

        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        page_data = extract_page_content(url, timeout=timeout)

        print("PAGE DATA =", page_data)

        # If extraction failed
        if page_data is None:
            print("extract_page_content returned None")
            continue

        # Skip pages with errors
        if page_data.get("error"):
            logger.warning("Skipping page due to fetch error: %s", url)
            continue

        # Skip empty pages
        if not page_data.get("content"):
            logger.warning("Skipping page with empty content: %s", url)
            continue

        pages.append(
            {
                "url": page_data["url"],
                "title": page_data["title"],
                "description": page_data["description"],
                "content": page_data["content"],
            }
        )

        print("TOTAL PAGES =", len(pages))

    logger.info(
        "Completed crawl of %s: %d pages collected",
        normalized_base,
        len(pages),
    )

    return pages

# --------------------------------------------------------------------------
# Content Aggregation
# --------------------------------------------------------------------------

def combine_content(
    pages: List[Dict[str, str]],
    max_length: int = MAX_COMBINED_CONTENT_LENGTH,
) -> str:
    """
    Merge all crawled page contents into a single text blob, bounded to
    a maximum length, suitable for feeding into an LLM for analysis.

    Args:
        pages: A list of page dictionaries as returned by crawl_website().
        max_length: The maximum length, in characters, of the combined text.

    Returns:
        A single combined string of all page contents, truncated to
        `max_length` characters if necessary. Each page's content is
        prefixed with a small header indicating its source URL and title.
    """
    combined_parts: List[str] = []
    running_length = 0

    for page in pages:
        header = f"\n\n--- Source: {page.get('url', '')} | {page.get('title', '')} ---\n"
        content = page.get("content", "")
        segment = header + content

        if running_length + len(segment) > max_length:
            remaining = max_length - running_length
            if remaining > 0:
                combined_parts.append(segment[:remaining])
                running_length += remaining
            break

        combined_parts.append(segment)
        running_length += len(segment)

    combined_text = "".join(combined_parts).strip()
    logger.info(
        "Combined %d pages into %d characters (max %d)",
        len(pages),
        len(combined_text),
        max_length,
    )
    return combined_text[:max_length]


# --------------------------------------------------------------------------
# Contact Information Extraction
# --------------------------------------------------------------------------

def extract_contact_information(text: str) -> Dict[str, Any]:
    """
    Extract contact information (emails, phone numbers, and a possible
    address) from a block of text using regex heuristics.

    Args:
        text: The text to scan for contact details.

    Returns:
        A dictionary shaped as:
            {
                "emails": List[str],
                "phones": List[str],
                "address": str,
            }
    """
    if not text:
        return {"emails": [], "phones": [], "address": ""}

    emails = list(dict.fromkeys(EMAIL_PATTERN.findall(text)))
    raw_phones = PHONE_PATTERN.findall(text)

    # PHONE_PATTERN.findall with capture groups returns tuples; re-run
    # with finditer to get full matches instead.
    phone_matches = [
        match.group(0).strip()
        for match in PHONE_PATTERN.finditer(text)
        if len(re.sub(r"\D", "", match.group(0))) >= 7
    ]
    phones = list(dict.fromkeys(phone_matches))

    address = ""
    # Look for a sentence/segment containing address-like keywords.
    segments = re.split(r"(?<=[.!?])\s+|\n", text)
    for segment in segments:
        lowered = segment.lower()
        if any(keyword in lowered for keyword in ADDRESS_KEYWORDS):
            address = segment.strip()
            break

    logger.info(
        "Extracted contact info: %d emails, %d phones, address_found=%s",
        len(emails),
        len(phones),
        bool(address),
    )

    return {
        "emails": emails,
        "phones": phones,
        "address": address,
    }


# --------------------------------------------------------------------------
# Products & Services Extraction
# --------------------------------------------------------------------------

def extract_products_services(text: str) -> Dict[str, List[str]]:
    """
    Extract likely product and service names from text, based on
    heading-like patterns and keyword-adjacent phrases.

    This is a heuristic, best-effort extraction intended to give an AI
    analysis step useful candidate terms rather than a definitive list.

    Args:
        text: The combined page text to analyze.

    Returns:
        A dictionary shaped as:
            {
                "products": List[str],
                "services": List[str],
            }
    """
    if not text:
        return {"products": [], "services": []}

    products: List[str] = []
    services: List[str] = []

    # Split into sentence-like segments for scanning.
    segments = re.split(r"(?<=[.!?])\s+|\n", text)

    product_keywords = ["product", "products", "solution", "solutions"]
    service_keywords = ["service", "services", "offering", "offerings"]

    for segment in segments:
        cleaned = segment.strip()
        if not cleaned or len(cleaned) > 200:
            continue

        lowered = cleaned.lower()

        if any(keyword in lowered for keyword in product_keywords):
            if cleaned not in products:
                products.append(cleaned)
        elif any(keyword in lowered for keyword in service_keywords):
            if cleaned not in services:
                services.append(cleaned)

    # Cap results to keep output manageable for downstream consumers.
    max_items = 25
    result = {
        "products": products[:max_items],
        "services": services[:max_items],
    }

    logger.info(
        "Extracted %d candidate products and %d candidate services",
        len(result["products"]),
        len(result["services"]),
    )
    return result


# --------------------------------------------------------------------------
# Module Self-Test (only runs when executed directly, not on import)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    test_url = "https://www.python.org"
    crawled_pages = crawl_website(test_url)

    for crawled_page in crawled_pages:
        print(crawled_page["url"], "->", crawled_page["title"])

    combined = combine_content(crawled_pages)
    print("\nCombined content length:", len(combined))

    contact = extract_contact_information(combined)
    print("\nContact info:", contact)

    products_services = extract_products_services(combined)
    print("\nProducts/Services:", products_services)