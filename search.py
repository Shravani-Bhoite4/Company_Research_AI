"""
search.py
=========

Company Research Assistant - Search Integration Module.

This module provides all company search functionality by integrating with
the Serper.dev Google Search API (https://serper.dev). It is responsible for:

    * Validating whether a given string is already a URL.
    * Executing raw Google searches via Serper.dev.
    * Resolving a company's official website.
    * Aggregating structured company information (profile, phone, address,
      products, services, sources) from multiple targeted searches.
    * Extracting useful structured data out of raw Serper.dev search results.
    * Handling API failures (timeouts, invalid keys, rate limits, network
      errors) gracefully, with friendly, user-safe error messages.

This module contains NO user interface code. It is intended to be imported
and used by higher-level application/service/UI layers.

Environment Variables
----------------------
SERPER_API_KEY : str
    Required. Your Serper.dev API key. Loaded from a `.env` file via
    python-dotenv, or from the environment directly.

Example
-------
    from search import search_company_information

    result = search_company_information("OpenAI")
    print(result["website"])
    print(result["description"])
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
import os

# --------------------------------------------------------------------------
# Environment & Constants
# --------------------------------------------------------------------------

load_dotenv()

SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")

SERPER_SEARCH_URL: str = "https://google.serper.dev/search"
DEFAULT_TIMEOUT_SECONDS: int = 15
DEFAULT_RESULT_COUNT: int = 10

# HTTP status codes worth special-casing when talking to Serper.dev.
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429

# Regex used by is_valid_url() as a fast pre-check before urlparse validation.
_URL_REGEX = re.compile(
    r"^(https?://)?"                      # optional scheme
    r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,63}"    # domain(s) + TLD
    r"(:[0-9]{1,5})?"                     # optional port
    r"(/[^\s]*)?$"                        # optional path/query/fragment
)

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

class SerperAPIError(Exception):
    """Base exception for all Serper.dev API related failures."""


class SerperAuthError(SerperAPIError):
    """Raised when the API key is missing, invalid, or unauthorized."""


class SerperRateLimitError(SerperAPIError):
    """Raised when the Serper.dev API rate limit has been exceeded."""


class SerperTimeoutError(SerperAPIError):
    """Raised when a request to Serper.dev times out."""


class SerperConnectionError(SerperAPIError):
    """Raised when a network-level error prevents reaching Serper.dev."""


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def is_valid_url(text: str) -> bool:
    """
    Determine whether the given text is already a well-formed website URL.

    Args:
        text: The input string to evaluate (e.g. a company name or a URL).

    Returns:
        True if `text` looks like a valid URL, False otherwise.

    Examples:
        >>> is_valid_url("https://openai.com")
        True
        >>> is_valid_url("openai.com")
        True
        >>> is_valid_url("OpenAI Inc.")
        False
    """
    if not text or not isinstance(text, str):
        return False

    candidate = text.strip()

    # Quick structural check first (cheap, avoids unnecessary parsing).
    if not _URL_REGEX.match(candidate):
        return False

    # Normalize with a scheme so urlparse can validate netloc reliably.
    normalized = candidate if candidate.startswith(("http://", "https://")) else f"https://{candidate}"

    try:
        parsed = urlparse(normalized)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except ValueError:
        return False


def _get_api_key() -> str:
    """
    Retrieve the Serper.dev API key, raising a clear error if missing.

    Returns:
        The API key string.

    Raises:
        SerperAuthError: If SERPER_API_KEY is not set in the environment.
    """
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY is not set in environment/.env file.")
        raise SerperAuthError(
            "Missing API key. Please set SERPER_API_KEY in your .env file."
        )
    return SERPER_API_KEY


def _build_headers() -> Dict[str, str]:
    """
    Build the required HTTP headers for a Serper.dev API request.

    Returns:
        A dictionary of headers including the X-API-KEY authorization header.
    """
    return {
        "X-API-KEY": _get_api_key(),
        "Content-Type": "application/json",
    }


def _friendly_error_message(error: Exception) -> str:
    """
    Convert an internal exception into a friendly, user-safe error message.

    Args:
        error: The exception raised during a search operation.

    Returns:
        A human-readable error message suitable for display to end users.
    """
    if isinstance(error, SerperAuthError):
        return (
            "Authentication failed: your Serper.dev API key is missing or "
            "invalid. Please check your .env configuration."
        )
    if isinstance(error, SerperRateLimitError):
        return (
            "Search rate limit exceeded. Please wait a moment and try again."
        )
    if isinstance(error, SerperTimeoutError):
        return (
            "The search request timed out. Please check your connection "
            "and try again."
        )
    if isinstance(error, SerperConnectionError):
        return (
            "Unable to reach the search service. Please check your "
            "internet connection and try again."
        )
    if isinstance(error, SerperAPIError):
        return f"Search service error: {error}"
    return "An unexpected error occurred while searching. Please try again."


# --------------------------------------------------------------------------
# Core Search Function
# --------------------------------------------------------------------------

def search_google(
    query: str,
    num_results: int = DEFAULT_RESULT_COUNT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Execute a Google search via the Serper.dev API.

    Args:
        query: The search query string.
        num_results: Desired number of results (Serper.dev may cap this).
        timeout: Request timeout, in seconds.

    Returns:
        The parsed JSON response from Serper.dev as a dictionary. On
        failure, returns a dictionary with an "error" key containing a
        friendly error message instead of raising, so calling code can
        remain simple. (Lower-level exceptions are still raised internally
        and caught here.)

    Raises:
        ValueError: If `query` is empty or not a string.
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    payload = {"q": query.strip(), "num": num_results}

    try:
        headers = _build_headers()
    except SerperAuthError as auth_err:
        logger.error("Auth error before sending request: %s", auth_err)
        return {"error": _friendly_error_message(auth_err)}

    logger.info("Searching Serper.dev for query: %r", query)

    try:
        response = requests.post(
            SERPER_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        logger.error("Request to Serper.dev timed out: %s", exc)
        err = SerperTimeoutError("Request timed out.")
        return {"error": _friendly_error_message(err)}
    except requests.exceptions.ConnectionError as exc:
        logger.error("Network error while contacting Serper.dev: %s", exc)
        err = SerperConnectionError("Network connection error.")
        return {"error": _friendly_error_message(err)}
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected requests error: %s", exc)
        err = SerperAPIError(str(exc))
        return {"error": _friendly_error_message(err)}

    # Handle HTTP-level failures.
    if response.status_code in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
        logger.error(
            "Serper.dev authorization failure (status %s): %s",
            response.status_code,
            response.text,
        )
        err = SerperAuthError("Invalid or unauthorized API key.")
        return {"error": _friendly_error_message(err)}

    if response.status_code == HTTP_TOO_MANY_REQUESTS:
        logger.error("Serper.dev rate limit hit: %s", response.text)
        err = SerperRateLimitError("Rate limit exceeded.")
        return {"error": _friendly_error_message(err)}

    if not response.ok:
        logger.error(
            "Serper.dev returned an error status %s: %s",
            response.status_code,
            response.text,
        )
        err = SerperAPIError(
            f"Unexpected status code {response.status_code} from search API."
        )
        return {"error": _friendly_error_message(err)}

    try:
        data: Dict[str, Any] = response.json()
    except ValueError as exc:
        logger.error("Failed to parse Serper.dev JSON response: %s", exc)
        err = SerperAPIError("Invalid response format from search API.")
        return {"error": _friendly_error_message(err)}

    return data


# --------------------------------------------------------------------------
# Official Website Resolution
# --------------------------------------------------------------------------

def get_official_website(company_name: str) -> Optional[str]:
    """
    Resolve the official website of a company.

    If `company_name` is already a valid URL, it is returned as-is
    (normalized with an https:// scheme if missing). Otherwise, this
    performs a Serper.dev search for "<company_name> official website"
    and returns the top organic result's link.

    Args:
        company_name: The company's name, or possibly already a URL.

    Returns:
        The official website URL as a string, or None if it could not
        be determined.
    """
    if not company_name or not isinstance(company_name, str):
        return None

    candidate = company_name.strip()

    if is_valid_url(candidate):
        normalized = (
            candidate if candidate.startswith(("http://", "https://"))
            else f"https://{candidate}"
        )
        logger.info("Input already a URL, using directly: %s", normalized)
        return normalized

    query = f"{candidate} official website"
    results = search_google(query)

    if "error" in results:
        logger.warning(
            "Could not resolve official website for %s: %s",
            candidate,
            results["error"],
        )
        return None

    organic_results: List[Dict[str, Any]] = results.get("organic", [])
    if not organic_results:
        logger.info("No organic results found for query: %r", query)
        return None

    top_link = organic_results[0].get("link")
    if top_link and is_valid_url(top_link):
        logger.info("Resolved official website for %s: %s", candidate, top_link)
        return top_link

    logger.info("Top result for %r did not contain a valid link.", query)
    return None


# --------------------------------------------------------------------------
# Extraction Helpers
# --------------------------------------------------------------------------

def extract_company_details(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract useful structured information from a raw Serper.dev results
    dictionary.

    This pulls out:
        * A textual description (from the knowledge graph if present,
          otherwise the top organic snippet).
        * A phone number, if discoverable in the knowledge graph or
          snippets.
        * An address, if discoverable in the knowledge graph or snippets.
        * A list of candidate source URLs.

    Args:
        results: The parsed JSON dictionary returned by search_google().

    Returns:
        A dictionary with keys: "description", "phone", "address",
        "sources". Any field that could not be determined will be an
        empty string (or empty list for "sources").
    """
    extracted: Dict[str, Any] = {
        "description": "",
        "phone": "",
        "address": "",
        "sources": [],
    }

    if not results or "error" in results:
        return extracted

    knowledge_graph: Dict[str, Any] = results.get("knowledgeGraph", {}) or {}
    organic_results: List[Dict[str, Any]] = results.get("organic", []) or []

    # --- Description ---
    if knowledge_graph.get("description"):
        extracted["description"] = knowledge_graph["description"]
    elif organic_results:
        extracted["description"] = organic_results[0].get("snippet", "")

    # --- Phone ---
    kg_attributes: Dict[str, Any] = knowledge_graph.get("attributes", {}) or {}
    phone_candidate = (
        knowledge_graph.get("phone")
        or kg_attributes.get("Phone")
        or kg_attributes.get("Customer service")
        or ""
    )
    if not phone_candidate:
        phone_candidate = _extract_phone_from_snippets(organic_results)
    extracted["phone"] = phone_candidate

    # --- Address ---
    address_candidate = (
        knowledge_graph.get("address")
        or kg_attributes.get("Headquarters")
        or kg_attributes.get("Address")
        or ""
    )
    if not address_candidate:
        address_candidate = _extract_address_from_snippets(organic_results)
    extracted["address"] = address_candidate

    # --- Sources ---
    sources = [
        item.get("link") for item in organic_results if item.get("link")
    ]
    extracted["sources"] = sources

    return extracted


def _extract_phone_from_snippets(organic_results: List[Dict[str, Any]]) -> str:
    """
    Attempt to find a phone number pattern within organic result snippets.

    Args:
        organic_results: List of organic result dictionaries from Serper.dev.

    Returns:
        The first phone number match found, or an empty string.
    """
    phone_pattern = re.compile(
        r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}"
    )
    for item in organic_results:
        snippet = item.get("snippet", "")
        match = phone_pattern.search(snippet)
        if match:
            return match.group(0).strip()
    return ""


def _extract_address_from_snippets(organic_results: List[Dict[str, Any]]) -> str:
    """
    Attempt to find an address-like substring within organic result snippets.

    This is a best-effort heuristic looking for common address keywords.

    Args:
        organic_results: List of organic result dictionaries from Serper.dev.

    Returns:
        The first snippet that appears to contain an address, or an
        empty string.
    """
    address_keywords = (
        "Street", "St.", "Avenue", "Ave.", "Road", "Rd.",
        "Boulevard", "Blvd", "Suite", "Floor", "Building",
    )
    for item in organic_results:
        snippet = item.get("snippet", "")
        if any(keyword in snippet for keyword in address_keywords):
            return snippet
    return ""


def _extract_list_items(
    organic_results: List[Dict[str, Any]],
    keywords: List[str],
) -> List[str]:
    """
    Extract a de-duplicated list of relevant snippet fragments matching
    given keywords (used for products/services extraction).

    Args:
        organic_results: List of organic result dictionaries from Serper.dev.
        keywords: Keywords used to identify relevant snippets.

    Returns:
        A list of unique snippet strings that mention any of the keywords.
    """
    matches: List[str] = []
    seen = set()

    for item in organic_results:
        snippet = item.get("snippet", "")
        if not snippet:
            continue
        lowered = snippet.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            if snippet not in seen:
                matches.append(snippet)
                seen.add(snippet)

    return matches


# --------------------------------------------------------------------------
# High-Level Aggregation Function
# --------------------------------------------------------------------------

def search_company_information(company_name: str) -> Dict[str, Any]:
    """
    Perform a full company research sweep and return a structured
    dictionary of findings.

    This orchestrates multiple targeted Serper.dev searches:
        * "<company_name> official website"
        * "<company_name> company profile"
        * "<company_name> phone number"
        * "<company_name> address"
        * "<company_name> products"
        * "<company_name> services"

    Args:
        company_name: The name of the company to research (or a URL).

    Returns:
        A dictionary with the following shape:

            {
                "company_name": str,
                "website": str,
                "phone": str,
                "address": str,
                "description": str,
                "products": List[str],
                "services": List[str],
                "sources": List[str],
            }

        If a fatal error occurs (e.g. missing API key), the dictionary
        will also include an "error" key with a friendly message, and
        the other fields will be left at their default empty values.
    """
    result: Dict[str, Any] = {
        "company_name": company_name.strip() if isinstance(company_name, str) else "",
        "website": "",
        "phone": "",
        "address": "",
        "description": "",
        "products": [],
        "services": [],
        "sources": [],
    }

    if not company_name or not isinstance(company_name, str) or not company_name.strip():
        result["error"] = "Please provide a valid company name."
        return result

    name = company_name.strip()
    logger.info("Starting company information search for: %s", name)

    # 1. Resolve official website first.
    website = get_official_website(name)
    if website:
        result["website"] = website

    # 2. General company profile search.
    profile_results = search_google(f"{name} company profile")
    if "error" in profile_results:
        result["error"] = profile_results["error"]
        return result

    details = extract_company_details(profile_results)
    result["description"] = details["description"]
    result["phone"] = details["phone"]
    result["address"] = details["address"]
    result["sources"].extend(details["sources"])

    # 3. Dedicated phone search, if not already found.
    if not result["phone"]:
        phone_results = search_google(f"{name} phone number")
        if "error" not in phone_results:
            phone_details = extract_company_details(phone_results)
            if phone_details["phone"]:
                result["phone"] = phone_details["phone"]
            result["sources"].extend(phone_details["sources"])

    # 4. Dedicated address search, if not already found.
    if not result["address"]:
        address_results = search_google(f"{name} address")
        if "error" not in address_results:
            address_details = extract_company_details(address_results)
            if address_details["address"]:
                result["address"] = address_details["address"]
            result["sources"].extend(address_details["sources"])

    # 5. Products search.
    products_results = search_google(f"{name} products")
    if "error" not in products_results:
        organic = products_results.get("organic", []) or []
        result["products"] = _extract_list_items(
            organic, keywords=["product", "products"]
        )
        result["sources"].extend(
            item.get("link") for item in organic if item.get("link")
        )

    # 6. Services search.
    services_results = search_google(f"{name} services")
    if "error" not in services_results:
        organic = services_results.get("organic", []) or []
        result["services"] = _extract_list_items(
            organic, keywords=["service", "services"]
        )
        result["sources"].extend(
            item.get("link") for item in organic if item.get("link")
        )

    # De-duplicate sources while preserving order.
    result["sources"] = list(dict.fromkeys(filter(None, result["sources"])))

    logger.info("Completed company information search for: %s", name)
    return result


# --------------------------------------------------------------------------
# Module Self-Test (only runs when executed directly, not on import)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    test_company = "OpenAI"
    info = search_company_information(test_company)
    for key, value in info.items():
        print(f"{key}: {value}")