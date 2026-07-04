"""
ai.py
=====

Company Research Assistant - AI Processing Module.

This module is responsible ONLY for AI processing: taking aggregated
company data (from search.py and crawler.py) and producing a structured,
AI-generated business research report via the OpenRouter Chat Completions
API (https://openrouter.ai).

Responsibilities:
    * Selecting which OpenRouter model to use (user-selectable, with a
      sensible free-tier default).
    * Building a professional, structured prompt from raw company data.
    * Calling the OpenRouter API to generate an analytical report.
    * Parsing the model's raw text output into a strict, structured
      dictionary suitable for downstream consumption (e.g. by a UI or
      another service).
    * Handling API failures (invalid key, rate limits, server errors,
      timeouts, connection errors) gracefully with friendly messages.

This module contains NO user interface code. It is intended to be
imported and used by higher-level application layers (e.g. app.py),
typically after search.py and crawler.py have gathered raw company data.

Example
-------
    from ai import generate_ai_report

    company_data = {
        "company_name": "Tesla",
        "website": "https://tesla.com",
        "description": "Electric vehicles and clean energy company.",
        "products": ["Model 3", "Model Y", "Solar Roof"],
        "services": ["Charging network", "Insurance"],
        "website_content": "... crawled site text ...",
    }

    report = generate_ai_report(company_data)
    print(report["summary"])
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Environment & Constants
# --------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_CHAT_COMPLETIONS_URL: str = (
    "https://openrouter.ai/api/v1/chat/completions"
)
DEFAULT_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_TIMEOUT_SECONDS: int = 60
DEFAULT_TEMPERATURE: float = 0.4
DEFAULT_MAX_TOKENS: int = 2000

# Maximum characters of crawled website content to include in the prompt,
# to keep requests within reasonable token limits.
MAX_WEBSITE_CONTENT_CHARS: int = 12_000

HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500

# The canonical keys expected in a structured AI report. Used both to
# build the prompt's expected JSON schema and to normalize parsed output.
REPORT_SCHEMA_KEYS: List[str] = [
    "summary",
    "products",
    "services",
    "business_model",
    "pain_points",
    "target_customers",
    "strengths",
    "weaknesses",
    "competitors",
    "growth_opportunities",
    "technology_stack_guess",
]

# Keys that should always be lists (as opposed to plain strings).
LIST_TYPE_KEYS: List[str] = [
    "products",
    "services",
    "pain_points",
    "target_customers",
    "strengths",
    "weaknesses",
    "competitors",
    "growth_opportunities",
    "technology_stack_guess",
]

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

class OpenRouterError(Exception):
    """Base exception for all OpenRouter API related failures."""


class OpenRouterAuthError(OpenRouterError):
    """Raised when the API key is missing, invalid, or unauthorized."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when the OpenRouter API rate limit has been exceeded."""


class OpenRouterServerError(OpenRouterError):
    """Raised when OpenRouter responds with a server-side (5xx) error."""


class OpenRouterTimeoutError(OpenRouterError):
    """Raised when a request to OpenRouter times out."""


class OpenRouterConnectionError(OpenRouterError):
    """Raised when a network-level error prevents reaching OpenRouter."""


# --------------------------------------------------------------------------
# Model Selection
# --------------------------------------------------------------------------

def get_available_model(default_model: Optional[str] = None) -> str:
    """
    Resolve which OpenRouter model should be used for a request.

    Allows callers (and ultimately end users, via the calling layer) to
    select any OpenRouter-supported model. Falls back to a sensible
    free-tier default when none is specified.

    Args:
        default_model: An optional model identifier requested by the
            caller/user (e.g. "openai/gpt-4o-mini"). If falsy, the
            module-level DEFAULT_MODEL is used instead.

    Returns:
        The resolved OpenRouter model identifier string to use.
    """
    if default_model and isinstance(default_model, str) and default_model.strip():
        model = default_model.strip()
        logger.info("Using user-specified OpenRouter model: %s", model)
        return model

    logger.info("No model specified, falling back to default: %s", DEFAULT_MODEL)
    return DEFAULT_MODEL


# --------------------------------------------------------------------------
# Internal Helpers
# --------------------------------------------------------------------------

def _get_api_key() -> str:
    """
    Retrieve the OpenRouter API key, raising a clear error if missing.

    Returns:
        The API key string.

    Raises:
        OpenRouterAuthError: If OPENROUTER_API_KEY is not set.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment/.env file.")
        raise OpenRouterAuthError(
            "Missing API key. Please set OPENROUTER_API_KEY in your .env file."
        )
    return OPENROUTER_API_KEY


def _build_headers() -> Dict[str, str]:
    """
    Build the required HTTP headers for an OpenRouter API request.

    Returns:
        A dictionary of headers including the Bearer authorization token.
    """
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _friendly_error_message(error: Exception) -> str:
    """
    Convert an internal exception into a friendly, user-safe error message.

    Args:
        error: The exception raised during an AI report generation call.

    Returns:
        A human-readable error message suitable for display to end users.
    """
    if isinstance(error, OpenRouterAuthError):
        return (
            "Authentication failed: your OpenRouter API key is missing or "
            "invalid. Please check your .env configuration."
        )
    if isinstance(error, OpenRouterRateLimitError):
        return (
            "AI service rate limit exceeded. Please wait a moment and try "
            "again."
        )
    if isinstance(error, OpenRouterServerError):
        return (
            "The AI service is currently experiencing issues. Please try "
            "again shortly."
        )
    if isinstance(error, OpenRouterTimeoutError):
        return (
            "The AI request timed out. Please try again, possibly with a "
            "shorter input or a different model."
        )
    if isinstance(error, OpenRouterConnectionError):
        return (
            "Unable to reach the AI service. Please check your internet "
            "connection and try again."
        )
    if isinstance(error, OpenRouterError):
        return f"AI service error: {error}"
    return "An unexpected error occurred while generating the report."


def _truncate(text: str, max_length: int) -> str:
    """
    Truncate text to a maximum length, appending a marker if truncated.

    Args:
        text: The text to truncate.
        max_length: The maximum allowed length.

    Returns:
        The (possibly truncated) text.
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + " ... [content truncated]"


def _empty_report() -> Dict[str, Any]:
    """
    Build an empty, schema-compliant report dictionary with default
    empty values for every expected key.

    Returns:
        A dictionary with all REPORT_SCHEMA_KEYS present, lists defaulting
        to [] and strings defaulting to "".
    """
    report: Dict[str, Any] = {}
    for key in REPORT_SCHEMA_KEYS:
        report[key] = [] if key in LIST_TYPE_KEYS else ""
    return report


# --------------------------------------------------------------------------
# Prompt Construction
# --------------------------------------------------------------------------

def build_prompt(company_data: Dict[str, Any]) -> str:
    """
    Build a professional analytical prompt for the AI model, requesting
    a structured business research report.

    Args:
        company_data: A dictionary shaped as:
            {
                "company_name": str,
                "website": str,
                "description": str,
                "products": List[str],
                "services": List[str],
                "website_content": str,
            }
            All keys are optional; missing values are treated as empty.

    Returns:
        A fully-formed prompt string instructing the model to return
        a strict JSON object matching the required report schema.
    """
    company_name = company_data.get("company_name", "") or "Unknown Company"
    website = company_data.get("website", "") or "Not available"
    description = company_data.get("description", "") or "Not available"
    products = company_data.get("products", []) or []
    services = company_data.get("services", []) or []
    website_content = _truncate(
        company_data.get("website_content", "") or "",
        MAX_WEBSITE_CONTENT_CHARS,
    )

    products_str = ", ".join(products) if products else "Not available"
    services_str = ", ".join(services) if services else "Not available"

    schema_example = {key: ([] if key in LIST_TYPE_KEYS else "") for key in REPORT_SCHEMA_KEYS}

    prompt = f"""You are a senior business and technology analyst producing a due-diligence
style research report on a company, based only on the information provided below.

COMPANY DATA
------------
Company Name: {company_name}
Website: {website}
Known Description: {description}
Known Products: {products_str}
Known Services: {services_str}

Website Content (raw excerpt, may contain navigation remnants or noise):
\"\"\"
{website_content if website_content else "Not available"}
\"\"\"

TASK
----
Using only the information above (and reasonable, clearly-labeled inference
where data is incomplete), produce a professional analysis covering:

1. Company Summary - a concise overview of what the company does.
2. Products - a list of the company's key products.
3. Services - a list of the company's key services.
4. Business Model - how the company generates revenue and operates.
5. Pain Points - likely operational, market, or customer pain points the
   company addresses or experiences.
6. Target Customers - the company's likely target customer segments.
7. Strengths - key competitive strengths.
8. Weaknesses - potential weaknesses or risks.
9. Competitors - likely competitors in the same market/industry.
10. Growth Opportunities - plausible avenues for growth or expansion.
11. Technology Stack Guess - a best-effort guess at technologies the
    company likely uses, based on available signals.

OUTPUT FORMAT
-------------
Respond with ONLY a single valid JSON object and nothing else - no
markdown code fences, no preamble, no explanation. The JSON object must
have exactly these keys:

{json.dumps(schema_example, indent=2)}

Rules:
- "summary" and "business_model" must be strings.
- All other fields must be arrays of short strings (bullet-style points).
- If information is not available or cannot reasonably be inferred, use
  an empty string or empty array for that field rather than inventing
  specifics.
- Do not include any text outside of the JSON object.
"""
    return prompt.strip()


# --------------------------------------------------------------------------
# OpenRouter API Call
# --------------------------------------------------------------------------

def _call_openrouter(
    prompt: str,
    model: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Send a chat completion request to OpenRouter and return the parsed
    JSON response.

    Args:
        prompt: The full user prompt to send to the model.
        model: The OpenRouter model identifier to use.
        timeout: Request timeout, in seconds.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        The parsed JSON response body from OpenRouter as a dictionary.
        On failure, returns a dictionary with an "error" key containing
        a friendly error message instead of raising.
    """
    try:
        headers = _build_headers()
    except OpenRouterAuthError as auth_err:
        logger.error("Auth error before sending request: %s", auth_err)
        return {"error": _friendly_error_message(auth_err)}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise business analyst that always "
                    "responds with strictly valid JSON, matching the "
                    "requested schema exactly, and no additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("Calling OpenRouter with model=%s", model)

    try:
        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        print("STATUS =", response.status_code)
        print("URL =", response.url)
        print("API =", OPENROUTER_CHAT_COMPLETIONS_URL)
        print("BODY =", response.text)
          
    except requests.exceptions.Timeout as exc:
        logger.error("Request to OpenRouter timed out: %s", exc)
        err = OpenRouterTimeoutError("Request timed out.")
        return {"error": _friendly_error_message(err)}
    except requests.exceptions.ConnectionError as exc:
        logger.error("Network error while contacting OpenRouter: %s", exc)
        err = OpenRouterConnectionError("Network connection error.")
        return {"error": _friendly_error_message(err)}
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected requests error: %s", exc)
        err = OpenRouterError(str(exc))
        return {"error": _friendly_error_message(err)}

    if response.status_code in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
        logger.error(
            "OpenRouter authorization failure (status %s): %s",
            response.status_code,
            response.text,
        )
        err = OpenRouterAuthError("Invalid or unauthorized API key.")
        return {"error": _friendly_error_message(err)}

    if response.status_code == HTTP_TOO_MANY_REQUESTS:
        logger.error("OpenRouter rate limit hit: %s", response.text)
        err = OpenRouterRateLimitError("Rate limit exceeded.")
        return {"error": _friendly_error_message(err)}

    if response.status_code >= HTTP_SERVER_ERROR_MIN:
        logger.error(
            "OpenRouter server error (status %s): %s",
            response.status_code,
            response.text,
        )
        err = OpenRouterServerError(
            f"Server returned status {response.status_code}."
        )
        return {"error": _friendly_error_message(err)}

    if not response.ok:
        logger.error(
            "OpenRouter returned an unexpected error status %s: %s",
            response.status_code,
            response.text,
        )
        err = OpenRouterError(
            f"Unexpected status code {response.status_code} from AI API."
        )
        return {"error": _friendly_error_message(err)}

    try:
        data: Dict[str, Any] = response.json()
    except ValueError as exc:
        logger.error("Failed to parse OpenRouter JSON response: %s", exc)
        err = OpenRouterError("Invalid response format from AI API.")
        return {"error": _friendly_error_message(err)}

    return data


# --------------------------------------------------------------------------
# Response Parsing
# --------------------------------------------------------------------------

def _extract_message_text(api_response: Dict[str, Any]) -> str:
    """
    Extract the raw assistant message text from an OpenRouter chat
    completion API response.

    Args:
        api_response: The parsed JSON response from OpenRouter.

    Returns:
        The raw text content of the first choice's message, or an
        empty string if it could not be found.
    """
    try:
        choices = api_response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""
    except (AttributeError, IndexError, TypeError) as exc:
        logger.warning("Could not extract message text from response: %s", exc)
        return ""


def _extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first plausible JSON object substring from a block of
    text, tolerating markdown code fences or stray leading/trailing text.

    Args:
        text: The raw text potentially containing a JSON object.

    Returns:
        The extracted JSON substring, or None if none could be found.
    """
    if not text:
        return None

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```).
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Otherwise, find the first '{' and the matching last '}' in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    return None


def parse_ai_response(raw_response: Any) -> Dict[str, Any]:
    """
    Convert raw AI model output (either a full OpenRouter API response
    dictionary or a raw text string) into a structured, schema-compliant
    report dictionary.

    Args:
        raw_response: Either the full OpenRouter API response dictionary
            (containing "choices") or a raw string of model output text.

    Returns:
        A dictionary matching REPORT_SCHEMA_KEYS. If parsing fails, an
        empty schema-compliant report is returned along with an "error"
        key describing the issue.
    """
    if isinstance(raw_response, dict) and "choices" in raw_response:
        message_text = _extract_message_text(raw_response)
    elif isinstance(raw_response, str):
        message_text = raw_response
    else:
        logger.warning("parse_ai_response received an unsupported type: %s", type(raw_response))
        report = _empty_report()
        report["error"] = "Unsupported AI response format."
        return report

    json_block = _extract_json_block(message_text)

    if not json_block:
        logger.warning("No JSON object could be located in AI response text.")
        report = _empty_report()
        report["error"] = "AI response did not contain valid JSON."
        return report

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode JSON from AI response: %s", exc)
        report = _empty_report()
        report["error"] = "Failed to parse structured data from AI response."
        return report

    if not isinstance(parsed, dict):
        logger.warning("Parsed AI response JSON is not an object: %s", type(parsed))
        report = _empty_report()
        report["error"] = "AI response JSON was not an object."
        return report

    return _normalize_report(parsed)


def _normalize_report(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw parsed JSON dictionary into a strict schema-compliant
    report, coercing types and filling in any missing keys with defaults.

    Args:
        parsed: The raw dictionary parsed from the model's JSON output.

    Returns:
        A dictionary containing exactly REPORT_SCHEMA_KEYS, with correct
        types (str for text fields, List[str] for list fields).
    """
    normalized: Dict[str, Any] = {}

    for key in REPORT_SCHEMA_KEYS:
        value = parsed.get(key)

        if key in LIST_TYPE_KEYS:
            if isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                # Model may have returned a comma/newline separated string
                # instead of a JSON array; split it into a list gracefully.
                parts = re.split(r"[\n,]+", value)
                normalized[key] = [part.strip("-• ").strip() for part in parts if part.strip()]
            else:
                normalized[key] = []
        else:
            if isinstance(value, str):
                normalized[key] = value.strip()
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = str(value).strip()

    return normalized


# --------------------------------------------------------------------------
# High-Level Report Generation
# --------------------------------------------------------------------------

def generate_ai_report(
    company_data: Dict[str, Any],
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Generate a structured AI business research report for a company.

    This orchestrates prompt construction, the OpenRouter API call, and
    response parsing into a single convenient entry point.

    Args:
        company_data: A dictionary shaped as:
            {
                "company_name": str,
                "website": str,
                "description": str,
                "products": List[str],
                "services": List[str],
                "website_content": str,
            }
        model: An optional OpenRouter model identifier. If not provided,
            the default free-tier model is used.
        timeout: Request timeout, in seconds.

    Returns:
        A dictionary matching the report schema:
            {
                "summary": str,
                "products": List[str],
                "services": List[str],
                "business_model": str,
                "pain_points": List[str],
                "target_customers": List[str],
                "strengths": List[str],
                "weaknesses": List[str],
                "competitors": List[str],
                "growth_opportunities": List[str],
                "technology_stack_guess": List[str],
            }
        If a fatal error occurs (missing API key, network failure, rate
        limiting, invalid response, etc.), the dictionary will also
        contain an "error" key with a friendly message, and the rest of
        the fields will default to empty values.
    """
    if not isinstance(company_data, dict) or not company_data.get("company_name"):
        logger.error("generate_ai_report called with invalid company_data.")
        report = _empty_report()
        report["error"] = "Company data with at least a company name is required."
        return report

    resolved_model = get_available_model(model)
    prompt = build_prompt(company_data)

    api_response = _call_openrouter(prompt, model=resolved_model, timeout=timeout)

    if "error" in api_response:
        logger.error(
            "AI report generation failed for %s: %s",
            company_data.get("company_name"),
            api_response["error"],
        )
        report = _empty_report()
        report["error"] = api_response["error"]
        return report

    report = parse_ai_response(api_response)

    if "error" in report:
        logger.warning(
            "AI report parsing issue for %s: %s",
            company_data.get("company_name"),
            report["error"],
        )
    else:
        logger.info(
            "Successfully generated AI report for %s using model %s",
            company_data.get("company_name"),
            resolved_model,
        )

    return report


# --------------------------------------------------------------------------
# Module Self-Test (only runs when executed directly, not on import)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    sample_company_data = {
        "company_name": "Tesla",
        "website": "https://tesla.com",
        "description": "Electric vehicles, energy storage, and solar products.",
        "products": ["Model 3", "Model Y", "Model S", "Solar Roof"],
        "services": ["Supercharger network", "Vehicle insurance"],
        "website_content": (
            "Tesla designs and manufactures electric vehicles, battery "
            "energy storage, and solar products, aiming to accelerate "
            "the world's transition to sustainable energy."
        ),
    }

    ai_report = generate_ai_report(sample_company_data)
    print(json.dumps(ai_report, indent=2))