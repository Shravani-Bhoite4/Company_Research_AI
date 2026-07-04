"""
utils.py
========

Company Research Assistant - Shared Utility Module.

This module contains lightweight, reusable helper functions shared across
the project's other modules:

    * search.py         - Serper.dev search integration
    * crawler.py         - website crawling and content extraction
    * ai.py               - OpenRouter AI report generation
    * pdf_generator.py   - PDF report generation
    * app.py              - application orchestration layer

Responsibilities:
    * Environment variable loading and validation.
    * Centralized logging configuration (console + rotating file).
    * URL validation.
    * Text cleaning and truncation.
    * List de-duplication.
    * A `safe_request` decorator for graceful network error handling.
    * Reports directory management.
    * Timestamp generation.
    * Filename sanitization.
    * JSON read/write helpers.

This module contains NO UI code and should have no dependency on any
of the other project modules (to avoid circular imports).

Example
-------
    from utils import setup_logger, load_environment, clean_text

    logger = setup_logger(__name__)
    env = load_environment()
    text = clean_text("Some   messy\\n\\n\\ttext")
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from datetime import datetime
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

try:
    import validators  # type: ignore
    _VALIDATORS_AVAILABLE = True
except ImportError:  # pragma: no cover - validators is an expected dependency,
    # but is_valid_url() falls back gracefully if it's ever unavailable.
    validators = None  # type: ignore
    _VALIDATORS_AVAILABLE = False

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REPORTS_DIRECTORY: str = "reports"
LOGS_DIRECTORY: str = "logs"
LOG_FILENAME: str = "app.log"

LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"

DEFAULT_MAX_FILENAME_LENGTH: int = 80
DEFAULT_TRUNCATE_SUFFIX: str = "..."

REQUIRED_ENV_VARS: List[str] = ["SERPER_API_KEY", "OPENROUTER_API_KEY"]

# Characters that are illegal (or unwise) in filenames across common
# operating systems (Windows, macOS, Linux).
ILLEGAL_FILENAME_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

# Characters to strip out during general text cleaning (control chars,
# zero-width spaces, and other invisible/unwanted unicode noise).
UNWANTED_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\u200B-\u200F\uFEFF]")

_T = TypeVar("_T")

# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------

class EnvironmentConfigurationError(Exception):
    """Raised when required environment variables are missing or invalid."""


# --------------------------------------------------------------------------
# Environment Loading
# --------------------------------------------------------------------------

def load_environment(required_vars: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Load environment variables from a `.env` file (and the surrounding
    environment), validating that required API keys are present.

    Args:
        required_vars: An optional list of environment variable names that
            must be present. Defaults to REQUIRED_ENV_VARS
            (SERPER_API_KEY, OPENROUTER_API_KEY).

    Returns:
        A dictionary mapping each required variable name to its value,
        e.g. {"SERPER_API_KEY": "...", "OPENROUTER_API_KEY": "..."}.

    Raises:
        EnvironmentConfigurationError: If any required variable is missing
            or empty.
    """
    load_dotenv()

    variables_to_check = required_vars if required_vars is not None else REQUIRED_ENV_VARS

    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for var_name in variables_to_check:
        value = os.getenv(var_name)
        if value and value.strip():
            resolved[var_name] = value.strip()
        else:
            missing.append(var_name)

    if missing:
        message = (
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Please set them in your .env file."
        )
        raise EnvironmentConfigurationError(message)

    return resolved


# --------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------

def setup_logger(
    name: str = "company_research_assistant",
    logs_directory: str = LOGS_DIRECTORY,
    log_filename: str = LOG_FILENAME,
    level: int = logging.INFO,
) -> Logger:
    """
    Configure and return a logger that writes to both the console and a
    log file under an automatically-created logs directory.

    Log format: `YYYY-MM-DD HH:MM:SS | LEVEL | name | message`

    Args:
        name: The logger's name (typically `__name__` of the calling module).
        logs_directory: Directory in which the log file will be created.
        log_filename: The name of the log file.
        level: The logging level to configure (default: logging.INFO).

    Returns:
        A configured `logging.Logger` instance.
    """
    os.makedirs(logs_directory, exist_ok=True)
    log_filepath = os.path.join(logs_directory, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid attaching duplicate handlers if setup_logger() is called
    # multiple times for the same logger name (e.g. across module imports).
    if not logger.handlers:
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger


# --------------------------------------------------------------------------
# URL Validation
# --------------------------------------------------------------------------

def is_valid_url(text: str) -> bool:
    """
    Determine whether a given string is a valid, well-formed URL.

    Uses the `validators` library as the primary check, with a fallback
    to `urllib.parse` for edge cases, ensuring both a scheme and network
    location are present.

    Args:
        text: The string to validate.

    Returns:
        True if `text` is a valid URL, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    candidate = text.strip()

    if _VALIDATORS_AVAILABLE:
        try:
            if validators.url(candidate) is True:
                return True
        except Exception:  # noqa: BLE001 - validators may raise on odd input; fall back.
            pass

    try:
        parsed = urlparse(candidate)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Text Cleaning
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean a block of text by removing extra spaces, collapsing multiple
    newlines, converting tabs to spaces, and stripping unwanted control
    or invisible unicode characters.

    Args:
        text: The raw text to clean.

    Returns:
        The cleaned text, or an empty string if `text` is falsy.
    """
    if not text:
        return ""

    cleaned = UNWANTED_CHARS_PATTERN.sub("", text)

    # Convert tabs to single spaces.
    cleaned = cleaned.replace("\t", " ")

    # Collapse 2+ consecutive newlines (with optional whitespace between)
    # down to a single newline.
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)

    # Collapse repeated spaces/horizontal whitespace into a single space.
    cleaned = re.sub(r"[ \u00A0]{2,}", " ", cleaned)

    # Trim trailing whitespace on each line, then strip the whole block.
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = cleaned.strip()

    return cleaned


def truncate_text(
    text: str, max_length: int, suffix: str = DEFAULT_TRUNCATE_SUFFIX
) -> str:
    """
    Safely truncate text to a maximum length, appending a suffix marker
    if truncation occurred. Never raises on invalid/edge-case input.

    Args:
        text: The text to truncate.
        max_length: The maximum allowed length of the returned string
            (including the suffix).
        suffix: The marker appended when truncation occurs (default "...").

    Returns:
        The original text if it's within `max_length`, otherwise a
        truncated version ending with `suffix`.
    """
    if not text:
        return ""

    if max_length <= 0:
        return ""

    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return text[:max_length]

    return text[: max_length - len(suffix)].rstrip() + suffix


# --------------------------------------------------------------------------
# List Helpers
# --------------------------------------------------------------------------

def remove_duplicates(list_data: List[_T]) -> List[_T]:
    """
    Remove duplicate values from a list while preserving original order.

    Unhashable items (e.g. dicts) are supported via a fallback that
    compares by JSON serialization; if that also fails, items are
    compared by identity/equality using a linear scan.

    Args:
        list_data: The list of values to de-duplicate.

    Returns:
        A new list with duplicates removed, preserving first-seen order.
    """
    if not list_data:
        return []

    try:
        seen_hashable: set = set()
        result: List[_T] = []
        for item in list_data:
            if item not in seen_hashable:
                seen_hashable.add(item)
                result.append(item)
        return result
    except TypeError:
        # Items are unhashable (e.g. dicts/lists); fall back to a
        # JSON-based dedup key, then a linear-scan fallback if that fails.
        seen_keys: set = set()
        result = []
        for item in list_data:
            try:
                key = json.dumps(item, sort_keys=True, default=str)
            except (TypeError, ValueError):
                if item not in result:
                    result.append(item)
                continue
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(item)
        return result


# --------------------------------------------------------------------------
# Network Error Handling Decorator
# --------------------------------------------------------------------------

def safe_request(function: Callable[..., _T]) -> Callable[..., Any]:
    """
    Decorator that wraps a network-calling function, catching common
    `requests` exceptions (timeouts, connection errors, and other request
    errors) and returning a friendly error dictionary instead of raising.

    The decorated function's normal return value is passed through
    unchanged on success.

    Args:
        function: The function to wrap (typically one that performs an
            HTTP request via the `requests` library).

    Returns:
        A wrapped version of `function` that never raises `requests`
        exceptions; on failure it returns `{"error": "<friendly message>"}`.
    """
    logger = logging.getLogger(function.__module__)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Imported locally to keep `requests` an optional/soft dependency
        # for any consumer that only needs the non-network utilities.
        import requests

        try:
            return function(*args, **kwargs)
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout in %s: %s", function.__name__, exc)
            return {"error": "The request timed out. Please try again."}
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error in %s: %s", function.__name__, exc)
            return {
                "error": "Unable to connect. Please check your internet connection."
            }
        except requests.exceptions.RequestException as exc:
            logger.error("Request error in %s: %s", function.__name__, exc)
            return {"error": f"A network request error occurred: {exc}"}

    return wrapper


# --------------------------------------------------------------------------
# Filesystem Helpers
# --------------------------------------------------------------------------

def create_reports_directory(directory: str = REPORTS_DIRECTORY) -> str:
    """
    Ensure the reports output directory exists, creating it if necessary.

    Args:
        directory: The directory path to ensure exists (default "reports").

    Returns:
        The (now guaranteed to exist) directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """
    Generate a filesystem-safe timestamp string for the current moment.

    Returns:
        A string in the format "YYYY-MM-DD_HH-MM-SS".
    """
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def format_company_name(name: str, max_length: int = DEFAULT_MAX_FILENAME_LENGTH) -> str:
    """
    Convert a company name into a safe string for use in filenames,
    stripping illegal filesystem characters and collapsing whitespace.

    Args:
        name: The raw company name.
        max_length: Maximum length of the resulting safe name.

    Returns:
        A filesystem-safe version of the company name (falls back to
        "Unknown_Company" if `name` is empty or becomes empty after
        cleaning).
    """
    if not name or not isinstance(name, str):
        return "Unknown_Company"

    without_illegal_chars = ILLEGAL_FILENAME_CHARS_PATTERN.sub("", name)
    collapsed = re.sub(r"\s+", "_", without_illegal_chars.strip())
    safe_name = collapsed.strip("._-")

    if not safe_name:
        return "Unknown_Company"

    return safe_name[:max_length]


# --------------------------------------------------------------------------
# JSON Helpers
# --------------------------------------------------------------------------

def save_json(data: Any, path: str, indent: int = 2) -> str:
    """
    Write data to a JSON file, creating any necessary parent directories.

    Args:
        data: The JSON-serializable data to write.
        path: The destination file path.
        indent: Indentation level for pretty-printing (default 2).

    Returns:
        The file path that was written to.

    Raises:
        OSError: If the file could not be written due to a filesystem error.
        TypeError: If `data` is not JSON-serializable.
    """
    logger = logging.getLogger(__name__)

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=indent, ensure_ascii=False, default=str)
    except TypeError as exc:
        logger.error("Failed to serialize data to JSON for %s: %s", path, exc)
        raise
    except OSError as exc:
        logger.error("Failed to write JSON file %s: %s", path, exc)
        raise

    logger.info("Saved JSON data to %s", path)
    return path


def load_json(path: str) -> Any:
    """
    Read and parse JSON data from a file.

    Args:
        path: The path to the JSON file to read.

    Returns:
        The parsed JSON data (typically a dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    logger = logging.getLogger(__name__)

    if not os.path.isfile(path):
        logger.error("JSON file not found: %s", path)
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from %s: %s", path, exc)
        raise

    logger.info("Loaded JSON data from %s", path)
    return data


# --------------------------------------------------------------------------
# Module Self-Test (only runs when executed directly, not on import)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    test_logger = setup_logger(__name__)
    test_logger.info("utils.py self-test starting")

    assert is_valid_url("https://tesla.com") is True
    assert is_valid_url("not a url") is False

    messy = "Hello   world\n\n\n\tThis is\ta test.  "
    assert clean_text(messy) == "Hello world\nThis is a test."

    assert truncate_text("Hello World", 5) == "He..."
    assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]

    safe_name = format_company_name('Tesla, Inc. / R&D?*')
    test_logger.info("Formatted company name: %s", safe_name)

    reports_dir = create_reports_directory()
    timestamp = get_timestamp()
    test_logger.info("Reports dir: %s, timestamp: %s", reports_dir, timestamp)

    sample_path = os.path.join(reports_dir, f"utils_selftest_{timestamp}.json")
    save_json({"company": "Tesla", "valid": True}, sample_path)
    loaded = load_json(sample_path)
    assert loaded["company"] == "Tesla"

    test_logger.info("utils.py self-test completed successfully")