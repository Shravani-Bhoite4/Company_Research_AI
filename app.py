"""
app.py
======

Company Research Assistant - Streamlit Application.

This is the top-level orchestration layer for the AI Company Research
Assistant. It provides a modern, ChatGPT-style Streamlit interface that
ties together:

    * search.py         - Serper.dev company/website search
    * crawler.py         - website crawling and content extraction
    * ai.py               - OpenRouter AI business report generation
    * pdf_generator.py   - professional PDF report generation
    * utils.py            - shared helpers (env, logging, cleaning, etc.)

Workflow
--------
    1. User enters a company name OR a website URL in the chat input.
    2. If a company name was given, resolve its official website via
       search.get_official_website().
    3. Gather company profile data via search.search_company_information().
    4. Crawl the website via crawler.crawl_website().
    5. Combine crawled page content via crawler.combine_content().
    6. Generate a structured AI business report via ai.generate_ai_report().
    7. Display company info cards, the AI report, and competitor cards.
    8. Generate a downloadable PDF via pdf_generator.create_pdf().
    9. Offer a download button for the generated PDF.

This module contains UI code (Streamlit) and orchestration logic only;
all business logic lives in the imported modules.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from ai import DEFAULT_MODEL, generate_ai_report, get_available_model
from crawler import combine_content, crawl_website
from pdf_generator import PDFGenerationError, create_pdf
from search import get_official_website, search_company_information
from utils import (
    EnvironmentConfigurationError,
    clean_text,
    format_company_name,
    is_valid_url,
    load_environment,
    remove_duplicates,
    setup_logger,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

APP_TITLE: str = "AI Company Research Assistant"
APP_ICON: str = "🏢"

CURATED_MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat:free",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
]
CUSTOM_MODEL_OPTION: str = "Custom model..."

MAX_COMPETITOR_WEBSITE_LOOKUPS: int = 6

PIPELINE_STEPS: List[Dict[str, Any]] = [
    {"label": "Searching company information...", "progress": 15},
    {"label": "Crawling official website...", "progress": 40},
    {"label": "Analyzing content with AI...", "progress": 70},
    {"label": "Generating PDF report...", "progress": 90},
    {"label": "Completed", "progress": 100},
]

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logger: logging.Logger = setup_logger(__name__)


# --------------------------------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------------------------------

def configure_page() -> None:
    """
    Configure Streamlit page metadata and apply light custom CSS for a
    modern, professional, theme-adaptive appearance.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal, theme-adaptive CSS: relies on Streamlit's native light/dark
    # variables rather than hardcoded colors, so it stays readable in both
    # themes. Adds card polish, spacing, and subtle hover affordances.
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1100px;
            }
            div[data-testid="stChatMessage"] {
                border-radius: 14px;
                padding: 0.25rem 0.5rem;
            }
            .research-card {
                border-radius: 12px;
                padding: 1rem 1.25rem;
                border: 1px solid rgba(128, 128, 128, 0.25);
                margin-bottom: 0.75rem;
            }
            .research-card h4 {
                margin-top: 0;
                margin-bottom: 0.5rem;
            }
            .pill {
                display: inline-block;
                padding: 0.15rem 0.65rem;
                border-radius: 999px;
                border: 1px solid rgba(128, 128, 128, 0.35);
                font-size: 0.8rem;
                margin: 0.15rem 0.25rem 0.15rem 0;
            }
            section[data-testid="stSidebar"] .stButton button {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------

def initialize_session_state() -> None:
    """
    Initialize all Streamlit session state keys used across reruns, if
    they are not already present.
    """
    defaults: Dict[str, Any] = {
        "messages": [],
        "last_company_info": None,
        "last_ai_report": None,
        "last_pdf_path": None,
        "is_processing": False,
        "applicant_name": "",
        "applicant_email": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_api_status() -> None:
    """
    Check for required API keys and render a friendly status indicator
    for each in the sidebar (without making a live network call).
    """
    st.sidebar.markdown("### 🔌 API Status")
    try:
        load_environment()
        st.sidebar.success("Serper.dev API key detected")
        st.sidebar.success("OpenRouter API key detected")
    except EnvironmentConfigurationError as exc:
        st.sidebar.error(f"Configuration issue: {exc}")


def render_model_selection() -> str:
    """
    Render the AI model selection controls in the sidebar and return the
    resolved OpenRouter model identifier to use for report generation.

    Returns:
        The selected (or custom) OpenRouter model identifier string.
    """
    st.sidebar.markdown("### 🤖 AI Model Selection")

    options = CURATED_MODELS + [CUSTOM_MODEL_OPTION]
    selected = st.sidebar.selectbox(
        "Choose an OpenRouter model",
        options=options,
        index=0,
        help="Select any OpenRouter-supported model. Free-tier models are "
        "marked with ':free'.",
    )

    if selected == CUSTOM_MODEL_OPTION:
        custom_model = st.sidebar.text_input(
            "Custom model identifier",
            placeholder="e.g. openai/gpt-4o-mini",
        )
        resolved_model = get_available_model(custom_model or None)
    else:
        resolved_model = get_available_model(selected)

    st.sidebar.caption(f"Active model: `{resolved_model}`")
    return resolved_model


def render_applicant_info() -> None:
    """
    Render sidebar inputs for the applicant's name and email, storing
    them in session state for use in report context/metadata.
    """
    st.sidebar.markdown("### 👤 Applicant Details")
    st.session_state["applicant_name"] = st.sidebar.text_input(
        "Your Name", value=st.session_state.get("applicant_name", "")
    )
    st.session_state["applicant_email"] = st.sidebar.text_input(
        "Your Email", value=st.session_state.get("applicant_email", "")
    )


def render_theme_information() -> None:
    """
    Render a small informational panel about theming in the sidebar.
    """
    with st.sidebar.expander("🎨 Theme Information"):
        st.write(
            "This app automatically adapts to your system's light or dark "
            "theme. You can switch themes from Streamlit's settings menu "
            "(top-right \u22ee menu → Settings → Theme)."
        )


def render_sidebar() -> str:
    """
    Render the full sidebar (model selection, applicant info, theme
    info, and API status) and return the resolved AI model to use.

    Returns:
        The resolved OpenRouter model identifier string.
    """
    st.sidebar.title(f"{APP_ICON} Research Assistant")
    model = render_model_selection()
    render_applicant_info()
    render_theme_information()
    render_api_status()
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit, Serper.dev, and OpenRouter.")
    return model


# --------------------------------------------------------------------------
# Card Rendering Helpers
# --------------------------------------------------------------------------

def render_company_info_card(company_info: Dict[str, Any]) -> None:
    """
    Render a professional card showing core company information (name,
    website, phone, address).

    Args:
        company_info: The aggregated company information dictionary.
    """
    with st.container(border=True):
        st.markdown(f"#### 🏢 {company_info.get('company_name', 'Unknown Company')}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🌐 Website:** {company_info.get('website') or 'Not available'}")
            st.markdown(f"**📞 Phone:** {company_info.get('phone') or 'Not available'}")
        with col2:
            st.markdown(f"**📍 Address:** {company_info.get('address') or 'Not available'}")


def render_bullet_card(title: str, icon: str, items: List[str]) -> None:
    """
    Render a card containing a bulleted list of items (e.g. products,
    services, pain points), or an empty-state message.

    Args:
        title: The card's section title.
        icon: An emoji icon to prefix the title with.
        items: The list of string items to display as bullets.
    """
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        cleaned_items = remove_duplicates(
            [str(item).strip() for item in (items or []) if str(item).strip()]
        )
        if not cleaned_items:
            st.caption(f"No {title.lower()} identified.")
        else:
            for item in cleaned_items:
                st.markdown(f"- {item}")


def render_competitor_cards(competitors: List[Dict[str, str]]) -> None:
    """
    Render competitor information as a responsive grid of cards, each
    showing the competitor's name and (if available) website link.

    Args:
        competitors: A list of dictionaries shaped as
            {"name": str, "website": str}.
    """
    st.markdown("#### 🥊 Competitors")

    if not competitors:
        st.caption("No competitors identified.")
        return

    columns_per_row = 3
    for row_start in range(0, len(competitors), columns_per_row):
        row_items = competitors[row_start : row_start + columns_per_row]
        columns = st.columns(len(row_items))
        for column, competitor in zip(columns, row_items):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{competitor.get('name') or 'Unknown'}**")
                    website = competitor.get("website")
                    if website:
                        st.markdown(f"[{website}]({website})")
                    else:
                        st.caption("Website not found")


def render_ai_report(ai_report: Dict[str, Any]) -> None:
    """
    Render the full AI-generated business report using expanders for
    each analytical section, keeping the main view uncluttered.

    Args:
        ai_report: The structured AI report dictionary from
            ai.generate_ai_report().
    """
    st.markdown("### 🧠 AI Business Analysis")

    with st.container(border=True):
        st.markdown("#### 📋 Summary")
        st.write(ai_report.get("summary") or "No summary available.")

    with st.container(border=True):
        st.markdown("#### 💼 Business Model")
        st.write(ai_report.get("business_model") or "Not available.")

    section_definitions = [
        ("target_customers", "🎯 Target Customers"),
        ("strengths", "💪 Strengths"),
        ("weaknesses", "⚠️ Weaknesses"),
        ("growth_opportunities", "📈 Growth Opportunities"),
        ("technology_stack_guess", "🛠️ Technology Stack (Best Guess)"),
    ]

    for key, label in section_definitions:
        items = ai_report.get(key) or []
        with st.expander(label, expanded=False):
            if not items:
                st.caption("No information available.")
            else:
                for item in items:
                    st.markdown(f"- {item}")


# --------------------------------------------------------------------------
# Pipeline Helpers
# --------------------------------------------------------------------------

def _derive_company_name_from_url(url: str) -> str:
    """
    Derive a human-friendly company name guess from a website URL, used
    for search queries when the user directly provides a URL.

    Args:
        url: The website URL provided by the user.

    Returns:
        A best-effort, title-cased company name derived from the domain.
    """
    from urllib.parse import urlparse

    netloc = urlparse(url if "://" in url else f"https://{url}").netloc
    domain_root = netloc.replace("www.", "").split(".")[0]
    return domain_root.replace("-", " ").title() or "Unknown Company"


def _resolve_competitors_with_websites(
    competitor_names: List[str], max_lookups: int = MAX_COMPETITOR_WEBSITE_LOOKUPS
) -> List[Dict[str, str]]:
    """
    Convert a list of competitor names (as returned by the AI report)
    into structured dictionaries with a best-effort resolved website,
    limiting the number of live lookups performed.

    Args:
        competitor_names: List of competitor company names.
        max_lookups: Maximum number of website resolution lookups to
            perform, to bound API usage.

    Returns:
        A list of dictionaries shaped as {"name": str, "website": str}.
    """
    resolved: List[Dict[str, str]] = []
    cleaned_names = remove_duplicates(
        [str(name).strip() for name in competitor_names if str(name).strip()]
    )

    for index, name in enumerate(cleaned_names):
        website = ""
        if index < max_lookups:
            try:
                website = get_official_website(name) or ""
            except Exception as exc:  # noqa: BLE001 - never let a lookup break the pipeline.
                logger.warning("Failed to resolve website for competitor %s: %s", name, exc)
                website = ""
        resolved.append({"name": name, "website": website})

    return resolved


def run_research_pipeline(
    user_input: str, model: str, progress_callback: Any
) -> Dict[str, Any]:
    """
    Execute the full company research pipeline: search, crawl, analyze,
    and prepare data for PDF generation.

    Args:
        user_input: The company name or website URL entered by the user.
        model: The OpenRouter model identifier to use for AI analysis.
        progress_callback: A callable of signature (label: str, percent: int)
            used to report progress back to the UI.

    Returns:
        A dictionary shaped as:
            {
                "success": bool,
                "error": Optional[str],
                "company_info": Optional[Dict[str, Any]],
                "ai_report": Optional[Dict[str, Any]],
            }
    """
    result: Dict[str, Any] = {
        "success": False,
        "error": None,
        "company_info": None,
        "ai_report": None,
    }

    cleaned_input = clean_text(user_input).strip()
    if not cleaned_input:
        result["error"] = "Please enter a valid company name or website URL."
        return result

    # --- Step 1 & 2: Resolve website ---
    progress_callback(PIPELINE_STEPS[0]["label"], PIPELINE_STEPS[0]["progress"])

    if is_valid_url(cleaned_input):
        website = cleaned_input
        search_name = _derive_company_name_from_url(cleaned_input)
    else:
        search_name = cleaned_input
        website = get_official_website(cleaned_input)
        if not website:
            result["error"] = (
                f"Could not find an official website for '{cleaned_input}'. "
                "Please try a different company name or provide a website URL directly."
            )
            return result

    # --- Step 3: Company information search ---
    search_results = search_company_information(search_name)
    if search_results.get("error"):
        result["error"] = search_results["error"]
        return result

    # Prefer the explicitly resolved/provided website over the search result.
    search_results["website"] = website or search_results.get("website", "")

    # --- Step 4: Crawl website ---
    progress_callback(PIPELINE_STEPS[1]["label"], PIPELINE_STEPS[1]["progress"])
    try:
        pages = crawl_website(website)
    except Exception as exc:  # noqa: BLE001 - surface as a friendly crawl error.
        import traceback
        traceback.print_exc()
        logger.error("Crawling failed for %s: %s", website, exc)
        pages = []

    if not pages:
        result["error"] = (
            f"Could not crawl the website '{website}'. It may be unreachable, "
            "blocking automated access, or invalid."
        )
        return result

    # --- Step 5: Combine content ---
    combined_content = combine_content(pages)

    # --- Step 6: AI report generation ---
    progress_callback(PIPELINE_STEPS[2]["label"], PIPELINE_STEPS[2]["progress"])
    ai_input = {
        "company_name": search_results.get("company_name") or search_name,
        "website": website,
        "description": search_results.get("description", ""),
        "products": search_results.get("products", []),
        "services": search_results.get("services", []),
        "website_content": combined_content,
    }
    ai_report = generate_ai_report(ai_input, model=model)

    if ai_report.get("error"):
        result["error"] = ai_report["error"]
        return result

    # --- Merge into final company info ---
    company_info: Dict[str, Any] = {
        "company_name": ai_input["company_name"],
        "website": website,
        "phone": search_results.get("phone", ""),
        "address": search_results.get("address", ""),
        "summary": ai_report.get("summary", ""),
        "products": ai_report.get("products") or search_results.get("products", []),
        "services": ai_report.get("services") or search_results.get("services", []),
        "pain_points": ai_report.get("pain_points", []),
        "competitors": _resolve_competitors_with_websites(ai_report.get("competitors", [])),
    }

    result["success"] = True
    result["company_info"] = company_info
    result["ai_report"] = ai_report
    return result


def generate_pdf_report(company_info: Dict[str, Any], progress_callback: Any) -> Optional[str]:
    """
    Generate the downloadable PDF report for the researched company.

    Args:
        company_info: The merged company information dictionary.
        progress_callback: A callable of signature (label: str, percent: int)
            used to report progress back to the UI.

    Returns:
        The full file path to the generated PDF, or None if generation failed.
    """
    progress_callback(PIPELINE_STEPS[3]["label"], PIPELINE_STEPS[3]["progress"])
    try:
        pdf_path = create_pdf(company_info)
        return pdf_path
    except PDFGenerationError as exc:
        logger.error("PDF generation failed: %s", exc)
        st.error(f"Failed to generate PDF report: {exc}")
        return None


# --------------------------------------------------------------------------
# Chat & Main Content Rendering
# --------------------------------------------------------------------------

def render_chat_history() -> None:
    """
    Render all prior chat messages stored in session state.
    """
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_results_section() -> None:
    """
    Render the most recent research results (company cards, AI report,
    competitor cards, and PDF download button), if any exist in session
    state.
    """
    company_info = st.session_state.get("last_company_info")
    ai_report = st.session_state.get("last_ai_report")
    pdf_path = st.session_state.get("last_pdf_path")

    if not company_info:
        return

    st.markdown("---")
    st.markdown("## 📊 Research Results")

    render_company_info_card(company_info)

    col1, col2 = st.columns(2)
    with col1:
        render_bullet_card("Products", "📦", company_info.get("products", []))
    with col2:
        render_bullet_card("Services", "🛎️", company_info.get("services", []))

    render_bullet_card("Pain Points", "🩹", company_info.get("pain_points", []))

    render_competitor_cards(company_info.get("competitors", []))

    if ai_report:
        with st.expander("🧠 View Full AI Business Analysis", expanded=False):
            render_ai_report(ai_report)

    if pdf_path:
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=pdf_path.split("/")[-1].split("\\")[-1],
                mime="application/pdf",
                use_container_width=True,
            )
        except OSError as exc:
            logger.error("Could not read generated PDF at %s: %s", pdf_path, exc)
            st.error("The generated PDF could not be loaded for download.")


def handle_user_query(user_input: str, model: str) -> None:
    """
    Handle a new user query end-to-end: run the research pipeline with
    live progress feedback, update chat history, and store results in
    session state for rendering.

    Args:
        user_input: The company name or URL entered by the user.
        model: The resolved OpenRouter model identifier to use.
    """
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        progress_bar = st.progress(0, text="Starting research...")
        status_placeholder = st.empty()

        def progress_callback(label: str, percent: int) -> None:
            status_placeholder.info(label)
            progress_bar.progress(percent, text=label)

        with st.spinner("Researching company, please wait..."):
            pipeline_result = run_research_pipeline(user_input, model, progress_callback)

            if not pipeline_result["success"]:
                progress_bar.empty()
                status_placeholder.empty()
                error_message = pipeline_result["error"] or "An unknown error occurred."
                st.error(f"❌ {error_message}")
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"❌ {error_message}"}
                )
                logger.error("Research pipeline failed: %s", error_message)
                return

            company_info = pipeline_result["company_info"]
            ai_report = pipeline_result["ai_report"]

            pdf_path = generate_pdf_report(company_info, progress_callback)

            progress_callback(PIPELINE_STEPS[4]["label"], PIPELINE_STEPS[4]["progress"])

        progress_bar.empty()
        status_placeholder.empty()

        st.session_state["last_company_info"] = company_info
        st.session_state["last_ai_report"] = ai_report
        st.session_state["last_pdf_path"] = pdf_path

        summary_text = company_info.get("summary") or "Research completed."
        assistant_message = (
            f"✅ Research complete for **{company_info.get('company_name')}**.\n\n"
            f"{summary_text}"
        )
        st.success("Research completed successfully!")
        st.markdown(assistant_message)
        st.session_state["messages"].append(
            {"role": "assistant", "content": assistant_message}
        )
        logger.info("Research pipeline completed for %s", company_info.get("company_name"))


# --------------------------------------------------------------------------
# Main Application Entry Point
# --------------------------------------------------------------------------

def main() -> None:
    """
    Application entry point: configure the page, render the sidebar and
    chat interface, and handle the research workflow when the user
    submits a query.
    """
    configure_page()
    initialize_session_state()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Enter a company name or website URL to generate an AI-powered "
        "business research report, complete with a downloadable PDF."
    )

    model = render_sidebar()

    render_chat_history()

    if st.session_state["messages"]:
        render_results_section()

    user_input = st.chat_input(
        "Enter a company name or website URL (e.g. 'Tesla' or 'https://tesla.com')"
    )

    if user_input:
        try:
            load_environment()
        except EnvironmentConfigurationError as exc:
            st.error(
                f"⚠️ Missing API configuration: {exc} "
                "Please configure your .env file before researching a company."
            )
            logger.error("Environment configuration error: %s", exc)
            return

        handle_user_query(user_input, model)
        st.rerun()


if __name__ == "__main__":
    main()