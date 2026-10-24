"""
app.py – Streamlit UI for STEM Formula Extractor & Flashcards.

Supports both local execution (via .env) and Streamlit Cloud (via st.secrets).
Powered by Gemma 4 models via Google GenAI SDK.
"""

import io
import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load local environment variables from .env file if present
load_dotenv()

# --------------------------------------------------------------------------- #
# Supported Models – Gemma 4 only. First entry is preferred default.          #
# --------------------------------------------------------------------------- #
MODELS = [
    "gemma-4-26b-a4b-it",  # Faster default: 4B active parameters (MoE)
    "gemma-4-31b-it",      # High-capacity fallback: 31B dense parameters
]

from cards import (
    build_formula_synthesizer_prompt,
    build_prompt,
    cards_to_csv,
    get_pdf_page_count,
    parse_cards,
    parse_formulas,
    pdf_page_to_image,
    shrink_image,
)

# --------------------------------------------------------------------------- #
# Retry / fallback configuration                                               #
# --------------------------------------------------------------------------- #

_MODEL_RETRIES = 5
_RETRY_DELAYS = [2, 4, 8, 16, 32]

_RETRYABLE_SIGNALS = ("500", "503", "internal", "unavailable")
_SKIP_MODEL_SIGNALS = ("404", "not found", "not_found")
_FATAL_SIGNALS = (
    "400",
    "401",
    "403",
    "api_key",
    "invalid key",
    "permission",
    "billing",
)


def _classify(exc: Exception) -> str:
    """Return one of 'retry', 'skip', 'fatal', or 'unknown' for an exception."""
    msg = str(exc).lower()
    if any(s in msg for s in _FATAL_SIGNALS):
        return "fatal"
    if any(s in msg for s in _SKIP_MODEL_SIGNALS):
        return "skip"
    if any(s in msg for s in _RETRYABLE_SIGNALS):
        return "retry"
    return "unknown"


def _get_api_key() -> str:
    """Read API key from local environment variable (.env) or Streamlit Cloud secrets."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return str(key).strip()


# --------------------------------------------------------------------------- #
# Core API function with Gemma 4 fallback                                      #
# --------------------------------------------------------------------------- #


def generate_with_fallback(
    contents: list,
    preferred_model: str | None = None,
) -> tuple[str, str]:
    """
    Try each model in MODELS (preferred first) and return (response_text, model_name).

    Per-model behaviour:
      - Retryable error (500/503/INTERNAL/UNAVAILABLE): retry up to _MODEL_RETRIES
        times with _RETRY_DELAYS waits, then fall through to next model.
      - 404 / model not found: skip to next model immediately.
      - Fatal error (400/401/403 / bad key): raise RuntimeError immediately.
      - Any other error: fall through to next model.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Set it in your local .env or Streamlit Cloud secrets."
        )

    client = genai.Client(api_key=api_key)

    if preferred_model and preferred_model in MODELS:
        ordered = [preferred_model] + [m for m in MODELS if m != preferred_model]
    else:
        ordered = list(MODELS)

    last_error: Exception | None = None

    for model in ordered:
        for attempt in range(1 + _MODEL_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=8192,
                    ),
                )
                if response.text:
                    return response.text, model

                # Fallback: extract text from response candidates/parts
                text_parts = []
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if hasattr(part, "text") and part.text:
                                    text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts), model

                return "", model

            except Exception as exc:
                last_error = exc
                kind = _classify(exc)

                if kind == "fatal":
                    raise RuntimeError(
                        f"Authentication / quota error – check your API key. Details: {exc}"
                    ) from exc

                if kind == "skip":
                    break

                if kind == "retry" and attempt < _MODEL_RETRIES:
                    wait = _RETRY_DELAYS[attempt]
                    try:
                        st.toast(
                            f"⏳ {model} busy – retrying in {wait}s (attempt {attempt + 1}/{_MODEL_RETRIES})…",
                            icon="🔄",
                        )
                    except Exception:
                        pass
                    time.sleep(wait)
                    continue

                if attempt >= _MODEL_RETRIES:
                    try:
                        st.toast(
                            f"⚠️ {model} exceeded retries. Switching to fallback model…",
                            icon="🔀",
                        )
                    except Exception:
                        pass
                break

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# --------------------------------------------------------------------------- #
# Page Configuration                                                          #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="STEM Formula Extractor & Flashcards",
    page_icon="⚛️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Sidebar – Configuration & Mode                                               #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🎯 Mode Selection")
    active_mode = st.radio(
        "Choose Mode",
        options=["📐 Formula Extractor", "🃏 Flashcards"],
        index=0,
        help="Select whether to extract formula cards or generate study flashcards.",
    )

    preferred_model = st.selectbox(
        "Gemma 4 Model",
        options=MODELS,
        index=0,
        help="gemma-4-26b-a4b-it is fast MoE; gemma-4-31b-it is high-capacity fallback.",
    )

    num_items = st.slider(
        "Number of items to generate",
        min_value=3,
        max_value=15,
        value=6,
        help="Number of formulas or flashcards to extract.",
    )

    language = st.selectbox(
        "Language",
        options=["English", "Hindi"],
        index=0,
        help="Language for explanations and card text.",
    )

    st.divider()
    st.markdown(
        "<small>Powered by <b>Gemma 4</b> via Google GenAI SDK</small>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Main Header                                                                  #
# --------------------------------------------------------------------------- #

st.title("⚛️ STEM Formula Extractor & Flashcards")
st.caption(
    "Extract mathematical laws, SI variable units, and validity assumptions, "
    "or generate study flashcards directly from your handwritten notes, PDFs, or text."
)
st.divider()

# --------------------------------------------------------------------------- #
# Input Section                                                                #
# --------------------------------------------------------------------------- #

st.subheader("📥 Input Notes")

input_tab_upload, input_tab_text = st.tabs(
    [
        "📁 Upload Notes / PDF",
        "📝 Paste Text Notes",
    ]
)

active_text = ""
active_image_bytes = None

# Tab 1: Upload Notes / PDF
with input_tab_upload:
    uploaded = st.file_uploader(
        "Upload handwritten notes, textbook page, or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Supports PNG, JPG, JPEG, and PDF documents.",
    )
    if uploaded:
        raw_bytes = uploaded.read()
        is_pdf = uploaded.name.lower().endswith(".pdf") or uploaded.type == "application/pdf"

        if is_pdf:
            try:
                total_pages = get_pdf_page_count(raw_bytes)
            except ValueError as err:
                st.error(f"❌ Invalid PDF: {err}")
                st.stop()

            selected_page = 1
            if total_pages > 1:
                selected_page = st.number_input(
                    f"Select page to extract from (1 to {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                )

            pdf_cache_key = f"pdf_preview_{uploaded.name}_{selected_page}"
            if pdf_cache_key not in st.session_state:
                try:
                    st.session_state[pdf_cache_key] = pdf_page_to_image(
                        raw_bytes, page_number=selected_page - 1, dpi_scale=1.5
                    )
                except ValueError as err:
                    st.error(f"❌ Could not render PDF page: {err}")
                    st.stop()
            preview_bytes = st.session_state[pdf_cache_key]
            img = Image.open(io.BytesIO(preview_bytes))
            st.image(
                img,
                caption=f"{uploaded.name} (Page {selected_page} of {total_pages})",
                use_container_width=True,
            )
            active_image_bytes = preview_bytes
        else:
            if "img_rotation" not in st.session_state:
                st.session_state["img_rotation"] = 0

            col_img_h, col_img_r = st.columns([3, 1])
            with col_img_r:
                if st.button("🔄 Rotate 90°", help="Rotate image clockwise if notes appear sideways"):
                    st.session_state["img_rotation"] = (st.session_state["img_rotation"] + 90) % 360
                    st.rerun()

            rot_deg = st.session_state.get("img_rotation", 0)
            img = Image.open(io.BytesIO(raw_bytes))
            if rot_deg != 0:
                img = img.rotate(-rot_deg, expand=True)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=90)
                active_image_bytes = buf.getvalue()
            else:
                active_image_bytes = raw_bytes

            st.image(
                img,
                caption=uploaded.name + (f" (Rotated {rot_deg}°)" if rot_deg else ""),
                use_container_width=True,
            )

# Tab 2: Paste Raw Text Notes
with input_tab_text:
    sample_text = (
        "Thermodynamics First & Second Law:\n"
        "Delta U = Q - W, where Delta U is change in internal energy (Joules), "
        "Q is heat added to the system (Joules), and W is work done by the system (Joules).\n"
        "For an ideal gas, work during reversible isothermal expansion is W = n * R * T * ln(V_final / V_initial).\n"
        "Carnot engine efficiency: eta = 1 - (T_cold / T_hot), valid only for reversible Carnot cycles operating "
        "between two thermal reservoirs."
    )
    pasted_notes = st.text_area(
        "Paste STEM formulas, derivations, or lecture notes",
        value="" if uploaded else sample_text,
        height=180,
        placeholder="Type or paste your STEM notes here...",
    )
    if not active_image_bytes and pasted_notes.strip():
        active_text = pasted_notes.strip()

st.divider()

# --------------------------------------------------------------------------- #
# Action Button                                                                #
# --------------------------------------------------------------------------- #

btn_title = f"✨ Run {active_mode}"
if st.button(btn_title, type="primary", use_container_width=True):
    if not _get_api_key():
        st.error(
            "❌ **GEMINI_API_KEY is not configured.**\n\n"
            "- If running locally: create a `.env` file with `GEMINI_API_KEY=your_key`.\n"
            "- If running on Streamlit Cloud: set `GEMINI_API_KEY` in **App Settings → Secrets**."
        )
        st.stop()

    contents_list = []
    if active_image_bytes:
        s_bytes, s_mime = shrink_image(active_image_bytes, max_side=1200, quality=80)
        contents_list.append(types.Part.from_bytes(data=s_bytes, mime_type=s_mime))

    if active_text.strip():
        text_prefix = "The following is the STEM content to analyze:\n\n" + active_text.strip()
        contents_list.append(types.Part.from_text(text=text_prefix))

    if not contents_list:
        st.error("❌ Please upload an image/PDF or paste notes before generating.")
        st.stop()

    with st.spinner(f"Processing with {preferred_model}…"):
        try:
            if "Formula Extractor" in active_mode:
                formula_prompt = build_formula_synthesizer_prompt(language)
                call_contents = [
                    types.Content(parts=contents_list + [types.Part.from_text(text=formula_prompt)])
                ]
                raw_resp, used_m = generate_with_fallback(
                    call_contents, preferred_model=preferred_model
                )
                st.session_state["formulas"] = parse_formulas(raw_resp)
                st.session_state["used_model"] = used_m
            else:
                card_prompt = build_prompt(num_items, language)
                call_contents = [
                    types.Content(parts=contents_list + [types.Part.from_text(text=card_prompt)])
                ]
                raw_resp, used_m = generate_with_fallback(
                    call_contents, preferred_model=preferred_model
                )
                st.session_state["cards"] = parse_cards(raw_resp)
                st.session_state["used_model"] = used_m
            st.success("✅ Generation complete!")
        except Exception as exc:
            st.error(f"❌ Generation failed: {exc}")
            st.stop()

# --------------------------------------------------------------------------- #
# Output Tabs: Only Formula Extractor & Flashcards                             #
# --------------------------------------------------------------------------- #

has_formulas = "formulas" in st.session_state and st.session_state["formulas"]
has_cards = "cards" in st.session_state and st.session_state["cards"]

if has_formulas or has_cards:
    st.divider()

    tab_formulas, tab_cards = st.tabs(["📐 Formula Extractor", "🃏 Flashcards"])

    # 1. Formula Extractor Tab
    with tab_formulas:
        if has_formulas:
            formulas = st.session_state["formulas"]
            st.markdown(f"### 📐 Extracted Formulas ({len(formulas)})")
            st.caption("Governing equations, variable definitions with SI units, and validity conditions.")

            for i, f in enumerate(formulas, start=1):
                with st.container(border=True):
                    col_title, col_idx = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"#### {i}. {f.get('name', 'Formula')}")
                    with col_idx:
                        st.caption(f"`Formula #{i}`")

                    # LaTeX equation
                    raw_eq = f.get("latex_equation", "").strip()
                    if raw_eq:
                        try:
                            st.latex(raw_eq)
                        except Exception:
                            st.markdown(f"$${raw_eq}$$")

                    if f.get("description"):
                        st.markdown(f"**Description:** {f['description']}")

                    # Variables Table
                    vars_list = f.get("variables", [])
                    if vars_list:
                        st.markdown("**Variables & SI Units:**")
                        table_rows = ["| Symbol | Definition | SI Unit |", "| :--- | :--- | :--- |"]
                        for v in vars_list:
                            sym = v.get("symbol", "-")
                            defn = v.get("definition", "-")
                            unit = v.get("unit", "-")
                            table_rows.append(f"| `${sym}$` | {defn} | `{unit}` |")
                        st.markdown("\n".join(table_rows))

                    # Assumptions
                    assumptions = f.get("assumptions")
                    if assumptions:
                        st.info(f"**⚠️ Validity Limits & Conditions:** {assumptions}")
        else:
            st.info("ℹ️ Select **📐 Formula Extractor** in the sidebar and click Run to extract formulas.")

    # 2. Flashcards Tab
    with tab_cards:
        if has_cards:
            cards = st.session_state["cards"]
            st.markdown(f"### 🃏 Study Flashcards ({len(cards)})")
            st.caption("Review flashcards with LaTeX equations or export to Anki.")

            for i, card in enumerate(cards, start=1):
                with st.expander(f"Card {i}: {card['front']}"):
                    st.markdown(card["back"])

            st.divider()
            csv_text = cards_to_csv(cards)
            st.download_button(
                label="⬇️ Download CSV for Anki",
                data=csv_text.encode("utf-8"),
                file_name="stem_flashcards.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("ℹ️ Select **🃏 Flashcards** in the sidebar and click Run to generate flashcards.")

# --------------------------------------------------------------------------- #
# Footer                                                                       #
# --------------------------------------------------------------------------- #

st.divider()
used_model_display = st.session_state.get("used_model", "Gemma 4")
st.caption(f"Powered by {used_model_display} via Google GenAI SDK")
