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
    build_counter_question_prompt,
    build_elaboration_prompt,
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
    "or generate study flashcards with interactive quizzing and conceptual Q&A."
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
                st.session_state["quiz_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["revealed"] = False
            st.success("✅ Generation complete!")
        except Exception as exc:
            st.error(f"❌ Generation failed: {exc}")
            st.stop()

# --------------------------------------------------------------------------- #
# Output Tabs                                                                  #
# --------------------------------------------------------------------------- #

has_formulas = "formulas" in st.session_state and st.session_state["formulas"]
has_cards = "cards" in st.session_state and st.session_state["cards"]

if has_formulas or has_cards:
    st.divider()

    tab_formulas, tab_cards, tab_quiz, tab_counter = st.tabs(
        [
            "📐 Formula Extractor",
            "📖 Browse Flashcards",
            "🎯 Quiz Mode",
            "💬 Counter Question",
        ]
    )

    # ----------------------------------------------------------------------- #
    # 1. Formula Extractor Tab                                                #
    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    # 2. Browse Flashcards Tab                                                #
    # ----------------------------------------------------------------------- #
    with tab_cards:
        if has_cards:
            cards = st.session_state["cards"]
            st.markdown(f"### 🃏 Study Flashcards ({len(cards)})")
            st.caption("Review flashcards with LaTeX equations or export to Anki.")

            for i, card in enumerate(cards, start=1):
                with st.expander(f"Card {i}: {card['front']}"):
                    st.markdown(card["back"])

                    elab_key = f"elab_browse_{i}_{hash(card['front']) % 10000}"
                    if elab_key in st.session_state:
                        st.info(st.session_state[elab_key])
                    else:
                        if st.button("💡 Elaborate explanation", key=f"btn_elab_{i}"):
                            with st.spinner("Asking Gemma 4 to elaborate…"):
                                try:
                                    elab_prompt = build_elaboration_prompt(
                                        card["front"], card["back"], language
                                    )
                                    elab_contents = [
                                        types.Content(parts=[types.Part.from_text(text=elab_prompt)])
                                    ]
                                    elab_text, _ = generate_with_fallback(
                                        elab_contents, preferred_model=preferred_model
                                    )
                                    st.session_state[elab_key] = elab_text
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Could not elaborate: {err}")

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

    # ----------------------------------------------------------------------- #
    # 3. Interactive Quiz Mode Tab                                            #
    # ----------------------------------------------------------------------- #
    with tab_quiz:
        if has_cards:
            cards = st.session_state["cards"]
            idx: int = st.session_state.get("quiz_index", 0)
            score: int = st.session_state.get("quiz_score", 0)
            revealed: bool = st.session_state.get("revealed", False)

            if idx >= len(cards):
                st.balloons()
                st.subheader("🎉 Quiz Complete!")
                st.metric("Score", f"{score} / {len(cards)}")
                pct = int(score / len(cards) * 100)
                st.progress(pct / 100)
                if pct == 100:
                    st.success("Perfect mastery! Outstanding! 🏆")
                elif pct >= 70:
                    st.info("Solid understanding! Review missed cards to master the topic.")
                else:
                    st.warning("Keep practicing – reinforcement builds intuition! 💪")

                if st.button("🔄 Restart Quiz", use_container_width=True):
                    st.session_state["quiz_index"] = 0
                    st.session_state["quiz_score"] = 0
                    st.session_state["revealed"] = False
                    st.rerun()
            else:
                card = cards[idx]
                st.markdown(f"**Card {idx + 1} of {len(cards)}**")
                st.progress(idx / len(cards))
                st.markdown("---")
                st.subheader("❓ " + card["front"])
                st.markdown("---")

                if not revealed:
                    if st.button("👁 Reveal Answer", use_container_width=True):
                        st.session_state["revealed"] = True
                        st.rerun()
                else:
                    st.info(card["back"])

                    quiz_elab_key = f"elab_quiz_{idx}_{hash(card['front']) % 10000}"
                    if quiz_elab_key in st.session_state:
                        st.markdown("---")
                        st.markdown("#### 💡 Deep Dive Explanation")
                        st.markdown(st.session_state[quiz_elab_key])
                    else:
                        if st.button("💡 Elaborate explanation", key=f"btn_quiz_elab_{idx}"):
                            with st.spinner("Asking Gemma 4 to elaborate…"):
                                try:
                                    elab_prompt = build_elaboration_prompt(
                                        card["front"], card["back"], language
                                    )
                                    elab_contents = [
                                        types.Content(parts=[types.Part.from_text(text=elab_prompt)])
                                    ]
                                    elab_text, _ = generate_with_fallback(
                                        elab_contents, preferred_model=preferred_model
                                    )
                                    st.session_state[quiz_elab_key] = elab_text
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Could not elaborate: {err}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ I knew it", use_container_width=True):
                            st.session_state["quiz_score"] += 1
                            st.session_state["quiz_index"] += 1
                            st.session_state["revealed"] = False
                            st.rerun()
                    with col2:
                        if st.button("🔁 Review again", use_container_width=True):
                            cards.append(cards[idx])
                            st.session_state["cards"] = cards
                            st.session_state["quiz_index"] += 1
                            st.session_state["revealed"] = False
                            st.rerun()
        else:
            st.info("ℹ️ Quiz mode requires generated flashcards. Run synthesis in Flashcard mode first.")

    # ----------------------------------------------------------------------- #
    # 4. Counter-Question & Q&A Tab                                           #
    # ----------------------------------------------------------------------- #
    with tab_counter:
        st.markdown("### 💬 Counter-Question & Conceptual Socratic Q&A")
        st.caption(
            "Challenge or probe any synthesized concept or formula. Gemma 4 provides rigorous STEM explanations."
        )

        # Build options from flashcards or formulas
        q_options = []
        q_lookup = {}
        if has_cards:
            for i, c in enumerate(st.session_state["cards"], start=1):
                label = f"Flashcard #{i}: {c['front']}"
                q_options.append(label)
                q_lookup[label] = {"front": c["front"], "back": c["back"]}
        if has_formulas:
            for i, f in enumerate(st.session_state["formulas"], start=1):
                label = f"Formula #{i}: {f['name']} ({f.get('latex_equation', '')})"
                q_options.append(label)
                q_lookup[label] = {
                    "front": f"{f['name']}: {f.get('latex_equation', '')}",
                    "back": f"{f.get('description', '')}. Validity: {f.get('assumptions', '')}",
                }

        if q_options:
            selected_item = st.selectbox(
                "Select Concept or Formula to Probe",
                options=q_options,
                index=0,
            )
            context = q_lookup[selected_item]

            with st.container(border=True):
                st.markdown(f"**Concept:** {context['front']}")
                st.markdown(f"**Answer / Law:** {context['back']}")

            if "counter_history" not in st.session_state:
                st.session_state["counter_history"] = {}

            history_key = f"q_{abs(hash(selected_item)) % 100000}"
            current_history = st.session_state["counter_history"].get(history_key, [])

            if current_history:
                st.markdown("#### Discussion History")
                for item in current_history:
                    with st.chat_message("user"):
                        st.write(item["query"])
                    with st.chat_message("assistant"):
                        st.markdown(item["response"])
                        st.caption(
                            f"<small>Answered by <b>{item.get('model', 'Gemma 4')}</b></small>",
                            unsafe_allow_html=True,
                        )

            user_query = st.text_area(
                "Your Counter-Question or Query",
                placeholder="e.g., Why does this formula assume an ideal gas? What breaks down at relativistic speeds? What is the physical meaning of this constant?",
                key=f"query_input_{history_key}",
                height=100,
            )

            if st.button("🚀 Ask Gemma 4", type="primary", use_container_width=True, key=f"btn_ask_{history_key}"):
                if not user_query.strip():
                    st.warning("Please type a question before asking.")
                else:
                    with st.spinner(f"Asking {preferred_model} to analyze your question…"):
                        try:
                            counter_prompt = build_counter_question_prompt(
                                context["front"],
                                context["back"],
                                user_query.strip(),
                                language,
                            )
                            contents = [types.Content(parts=[types.Part.from_text(text=counter_prompt)])]
                            answer, ans_model = generate_with_fallback(
                                contents, preferred_model=preferred_model
                            )

                            if history_key not in st.session_state["counter_history"]:
                                st.session_state["counter_history"][history_key] = []

                            st.session_state["counter_history"][history_key].append({
                                "query": user_query.strip(),
                                "response": answer,
                                "model": ans_model,
                            })
                            st.rerun()
                        except RuntimeError as err:
                            st.error(f"❌ Error getting response from Gemma 4: {err}")
        else:
            st.info("ℹ️ Run generation first to populate concepts and formulas for Q&A.")

# --------------------------------------------------------------------------- #
# Footer                                                                       #
# --------------------------------------------------------------------------- #

st.divider()
used_model_display = st.session_state.get("used_model", "Gemma 4")
st.caption(f"Powered by {used_model_display} via Google GenAI SDK")
