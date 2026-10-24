"""
app.py – Streamlit UI for "Open-Source STEM Knowledge & Formula Synthesizer".

Designed for the Hacktoberfest OpenSTEM track.
Strictly configured to run on localhost (streamlit run app.py).
No cloud configurations, no Dockerfiles, no st.secrets.
Uses local .env file or sidebar password input for GEMINI_API_KEY.
Powered exclusively by Gemma 4 models via the Google GenAI SDK.
"""

import io
import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load local environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------- #
# Supported Models – Gemma 4 only. First entry is preferred default.          #
# --------------------------------------------------------------------------- #
MODELS = [
    "gemma-4-26b-a4b-it",  # Faster default: 4B active parameters (MoE)
    "gemma-4-31b-it",      # High-capacity fallback: 31B dense parameters
]

from cards import (
    build_concept_graph_prompt,
    build_counter_question_prompt,
    build_elaboration_prompt,
    build_formula_synthesizer_prompt,
    build_prompt,
    cards_to_csv,
    clean_dot_code,
    fetch_arxiv_stem_text,
    fetch_wikipedia_stem_text,
    get_pdf_page_count,
    parse_cards,
    parse_formulas,
    pdf_page_to_image,
    shrink_image,
)

# --------------------------------------------------------------------------- #
# Retry / fallback configuration                                               #
# --------------------------------------------------------------------------- #

_MODEL_RETRIES = 5  # Auto change model if one takes more than 5 retries
_RETRY_DELAYS = [2, 4, 8, 16, 32]  # seconds between attempts

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
    """Read API key from local environment variable."""
    return str(os.environ.get("GEMINI_API_KEY", "")).strip()


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
            "GEMINI_API_KEY is not set. Enter it in the sidebar or save it in your local .env file."
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
    page_title="OpenSTEM Knowledge & Formula Synthesizer",
    page_icon="⚛️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Sidebar – Configuration & Local API Key                                      #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=_get_api_key(),
        placeholder="Enter key or load from .env",
        help="Strictly localhost execution. Placed in local .env or session.",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()

    if not _get_api_key():
        st.warning("⚠️ Add your API key above or in `.env` to start.")

    st.divider()

    st.subheader("🎯 Synthesis Controls")
    synthesis_mode = st.radio(
        "Synthesis Mode",
        options=["Standard STEM Flashcards", "Formula & Derivation Extractor"],
        index=0,
        help="Choose between structured flashcards or detailed LaTeX formula extraction.",
    )

    preferred_model = st.selectbox(
        "Preferred Gemma 4 Model",
        options=MODELS,
        index=0,
        help="gemma-4-26b-a4b-it is fast MoE; gemma-4-31b-it is high-capacity fallback.",
    )

    num_cards = st.slider(
        "Number of Cards / Formulas",
        min_value=3,
        max_value=15,
        value=6,
        help="Target count for flashcards or extracted formulas.",
    )

    language = st.selectbox(
        "Synthesis Language",
        options=["English", "Hindi"],
        index=0,
        help="Language for explanations and card text.",
    )

    generate_graph = st.checkbox(
        "Generate Concept Graph (Graphviz)",
        value=True,
        help="Generate a hierarchical directed Graphviz concept map of laws and concepts.",
    )

    st.divider()
    st.markdown(
        "<small>🔬 <b>OpenSTEM Track</b> · Built for Hacktoberfest<br>"
        "Runs 100% locally on localhost.<br>"
        "Powered by <b>Gemma 4</b> via Google GenAI SDK</small>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Main Header                                                                  #
# --------------------------------------------------------------------------- #

st.title("⚛️ Open-Source STEM Knowledge & Formula Synthesizer")
st.caption(
    "Synthesize scientific notes, textbook pages, and verified open-source literature "
    "(Wikipedia & arXiv) into LaTeX formula sheets, interactive concept graphs, and Anki flashcards."
)
st.divider()

# --------------------------------------------------------------------------- #
# STEM Input & Open Data Grounding                                             #
# --------------------------------------------------------------------------- #

st.subheader("📥 1. Select STEM Input & Open Data Grounding")

input_tab_open, input_tab_text, input_tab_upload = st.tabs(
    [
        "🌐 Open Data Grounding (Wikipedia & arXiv)",
        "📝 Paste STEM Notes / Text",
        "📁 Upload Notes / PDF",
    ]
)

active_text = ""
active_image_bytes = None
active_mime_type = None

# Tab 1: Open Data Grounding
with input_tab_open:
    col_prov, col_q = st.columns([1, 3])
    with col_prov:
        open_provider = st.selectbox(
            "Source Provider",
            options=["Wikipedia", "arXiv"],
            help="Fetch verified STEM articles from Wikipedia or peer-reviewed abstracts from arXiv.",
        )
    with col_q:
        default_q = "Carnot cycle" if open_provider == "Wikipedia" else "Quantum Hall Effect"
        query_val = st.text_input(
            "Topic or Search Query",
            value=default_q,
            placeholder="e.g., Maxwell's equations, Photosynthesis, Navier-Stokes...",
        )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        fetch_clicked = st.button("🔍 Fetch Open STEM Text", use_container_width=True)

    if fetch_clicked:
        if not query_val.strip():
            st.error("Please provide a search topic.")
        else:
            with st.spinner(f"Querying verified open-source data from {open_provider}…"):
                try:
                    if open_provider == "Wikipedia":
                        wiki_res = fetch_wikipedia_stem_text(query_val.strip())
                        st.session_state["fetched_title"] = wiki_res["title"]
                        st.session_state["fetched_url"] = wiki_res["url"]
                        st.session_state["fetched_content"] = (
                            f"=== Topic: {wiki_res['title']} ===\n\n"
                            f"Summary:\n{wiki_res['summary']}\n\n"
                            f"Content:\n{wiki_res['content']}"
                        )
                    else:
                        arxiv_res = fetch_arxiv_stem_text(query_val.strip(), max_results=2)
                        papers_text = []
                        for idx, p in enumerate(arxiv_res, start=1):
                            papers_text.append(
                                f"=== Paper {idx}: {p['title']} ===\n"
                                f"Authors: {p['authors']} ({p['published']})\n"
                                f"URL: {p['url']}\n\n"
                                f"Abstract:\n{p['summary']}"
                            )
                        st.session_state["fetched_title"] = f"arXiv: {query_val.strip()}"
                        st.session_state["fetched_url"] = arxiv_res[0]["url"]
                        st.session_state["fetched_content"] = "\n\n".join(papers_text)
                    st.success(f"✅ Fetched verified text from {open_provider}!")
                except Exception as exc:
                    st.error(f"❌ Failed to fetch from {open_provider}: {exc}")

    if "fetched_content" in st.session_state:
        st.markdown(
            f"**Source Document:** [{st.session_state.get('fetched_title', 'Open Data')}]"
            f"({st.session_state.get('fetched_url', '#')})"
        )
        active_text = st.text_area(
            "Verified Scientific Grounding Text (Editable)",
            value=st.session_state["fetched_content"],
            height=200,
            key="open_text_area",
        )

# Tab 2: Paste Raw STEM Notes
with input_tab_text:
    pasted_sample = (
        "Thermodynamics First & Second Law:\n"
        "Delta U = Q - W, where Delta U is change in internal energy (Joules), "
        "Q is heat added to the system (Joules), and W is work done by the system (Joules).\n"
        "For an ideal gas, work during reversible isothermal expansion is W = n * R * T * ln(V_final / V_initial).\n"
        "Carnot engine efficiency: eta = 1 - (T_cold / T_hot), valid only for reversible Carnot cycles operating "
        "between two thermal reservoirs."
    )
    raw_pasted = st.text_area(
        "Paste STEM Notes, Textbook Excerpts, or Formulas",
        value="" if "fetched_content" in st.session_state else pasted_sample,
        height=200,
        placeholder="Type or paste your STEM formulas, physics derivations, or chemistry reaction notes...",
        key="raw_notes_area",
    )
    if not active_text and raw_pasted.strip():
        active_text = raw_pasted.strip()

# Tab 3: Upload Image / PDF
with input_tab_upload:
    uploaded = st.file_uploader(
        "Upload handwritten STEM notes or textbook PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Supports PNG, JPG, JPEG, and PDF documents. Hand-written notes are recognized.",
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
                    f"Select page to study (1 to {total_pages})",
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

st.divider()

# --------------------------------------------------------------------------- #
# Synthesize Action Button                                                    #
# --------------------------------------------------------------------------- #

st.subheader("⚡ 2. Run Gemma 4 Synthesis")

btn_label = f"✨ Synthesize STEM Knowledge ({synthesis_mode})"
if st.button(btn_label, type="primary", use_container_width=True):
    if not _get_api_key():
        st.error("❌ Please provide a Gemini API Key in the sidebar.")
        st.stop()

    # Determine input content parts
    contents_list = []
    if active_image_bytes:
        s_bytes, s_mime = shrink_image(active_image_bytes, max_side=1200, quality=80)
        img_part = types.Part.from_bytes(data=s_bytes, mime_type=s_mime)
        contents_list.append(img_part)

    if active_text.strip():
        text_prefix = "The following is the STEM content to analyze:\n\n" + active_text.strip()
        contents_list.append(types.Part.from_text(text=text_prefix))

    if not contents_list:
        st.error("❌ Please provide an input: fetch open data, paste notes, or upload an image/PDF.")
        st.stop()

    # Run Synthesis based on selected mode
    with st.spinner(f"Analyzing STEM concepts using {preferred_model}…"):
        try:
            if synthesis_mode == "Formula & Derivation Extractor":
                formula_prompt = build_formula_synthesizer_prompt(language)
                call_contents = [
                    types.Content(parts=contents_list + [types.Part.from_text(text=formula_prompt)])
                ]
                raw_formula_resp, used_m = generate_with_fallback(
                    call_contents, preferred_model=preferred_model
                )
                formulas = parse_formulas(raw_formula_resp)
                st.session_state["formulas"] = formulas
                st.session_state["used_model"] = used_m
            else:
                card_prompt = build_prompt(num_cards, language)
                call_contents = [
                    types.Content(parts=contents_list + [types.Part.from_text(text=card_prompt)])
                ]
                raw_card_resp, used_m = generate_with_fallback(
                    call_contents, preferred_model=preferred_model
                )
                cards = parse_cards(raw_card_resp)
                st.session_state["cards"] = cards
                st.session_state["used_model"] = used_m
                st.session_state["quiz_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["revealed"] = False
        except Exception as exc:
            st.error(f"❌ Synthesis failed: {exc}")
            st.stop()

    # Generate Concept Graph if checked
    if generate_graph:
        with st.spinner("Generating hierarchical Graphviz concept map…"):
            try:
                graph_prompt = build_concept_graph_prompt(language)
                graph_contents = [
                    types.Content(parts=contents_list + [types.Part.from_text(text=graph_prompt)])
                ]
                raw_graph_resp, _ = generate_with_fallback(
                    graph_contents, preferred_model=preferred_model
                )
                dot_code = clean_dot_code(raw_graph_resp)
                st.session_state["dot_code"] = dot_code
            except Exception as exc:
                st.warning(f"⚠️ Concept map generation had an issue: {exc}")

    st.success("🎉 Synthesis complete!")

# --------------------------------------------------------------------------- #
# Output Section: Tabs                                                         #
# --------------------------------------------------------------------------- #

has_formulas = "formulas" in st.session_state and st.session_state["formulas"]
has_cards = "cards" in st.session_state and st.session_state["cards"]
has_graph = "dot_code" in st.session_state and st.session_state["dot_code"]

if has_formulas or has_cards or has_graph:
    st.divider()
    st.subheader("📊 Synthesized Knowledge Artifacts")

    tab_formulas, tab_graph, tab_browse, tab_quiz, tab_counter = st.tabs(
        [
            "📐 Formula Synthesizer",
            "🗺️ Concept Graph",
            "📖 Flashcards & Anki",
            "🎯 Quiz Mode",
            "💬 Counter Question & Q&A",
        ]
    )

    # ----------------------------------------------------------------------- #
    # TAB 1: Formula Synthesizer                                              #
    # ----------------------------------------------------------------------- #
    with tab_formulas:
        if has_formulas:
            formulas = st.session_state["formulas"]
            st.markdown(f"### 📐 Synthesized Formula Cards ({len(formulas)} Extracted)")
            st.caption(
                "Governing equations, SI variable definitions, and validity assumptions formatted strictly in LaTeX."
            )

            for i, f in enumerate(formulas, start=1):
                with st.container(border=True):
                    col_name, col_id = st.columns([4, 1])
                    with col_name:
                        st.markdown(f"#### {i}. {f['name']}")
                    with col_id:
                        st.caption(f"`Law #{i}`")

                    # LaTeX Equation Display
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

                    # Core Assumptions & Validity Limits
                    assumptions = f.get("assumptions")
                    if assumptions:
                        st.info(f"**⚠️ Core Conditions & Limits of Validity:** {assumptions}")
        else:
            st.info(
                "ℹ️ No formulas synthesized yet. Select **'Formula & Derivation Extractor'** "
                "in the sidebar and click **Synthesize STEM Knowledge**."
            )

    # ----------------------------------------------------------------------- #
    # TAB 2: Concept Graph (Graphviz)                                          #
    # ----------------------------------------------------------------------- #
    with tab_graph:
        if has_graph:
            st.markdown("### 🗺️ Hierarchical STEM Concept Graph")
            st.caption(
                "Visualizes connections between overarching STEM domains, principles, and governing equations."
            )
            dot_code = st.session_state["dot_code"]
            try:
                st.graphviz_chart(dot_code, use_container_width=True)
            except Exception as g_err:
                st.error(f"Could not render Graphviz chart: {g_err}")

            with st.expander("🔍 View Raw Graphviz DOT Source Code"):
                st.code(dot_code, language="dot")
        else:
            st.info(
                "ℹ️ Concept graph not generated yet. Enable 'Generate Concept Graph' in sidebar and run synthesis."
            )

    # ----------------------------------------------------------------------- #
    # TAB 3: Flashcards & Anki CSV                                            #
    # ----------------------------------------------------------------------- #
    with tab_browse:
        if has_cards:
            cards = st.session_state["cards"]
            st.markdown(f"### 📖 STEM Study Flashcards ({len(cards)} Cards)")
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
            st.info(
                "ℹ️ Flashcards not generated yet. Select **'Standard STEM Flashcards'** in sidebar and run synthesis."
            )

    # ----------------------------------------------------------------------- #
    # TAB 4: Interactive Quiz Mode                                            #
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
    # TAB 5: Counter-Question & Q&A                                           #
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
            st.info("ℹ️ Run synthesis first to populate concepts and formulas for Q&A.")

# --------------------------------------------------------------------------- #
# Footer                                                                       #
# --------------------------------------------------------------------------- #

st.divider()
used_model_display = st.session_state.get("used_model", "Gemma 4")
st.caption(f"Powered by {used_model_display} via Google GenAI SDK · Localhost Edition")
st.markdown(
    "<center><small>OpenSTEM Knowledge & Formula Synthesizer · Built for Hacktoberfest Hack Day</small></center>",
    unsafe_allow_html=True,
)
