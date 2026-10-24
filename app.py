"""
app.py – Streamlit UI for "Notes to Flashcards".

All Gemini API interaction lives in generate_with_fallback(); everything else
is pure UI.  Only Gemma 4 models are used – never a Gemini model or any other
AI model.
"""

import os
import io
import time

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

from cards import build_prompt, parse_cards, cards_to_csv, shrink_image

# --------------------------------------------------------------------------- #
# Model list – Gemma 4 only.  First entry is the default preferred model.     #
# --------------------------------------------------------------------------- #
MODELS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]

# --------------------------------------------------------------------------- #
# Retry / fallback configuration                                               #
# --------------------------------------------------------------------------- #

# Per-model retry: how many *extra* attempts after the first
_MODEL_RETRIES = 2          # → up to 3 attempts per model
_RETRY_DELAYS  = [2, 4]     # seconds between attempts (len == _MODEL_RETRIES)

# Error signals that trigger a same-model retry (transient server errors)
_RETRYABLE_SIGNALS   = ("500", "503", "internal", "unavailable")

# Error signals that skip straight to the next model (model-not-found)
_SKIP_MODEL_SIGNALS  = ("404", "not found", "not_found")

# Error signals that stop everything immediately (auth / quota)
_FATAL_SIGNALS       = ("400", "401", "403", "api_key", "invalid key",
                        "permission", "billing")


def _classify(exc: Exception) -> str:
    """
    Return one of 'retry', 'skip', 'fatal', or 'unknown' for an exception.

    Decision order: fatal > skip > retry > unknown.
    """
    msg = str(exc).lower()
    if any(s in msg for s in _FATAL_SIGNALS):
        return "fatal"
    if any(s in msg for s in _SKIP_MODEL_SIGNALS):
        return "skip"
    if any(s in msg for s in _RETRYABLE_SIGNALS):
        return "retry"
    return "unknown"


# --------------------------------------------------------------------------- #
# Core API function                                                            #
# --------------------------------------------------------------------------- #

def generate_with_fallback(
    contents: list,
    preferred_model: str | None = None,
) -> tuple[str, str]:
    """
    Try each model in MODELS (preferred first) and return (response_text, model_name).

    Per-model behaviour:
      - Retryable error (500/503/INTERNAL/UNAVAILABLE): retry up to _MODEL_RETRIES
        more times with _RETRY_DELAYS waits, then fall through to the next model.
      - 404 / model not found: skip to the next model immediately.
      - Fatal error (400/401/403 / bad key): raise RuntimeError immediately.
      - Any other error: fall through to the next model.

    Raises RuntimeError if all models fail.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Export it before running the app."
        )

    client = genai.Client(api_key=api_key)

    # Build the ordered model list: preferred first, then the rest in order
    if preferred_model and preferred_model in MODELS:
        ordered = [preferred_model] + [m for m in MODELS if m != preferred_model]
    else:
        ordered = list(MODELS)

    last_error: Exception | None = None

    for model in ordered:
        for attempt in range(1 + _MODEL_RETRIES):   # 0 = first try, 1..N = retries
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    # No response_mime_type – Gemma may reject JSON mode
                )
                return response.text or "", model

            except Exception as exc:
                last_error = exc
                kind = _classify(exc)

                if kind == "fatal":
                    raise RuntimeError(
                        f"Authentication / quota error – check your API key. "
                        f"Details: {exc}"
                    ) from exc

                if kind == "skip":
                    # Model not found – no point retrying
                    break

                if kind == "retry" and attempt < _MODEL_RETRIES:
                    wait = _RETRY_DELAYS[attempt]
                    # Surface a non-blocking toast so the user knows we're waiting
                    try:
                        st.toast(
                            f"⏳ {model} busy – retrying in {wait} s "
                            f"(attempt {attempt + 1}/{_MODEL_RETRIES})…",
                            icon="🔄",
                        )
                    except Exception:
                        pass  # running outside Streamlit context (tests)
                    time.sleep(wait)
                    continue  # same model, next attempt

                # Unknown error or retries exhausted – move to next model
                break

    # All models exhausted
    raise RuntimeError(
        f"All models failed. Last error: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Page config                                                                  #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Notes to Flashcards",
    page_icon="🃏",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# API key guard                                                                #
# --------------------------------------------------------------------------- #
if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "⚠️ **GEMINI_API_KEY is not set.**\n\n"
        "Please export your Gemini API key before running the app:\n\n"
        "```bash\n"
        "export GEMINI_API_KEY=your_key_here   # macOS / Linux\n"
        "set GEMINI_API_KEY=your_key_here      # Windows CMD\n"
        "$env:GEMINI_API_KEY='your_key_here'   # PowerShell\n"
        "```\n\n"
        "You can get a free key at [aistudio.google.com](https://aistudio.google.com/)."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #

st.title("🃏 Notes to Flashcards")
st.caption(
    "Upload a photo of handwritten notes or a textbook page and Gemma 4 "
    "will turn them into study flashcards."
)
st.divider()

# --------------------------------------------------------------------------- #
# Sidebar – controls                                                           #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Settings")

    preferred_model = st.selectbox(
        "Model",
        options=MODELS,
        index=0,
        help="Preferred Gemma 4 model. The app will fall back to the other if needed.",
    )
    num_cards = st.slider(
        "Number of flashcards",
        min_value=3,
        max_value=15,
        value=8,
        help="How many flashcards to generate from the image.",
    )
    language = st.selectbox(
        "Language",
        options=["English", "Hindi"],
        index=0,
        help="Language for the flashcard text.",
    )
    st.divider()
    # Footer updated after a successful run (see below); show default for now
    st.markdown(
        "<small>Powered by <b>Gemma 4</b> via the Gemini API</small>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Image upload                                                                 #
# --------------------------------------------------------------------------- #

uploaded = st.file_uploader(
    "Upload your notes image",
    type=["png", "jpg", "jpeg"],
    help="Supports PNG, JPG, JPEG. Large images are automatically resized.",
)

if uploaded:
    raw_bytes = uploaded.read()
    img = Image.open(io.BytesIO(raw_bytes))
    st.image(img, caption=uploaded.name, use_container_width=True)
    st.divider()

    # ----------------------------------------------------------------------- #
    # Generate button                                                          #
    # ----------------------------------------------------------------------- #

    if st.button("✨ Make flashcards", type="primary", use_container_width=True):
        # Shrink before sending
        image_bytes, mime_type = shrink_image(raw_bytes)

        prompt = build_prompt(num_cards, language)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        text_part  = types.Part.from_text(text=prompt)
        contents   = [types.Content(parts=[image_part, text_part])]

        with st.spinner("Asking Gemma 4 to read your notes…"):
            try:
                raw_response, used_model = generate_with_fallback(
                    contents, preferred_model=preferred_model
                )
            except RuntimeError as err:
                st.error(f"❌ **Could not reach any model.** {err}")
                st.stop()

        try:
            cards = parse_cards(raw_response)
        except ValueError as err:
            st.error(f"❌ **Could not parse flashcards:** {err}")
            if st.button("🔄 Retry", key="retry_parse"):
                st.rerun()
            st.stop()

        st.session_state["cards"]      = cards
        st.session_state["used_model"] = used_model
        st.session_state["quiz_index"] = 0
        st.session_state["quiz_score"] = 0
        st.session_state["revealed"]   = False

# --------------------------------------------------------------------------- #
# Show flashcards                                                              #
# --------------------------------------------------------------------------- #

if "cards" in st.session_state and st.session_state["cards"]:
    cards: list[dict] = st.session_state["cards"]
    used_model: str   = st.session_state.get("used_model", MODELS[0])

    st.success(f"Generated {len(cards)} flashcard(s).")

    tab_browse, tab_quiz = st.tabs(["📖 Browse Cards", "🎯 Quiz Mode"])

    # ----------------------------------------------------------------------- #
    # Browse tab                                                               #
    # ----------------------------------------------------------------------- #
    with tab_browse:
        for i, card in enumerate(cards, start=1):
            with st.expander(f"Card {i}: {card['front']}"):
                st.markdown(card["back"])

        st.divider()

        csv_text = cards_to_csv(cards)
        st.download_button(
            label="⬇️ Download CSV for Anki",
            data=csv_text.encode("utf-8"),
            file_name="flashcards.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ----------------------------------------------------------------------- #
    # Quiz tab                                                                 #
    # ----------------------------------------------------------------------- #
    with tab_quiz:
        idx:      int  = st.session_state.get("quiz_index", 0)
        score:    int  = st.session_state.get("quiz_score", 0)
        revealed: bool = st.session_state.get("revealed", False)

        if idx >= len(cards):
            st.balloons()
            st.subheader("🎉 Quiz complete!")
            st.metric("Score", f"{score} / {len(cards)}")
            pct = int(score / len(cards) * 100)
            st.progress(pct / 100)
            if pct == 100:
                st.success("Perfect score! Great work! 🏆")
            elif pct >= 70:
                st.info("Nice job! Keep reviewing the ones you missed.")
            else:
                st.warning("Keep studying – you'll get there! 💪")

            if st.button("🔄 Restart Quiz", use_container_width=True):
                st.session_state["quiz_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["revealed"]   = False
                st.rerun()
        else:
            card = cards[idx]
            st.markdown(f"**Card {idx + 1} of {len(cards)}**")
            st.progress(idx / len(cards))
            st.markdown("---")
            st.subheader("❓ " + card["front"])
            st.markdown("---")

            if not revealed:
                if st.button("👁 Reveal answer", use_container_width=True):
                    st.session_state["revealed"] = True
                    st.rerun()
            else:
                st.info(card["back"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ I knew it", use_container_width=True):
                        st.session_state["quiz_score"] += 1
                        st.session_state["quiz_index"] += 1
                        st.session_state["revealed"]   = False
                        st.rerun()
                with col2:
                    if st.button("🔁 Review again", use_container_width=True):
                        cards.append(cards[idx])
                        st.session_state["cards"]      = cards
                        st.session_state["quiz_index"] += 1
                        st.session_state["revealed"]   = False
                        st.rerun()

# --------------------------------------------------------------------------- #
# Footer                                                                       #
# --------------------------------------------------------------------------- #

st.divider()
used_model_display = st.session_state.get("used_model", "Gemma 4")
st.caption(f"Powered by {used_model_display} via the Gemini API")
st.markdown(
    "<center><small>Built for Hacktoberfest Hack Day · OpenSTEM Track</small></center>",
    unsafe_allow_html=True,
)
