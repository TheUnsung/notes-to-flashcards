"""
cards.py – pure functions for Notes-to-Flashcards.
No Streamlit, no network calls here.
"""

import csv
import io
import json
import re
from typing import Any

from PIL import Image

# --------------------------------------------------------------------------- #
# Prompt builder                                                               #
# --------------------------------------------------------------------------- #

def build_prompt(num_cards: int, language: str, elaborate: bool = False) -> str:
    """Return the system-style instruction sent alongside the image."""
    lang_note = (
        "Respond in Hindi (Devanagari script)."
        if language == "Hindi"
        else "Respond in English."
    )
    if elaborate:
        back_rule = (
            "4. 'back' is an elaborated explanation (3 to 4 sentences): clearly state "
            "the core answer, explain the underlying principle or mechanism in depth, "
            "and ALWAYS include relevant mathematical equations, scientific formulae, "
            "variable definitions, and exact units wherever applicable."
        )
    else:
        back_rule = (
            "4. 'back' is the answer in at most 2 sentences. ALWAYS include relevant "
            "mathematical equations, scientific formulae, symbols, and exact units wherever applicable, "
            "preserving them exactly as they appear in the image."
        )

    return (
        f"You are an expert STEM tutor helping a student memorise key ideas "
        f"from their notes or textbook page.\n\n"
        f"Rules:\n"
        f"1. Generate exactly {num_cards} flashcards from the content visible "
        f"in the image.\n"
        f"2. Each flashcard covers ONE distinct idea, fact, formula, or term.\n"
        f"3. 'front' is a concise question (one sentence, ≤15 words).\n"
        f"{back_rule}\n"
        f"5. Include relevant equations and formulae wherever applicable.\n"
        f"6. Use ONLY information visible in the image – do not add outside "
        f"knowledge.\n"
        f"7. {lang_note}\n"
        f"8. Return ONLY a JSON array of objects, each with exactly the keys "
        f"\"front\" and \"back\". No markdown fences, no extra keys, no prose.\n\n"
        f"Example format:\n"
        f'[{{"front": "What is Newton\'s second law?", '
        f'"back": "F = ma, where F is force in Newtons, m is mass in kg, '
        f'and a is acceleration in m/s²."}}]'
    )


def build_elaboration_prompt(front: str, back: str, language: str) -> str:
    """Return a prompt asking Gemma 4 to deeply explain a specific flashcard concept."""
    lang_note = (
        "Explain in Hindi (Devanagari script)."
        if language == "Hindi"
        else "Explain in English."
    )
    return (
        f"You are an expert STEM tutor. A student studying from their notes wants a "
        f"clear, detailed, and elaborated explanation of this flashcard:\n\n"
        f"Question: {front}\n"
        f"Brief Answer: {back}\n\n"
        f"Please provide an elaborated explanation structured as follows:\n"
        f"- **Core Concept & Intuition**: Explain the fundamental principle in simple terms.\n"
        f"- **Step-by-step Breakdown**: How does it work? Detail any equations, variables, or units.\n"
        f"- **Practical Application / Example**: A real-world example or common problem scenario.\n\n"
        f"Keep the tone encouraging and academic. {lang_note}"
    )


def build_counter_question_prompt(
    card_front: str,
    card_back: str,
    user_query: str,
    language: str,
) -> str:
    """
    Build a prompt for Gemma 4 to answer a student's counter-question or query
    about a specific flashcard.
    """
    lang_note = (
        "Respond in Hindi (Devanagari script)."
        if language == "Hindi"
        else "Respond in English."
    )
    return (
        f"You are an encouraging expert STEM tutor. A student has generated flashcards "
        f"from their notes and has a follow-up query or counter-question about this specific card:\n\n"
        f"--- Flashcard Context ---\n"
        f"Question: {card_front}\n"
        f"Current Answer: {card_back}\n\n"
        f"--- Student's Query / Counter-Question ---\n"
        f"{user_query}\n\n"
        f"Instructions:\n"
        f"1. Directly and clearly answer the student's question or resolve their confusion.\n"
        f"2. Explain the scientific, physical, or mathematical reasoning step-by-step.\n"
        f"3. If their question challenges the premise of the card, explain whether their reasoning "
        f"is valid, under what conditions/edge cases it applies, or why the standard formula/answer holds.\n"
        f"4. Keep all formulas, symbols, and units exact.\n"
        f"5. Maintain an engaging, educational tutor tone. {lang_note}"
    )


# --------------------------------------------------------------------------- #
# Image shrinker & PDF converter                                               #
# --------------------------------------------------------------------------- #

def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the total number of pages in the given PDF bytes."""
    import pymupdf
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return len(doc)
    except Exception as exc:
        raise ValueError(f"Could not read PDF document: {exc}") from exc


def pdf_page_to_image(
    pdf_bytes: bytes,
    page_number: int = 0,
    dpi_scale: float = 2.0,
) -> bytes:
    """
    Render a single page of a PDF to PNG image bytes for Gemma 4 consumption.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        page_number: 0-indexed page number to render.
        dpi_scale: Scale factor for rendering resolution (default 2.0 ≈ 144-200 DPI).

    Returns:
        PNG image bytes.
    """
    import pymupdf
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF file: {exc}") from exc

    if len(doc) == 0:
        raise ValueError("PDF document has no pages.")

    if page_number < 0 or page_number >= len(doc):
        raise ValueError(
            f"Page number {page_number + 1} is out of range. "
            f"PDF has {len(doc)} page(s)."
        )

    page = doc[page_number]
    matrix = pymupdf.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes("png")


def shrink_image(image_bytes: bytes, max_side: int = 1600, quality: int = 85) -> tuple[bytes, str]:
    """
    Resize image so its longest side is at most *max_side* pixels, convert to
    RGB, automatically correct EXIF orientation, and re-encode as JPEG.

    Returns (jpeg_bytes, "image/jpeg").
    """
    from PIL import ImageOps

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


# --------------------------------------------------------------------------- #
# Response parser                                                              #
# --------------------------------------------------------------------------- #

# Matches optional language tag after the opening fence (```json or ``` alone)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)

# Matches unescaped backslashes in JSON (e.g. LaTeX formulas like \pm, \frac, \alpha)
_INVALID_ESCAPE_RE = re.compile(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})')


def _clean_json_escapes(s: str) -> str:
    r"""Escape unescaped backslashes (such as LaTeX \pm, \frac, etc.) in JSON strings."""
    return _INVALID_ESCAPE_RE.sub(r'\\\\', s)


def parse_cards(raw: str) -> list[dict[str, str]]:
    r"""
    Parse the model's raw text response into a list of flashcard dicts.

    Strips Markdown code fences (```json ... ``` or ``` ... ```), handles
    unescaped LaTeX backslashes (e.g. \pm, \approx, \frac), and surrounding
    whitespace before attempting json.loads.

    Returns a list like [{"front": "...", "back": "..."}, ...].

    Raises:
        ValueError – with a human-readable message on any validation failure.
    """
    if not raw or not raw.strip():
        raise ValueError("The model returned an empty response.")

    text = raw.strip()

    # Strip markdown code fences (handles ```json\n...\n``` and ```\n...\n```)
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        # Fallback line-by-line strip for malformed fences
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()

    # Multi-stage parsing: handles raw JSON, bracketed JSON with conversational wrapper,
    # and unescaped math/LaTeX backslashes.
    data: Any = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        candidate = text[start : end + 1] if (start != -1 and end != -1 and end > start) else text

        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            try:
                fixed = _clean_json_escapes(candidate)
                data = json.loads(fixed, strict=False)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse the model response as JSON.\n"
                    f"Details: {exc}\n"
                    f"Raw response (first 300 chars): {raw[:300]}"
                ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list but got {type(data).__name__}. "
            f"Raw response (first 300 chars): {raw[:300]}"
        )

    if len(data) == 0:
        raise ValueError(
            "The model returned an empty list of flashcards. "
            "Try uploading a clearer image or increasing the card count."
        )

    cards: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Card {i + 1} is not an object (got {type(item).__name__})."
            )
        missing = [k for k in ("front", "back") if k not in item]
        if missing:
            raise ValueError(
                f"Card {i + 1} is missing required key(s): {missing}."
            )
        cards.append(
            {"front": str(item["front"]).strip(), "back": str(item["back"]).strip()}
        )

    return cards


# --------------------------------------------------------------------------- #
# CSV export                                                                   #
# --------------------------------------------------------------------------- #

def cards_to_csv(cards: list[dict[str, str]]) -> str:
    """
    Serialise flashcards to Anki-compatible CSV text.

    Two columns (front, back), no header row.  Fields that contain commas,
    double-quotes, or newlines are quoted per RFC 4180.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    for card in cards:
        writer.writerow([card["front"], card["back"]])
    return buf.getvalue()
