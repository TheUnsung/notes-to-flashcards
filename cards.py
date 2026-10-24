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

def build_prompt(num_cards: int, language: str) -> str:
    """Return the system-style instruction sent alongside the image."""
    lang_note = (
        "Respond in Hindi (Devanagari script)."
        if language == "Hindi"
        else "Respond in English."
    )
    return (
        f"You are an expert STEM tutor helping a student memorise key ideas "
        f"from their notes or textbook page.\n\n"
        f"Rules:\n"
        f"1. Generate exactly {num_cards} flashcards from the content visible "
        f"in the image.\n"
        f"2. Each flashcard covers ONE distinct idea, fact, formula, or term.\n"
        f"3. 'front' is a concise question (one sentence, ≤15 words).\n"
        f"4. 'back' is the answer in at most 2 sentences; preserve formulas, "
        f"symbols, and units exactly as they appear in the image.\n"
        f"5. Use ONLY information visible in the image – do not add outside "
        f"knowledge.\n"
        f"6. {lang_note}\n"
        f"7. Return ONLY a JSON array of objects, each with exactly the keys "
        f"\"front\" and \"back\". No markdown fences, no extra keys, no prose.\n\n"
        f"Example format:\n"
        f'[{{"front": "What is Newton\'s second law?", '
        f'"back": "F = ma, where F is force in Newtons, m is mass in kg, '
        f'and a is acceleration in m/s²."}}]'
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
    RGB, and re-encode as JPEG.

    Returns (jpeg_bytes, "image/jpeg").
    """
    img = Image.open(io.BytesIO(image_bytes))
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


def parse_cards(raw: str) -> list[dict[str, str]]:
    """
    Parse the model's raw text response into a list of flashcard dicts.

    Strips Markdown code fences (```json ... ``` or ``` ... ```) and surrounding
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

    try:
        data: Any = json.loads(text)
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
