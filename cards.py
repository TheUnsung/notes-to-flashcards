"""
cards.py – Core engine for the Open-Source STEM Knowledge & Formula Synthesizer.
Pure functions for prompt building, response parsing, Open Data fetching,
and image/PDF preprocessing. No Streamlit or network UI code here.
"""

import csv
import io
import json
import re
from typing import Any

from PIL import Image

# --------------------------------------------------------------------------- #
# Supported Models – Gemma 4 only                                             #
# --------------------------------------------------------------------------- #
MODELS = [
    "gemma-4-26b-a4b-it",  # Faster default: 4B active parameters (MoE)
    "gemma-4-31b-it",      # High-capacity fallback: 31B dense parameters
]

# --------------------------------------------------------------------------- #
# Regex helpers                                                                #
# --------------------------------------------------------------------------- #

# Matches optional language tag after the opening fence (```json or ``` alone)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)

# Matches unescaped backslashes in JSON (e.g. LaTeX formulas like \pm, \alpha)
_INVALID_ESCAPE_RE = re.compile(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})')
# Matches backslashes for LaTeX commands that start with b, f, n, r, t followed by letters (e.g. \frac, \beta, \text)
_LATEX_CMD_ESCAPE_RE = re.compile(r'\\(?=[bfnrt][a-zA-Z])')


def _clean_json_escapes(s: str) -> str:
    r"""Escape unescaped backslashes (such as LaTeX \pm, \frac, \beta, etc.) in JSON strings."""
    s = _LATEX_CMD_ESCAPE_RE.sub(r'\\\\', s)
    return _INVALID_ESCAPE_RE.sub(r'\\\\', s)


# --------------------------------------------------------------------------- #
# Open Data Grounding (Wikipedia & arXiv)                                      #
# --------------------------------------------------------------------------- #

def fetch_wikipedia_stem_text(topic: str, max_chars: int = 3500) -> dict[str, str]:
    """
    Fetch verified STEM article summary and text from Wikipedia.

    Returns dict with keys: 'title', 'url', 'summary', 'content'.
    """
    import wikipedia

    if not topic or not topic.strip():
        raise ValueError("Please provide a search topic.")

    clean_topic = topic.strip()
    wikipedia.set_user_agent("OpenSTEM-Knowledge-Synthesizer/1.0 (contact@openstem.org)")

    try:
        search_results = wikipedia.search(clean_topic, results=5)
        target = search_results[0] if search_results else clean_topic
        page = wikipedia.page(target, auto_suggest=False)
        summary = page.summary
        text = page.content[:max_chars] if hasattr(page, "content") and page.content else summary
        return {
            "title": page.title,
            "url": page.url,
            "summary": summary,
            "content": text,
        }
    except wikipedia.DisambiguationError as e:
        opt = e.options[0] if e.options else clean_topic
        page = wikipedia.page(opt, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "summary": page.summary,
            "content": page.content[:max_chars] if hasattr(page, "content") and page.content else page.summary,
        }
    except Exception as exc:
        raise RuntimeError(f"Wikipedia lookup for '{clean_topic}' failed: {exc}") from exc


def fetch_arxiv_stem_text(query: str, max_results: int = 2) -> list[dict[str, str]]:
    """
    Fetch verified peer-reviewed scientific paper abstracts from arXiv.

    Returns list of dicts with keys: 'title', 'authors', 'published', 'summary', 'url'.
    """
    import arxiv

    if not query or not query.strip():
        raise ValueError("Please provide a search query.")

    clean_query = query.strip()
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=clean_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
            results.append({
                "title": paper.title,
                "authors": authors,
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
                "summary": paper.summary.replace("\n", " ").strip(),
                "url": paper.entry_id,
            })
        if not results:
            raise RuntimeError(f"No arXiv papers found for '{clean_query}'.")
        return results
    except Exception as exc:
        raise RuntimeError(f"arXiv search failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Prompt Builders                                                              #
# --------------------------------------------------------------------------- #

def build_prompt(num_cards: int, language: str, elaborate: bool = False) -> str:
    """Return prompt for generating STEM flashcards with strict LaTeX formatting."""
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
            "preserving them exactly as they appear in the content."
        )

    return (
        f"You are an expert STEM Knowledge Synthesizer and tutor.\n\n"
        f"Rules:\n"
        f"1. Generate exactly {num_cards} flashcards from the provided STEM content.\n"
        f"2. Each flashcard covers ONE distinct idea, fact, formula, or term.\n"
        f"3. 'front' is a concise question (one sentence, ≤15 words).\n"
        f"{back_rule}\n"
        f"5. LaTeX Formatting: Format ALL mathematical equations, physics laws, and chemical reactions "
        f"using valid LaTeX: $inline$ for inline formulas and $$block$$ for standalone equations.\n"
        f"6. Use ONLY information visible in the input content – do not add outside knowledge.\n"
        f"7. {lang_note}\n"
        f"8. Return ONLY a JSON array of objects, each with exactly the keys "
        f"\"front\" and \"back\". No markdown fences, no extra keys, no prose.\n\n"
        f"Example format:\n"
        f'[{{"front": "What is Newton\'s second law of motion?", '
        f'"back": "The net force is given by $F = m \\cdot a$, where $F$ is force in Newtons ($\\text{{N}}$), '
        f'$m$ is mass in $\\text{{kg}}$, and $a$ is acceleration in $\\text{{m/s}}^2$."}}]'
    )


def build_formula_synthesizer_prompt(language: str) -> str:
    """Return prompt to extract and synthesize mathematical/scientific formulas and laws."""
    lang_note = (
        "Respond in Hindi (Devanagari script) for descriptive text."
        if language == "Hindi"
        else "Respond in English."
    )
    return (
        f"You are an expert STEM Knowledge & Formula Synthesizer. Analyze the provided STEM content "
        f"(notes, textbook page, text, or research paper) and extract all governing mathematical equations, "
        f"physical laws, chemical formulas, or scientific relationships.\n\n"
        f"Formatting rules:\n"
        f"1. Format every equation strictly in valid LaTeX (do not include enclosing dollar signs in the json value).\n"
        f"2. For each formula, provide:\n"
        f"   - 'name': Name of the law, equation, theorem, or formula.\n"
        f"   - 'latex_equation': Clean LaTeX representation (e.g. 'F = m \\cdot a' or '\\lambda = \\frac{{D \\beta}}{{d}}').\n"
        f"   - 'description': Brief 1-2 sentence explanation of what the formula computes or represents.\n"
        f"   - 'variables': List of objects with 'symbol', 'definition', and 'unit' (standard SI unit or dimensionless).\n"
        f"   - 'assumptions': Crucial conditions, assumptions, or limits of validity (e.g., 'Valid only for constant acceleration', 'Non-relativistic speeds v << c', 'Ideal gas behavior').\n"
        f"3. {lang_note}\n"
        f"4. Return ONLY a JSON array of formula objects. No conversational text, no markdown fences.\n\n"
        f"Example JSON structure:\n"
        f'[\n'
        f'  {{\n'
        f'    "name": "Newton\'s Second Law",\n'
        f'    "latex_equation": "F = m \\cdot a",\n'
        f'    "description": "Relates the net force acting on an object to its mass and resulting acceleration.",\n'
        f'    "variables": [\n'
        f'      {{"symbol": "F", "definition": "Net Force", "unit": "N (kg*m/s^2)"}},\n'
        f'      {{"symbol": "m", "definition": "Mass of the object", "unit": "kg"}},\n'
        f'      {{"symbol": "a", "definition": "Acceleration", "unit": "m/s^2"}}\n'
        f'    ],\n'
        f'    "assumptions": "Valid for constant mass in an inertial reference frame at non-relativistic velocities."\n'
        f'  }}\n'
        f']'
    )


def build_concept_graph_prompt(language: str) -> str:
    """Return prompt to generate a Graphviz DOT hierarchical concept map."""
    return (
        "You are an expert STEM Knowledge Architect. Analyze the provided STEM content "
        "and generate a hierarchical concept map as a valid Graphviz DOT directed graph.\n\n"
        "Requirements:\n"
        "1. Return ONLY the valid Graphviz DOT string starting with 'digraph STEM_Map {' and ending with '}'.\n"
        "2. Do NOT include markdown code fences (no ```dot or ```).\n"
        "3. Structure the graph hierarchically from top to bottom (rankdir=TB).\n"
        "4. Include:\n"
        "   - Main STEM Field / Overarching Topic (Root node)\n"
        "   - Sub-concepts and principles (Intermediate nodes)\n"
        "   - Governing Equations and Laws (Leaf nodes)\n"
        "5. Style nodes with clean modern aesthetics: shape=box, style=\"rounded,filled\", "
        "fillcolor=\"#EBF3FB\", color=\"#2B5B84\", fontname=\"Helvetica\", fontsize=11.\n"
        "6. Connect nodes with descriptive edge labels where helpful (e.g. [label=\"governed by\"]).\n"
        "7. Ensure node names with spaces are properly double-quoted.\n\n"
        "Example output:\n"
        "digraph STEM_Map {\n"
        "    rankdir=TB;\n"
        "    node [shape=box, style=\"rounded,filled\", fillcolor=\"#EBF3FB\", color=\"#2B5B84\", fontname=\"Helvetica\"];\n"
        "    edge [color=\"#555555\", fontname=\"Helvetica\", fontsize=10];\n"
        "    \"Classical Mechanics\" -> \"Newton's Laws\" [label=\"fundamentals\"];\n"
        "    \"Newton's Laws\" -> \"F = ma\" [label=\"governing eq\"];\n"
        "}\n"
    )


def build_counter_question_prompt(
    card_front: str,
    card_back: str,
    user_query: str,
    language: str,
) -> str:
    """
    Build a prompt for Gemma 4 to answer a student's counter-question or query
    about a specific flashcard or formula.
    """
    lang_note = (
        "Respond in Hindi (Devanagari script)."
        if language == "Hindi"
        else "Respond in English."
    )
    return (
        f"You are an encouraging expert STEM tutor. A student studying STEM notes has a "
        f"follow-up query or counter-question about this concept:\n\n"
        f"--- Concept Context ---\n"
        f"Question / Concept: {card_front}\n"
        f"Current Answer / Law: {card_back}\n\n"
        f"--- Student's Query / Counter-Question ---\n"
        f"{user_query}\n\n"
        f"Instructions:\n"
        f"1. Directly and clearly answer the student's question or resolve their confusion.\n"
        f"2. Explain the scientific, physical, or mathematical reasoning step-by-step.\n"
        f"3. If their question challenges the premise, explain whether their reasoning "
        f"is valid, under what conditions/edge cases it applies, or why the standard formula holds.\n"
        f"4. Format all equations and mathematical terms using valid LaTeX ($inline$ and $$block$$).\n"
        f"5. Maintain an engaging, educational tutor tone. {lang_note}"
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
        f"clear, detailed, and elaborated explanation of this concept:\n\n"
        f"Question: {front}\n"
        f"Brief Answer: {back}\n\n"
        f"Please provide an elaborated explanation structured as follows:\n"
        f"- **Core Concept & Intuition**: Explain the fundamental principle in simple terms.\n"
        f"- **Step-by-step Breakdown**: How does it work? Detail any equations, variables, or units using LaTeX ($inline$ or $$block$$).\n"
        f"- **Practical Application / Example**: A real-world example or common problem scenario.\n\n"
        f"Keep the tone encouraging and academic. {lang_note}"
    )


# --------------------------------------------------------------------------- #
# Response Parsers                                                             #
# --------------------------------------------------------------------------- #

def clean_dot_code(raw: str) -> str:
    """Extract and sanitize a valid Graphviz DOT digraph string from LLM output."""
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"(digraph\s+[\w\d_]*\s*\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_formulas(raw: str) -> list[dict[str, Any]]:
    """Parse JSON array of synthesized formulas with variable definitions and assumptions."""
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty formula synthesis response.")

    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()

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
                raise ValueError(f"Could not parse synthesized formulas as JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of formulas, got {type(data).__name__}")

    validated = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"Formula {i + 1}")).strip()
        eq = str(item.get("latex_equation", item.get("equation", ""))).strip()
        desc = str(item.get("description", "")).strip()
        assumptions = str(item.get("assumptions", item.get("conditions", "Standard conditions"))).strip()
        vars_raw = item.get("variables", [])
        variables = []
        if isinstance(vars_raw, list):
            for v in vars_raw:
                if isinstance(v, dict):
                    variables.append({
                        "symbol": str(v.get("symbol", "")).strip(),
                        "definition": str(v.get("definition", "")).strip(),
                        "unit": str(v.get("unit", "-")).strip(),
                    })
        validated.append({
            "name": name,
            "latex_equation": eq,
            "description": desc,
            "variables": variables,
            "assumptions": assumptions,
        })
    return validated


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
            "Try uploading clearer content or increasing the card count."
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

    Two columns (front, back), no header row. Fields that contain commas,
    double-quotes, or newlines are quoted per RFC 4180.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    for card in cards:
        writer.writerow([card["front"], card["back"]])
    return buf.getvalue()


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
