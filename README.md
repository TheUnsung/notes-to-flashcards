# ⚛️ Open-Source STEM Knowledge & Formula Synthesizer

> **Hacktoberfest Hack Day · OpenSTEM Track**  
> _Empowering science, technology, engineering, and mathematics education through open-weight AI and open scientific data._

---

## 🌟 Overview

The **Open-Source STEM Knowledge & Formula Synthesizer** is an educational workstation built exclusively on **Gemma 4** open-weight multimodal models. It synthesizes complex handwritten scientific notes, textbook pages, PDFs, and verified open-source literature into rigorous LaTeX formula sheets, interactive concept graphs, and spaced-repetition study decks.

CRITICAL LOCALHOST DESIGN: This application is strictly architected and configured to run **100% locally on `localhost`**. It requires no remote hosting, no Docker containers, and no cloud secret managers. Configuration is handled entirely via a local `.env` file or an in-app sidebar key input.

---

## 🚀 Key Features

### 1. 🌐 STEM Input & Open Data Grounding
- **Open Data Grounding**: Query verified peer-reviewed scientific literature directly from **arXiv** or comprehensive scientific summaries from **Wikipedia** to ground synthesis in open data.
- **Raw STEM Text & Notes**: Paste derivations, equations, laboratory notes, or textbook excerpts directly.
- **Multimodal Vision & PDF Engine**: Upload photos of handwritten chalkboard/paper notes or multi-page PDF documents with instant 90° rotation and high-resolution rendering.
- **Synthesis Mode Toggle**:
  - `[Standard STEM Flashcards]`: Spaced-repetition study cards with mathematical equations.
  - `[Formula & Derivation Extractor]`: Comprehensive formula sheet extraction.

### 2. 📐 Native LaTeX & Formula Sheet Synthesizer
- **Strict LaTeX Formatting**: All mathematical equations, physical laws, and chemical reactions are formatted in valid LaTeX (`$inline$` and `$$block$$`).
- **Structured Formula Cards**: Dedicated **Formula Synthesizer** tab displaying:
  - **The Equation**: Rendered crisply with Streamlit's native `st.latex()`.
  - **Variable Definitions & SI Units**: Structured tabular mapping of symbols, definitions, and standard SI units.
  - **Core Assumptions & Validity Limits**: Explicit constraints (e.g., *ideal gas behavior*, *non-relativistic velocities*, *constant acceleration*).

### 3. 🗺️ Automated Concept Graphs (Graphviz)
- **Hierarchical Knowledge Mapping**: Gemma 4 generates a directed Graphviz `DOT` graph linking overarching STEM domains, sub-principles, and governing laws.
- **Interactive Visualization**: Rendered natively via Streamlit's client-side `st.graphviz_chart()` without requiring system-level binary dependencies.
- **Exportable DOT Code**: Inspect, copy, and customize the raw Graphviz source code.

### 4. 💬 Conceptual Counter-Question & Socratic Q&A
- Challenge or probe any extracted formula or flashcard concept.
- Gemma 4 provides step-by-step physical derivations, discusses edge cases, and clarifies mathematical intuition.

### 5. 🎯 Spaced-Repetition Quiz & Anki Export
- Interactive flashcard quizzing with progress tracking, score calculation, and on-demand deep-dive explanations.
- One-click Anki-compatible CSV export.

---

## 🧠 Why Open-Weight AI (Gemma 4)

Built exclusively on Google's **Gemma 4** open-weight model family via the Google GenAI SDK:
- **`gemma-4-26b-a4b-it` (Default Preferred)**: Mixture-of-Experts architecture with 4B active parameters per token, delivering low latency and rapid generation.
- **`gemma-4-31b-it` (High-Capacity Fallback)**: Dense 31B parameter model for complex conceptual derivations.

### Automatic Fallback & Resilience Matrix

| Error Type | Behavior |
|---|---|
| `500` / `503` / `INTERNAL` / `UNAVAILABLE` | Retries current model up to 5 times (2s, 4s, 8s, 16s, 32s); switches to fallback model if retries exceed 5. |
| `404` / Model Not Found | Skips to fallback model immediately. |
| `400` / `401` / `403` / Authentication | Stops immediately with a diagnostic message. |
| Model Busy / Token Quota | Seamless handoff to the alternative Gemma 4 model. |

---

## 💻 Localhost Quickstart

### Prerequisites
- Python 3.10+
- A Google Gemini API Key (free tier available at [aistudio.google.com](https://aistudio.google.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/TheUnsung/notes-to-flashcards.git
cd notes-to-flashcards
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
Create a `.env` file in the root folder:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
*(Alternatively, enter your key directly into the password field in the Streamlit sidebar.)*

### 5. Run the Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Testing

The repository includes a comprehensive 79+ test test suite with 100% mocked offline execution:

```bash
# Run the full test suite
python -m pytest tests/test_cards.py -v
```

---

## 📁 Repository Structure

```
├── app.py               # Streamlit application UI, tabs, & Gemma 4 fallback engine
├── cards.py             # Pure core logic: LaTeX prompt builders, parsers, Graphviz DOT cleaner, Open Data APIs
├── requirements.txt     # Python dependencies (wikipedia, arxiv, python-dotenv, pymupdf, google-genai)
├── .env                 # Local API key storage (gitignored)
├── .gitignore           # Ignores .env, .venv, bytecode, and temporary files
├── README.md            # Documentation and setup instructions
└── tests/
    ├── test_cards.py    # Pytest suite testing all pure functions and mocked API workflows
    └── create_sample.py # Test image utility
```

---

## 📜 License & Acknowledgements

Created for the **Hacktoberfest Hack Day · OpenSTEM Track**. Released under the MIT License.
Powered by Google Gemma 4, Wikipedia API, and arXiv API.
