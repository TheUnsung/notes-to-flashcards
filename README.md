# ⚛️ STEM Formula Extractor & Flashcards

> **Hacktoberfest Hack Day · OpenSTEM Track**  
> _Extract governing formulas, SI units, and validity assumptions, or generate study flashcards from your STEM notes._

---

## 🌟 Overview

A streamlined educational tool built exclusively with **Gemma 4** open-weight models. It focuses on two core tasks:
1. **📐 Formula Extractor**: Extracts all governing mathematical and physical equations, variable definitions with standard SI units, and core validity conditions/assumptions.
2. **🃏 Flashcards**: Generates concise study flashcards formatted with LaTeX mathematical equations, complete with one-click Anki CSV export.

CRITICAL LOCALHOST DESIGN: Strictly configured to run **100% locally on `localhost`**. Your API key is loaded securely in the background from your local `.env` file and never exposed in the UI.

---

## 📸 Screenshots

| 📐 Formula Extractor | 🃏 Study Flashcards |
|:---:|:---:|
| ![Formula Extractor](screenshots/formula_extractor.png) | ![Study Flashcards](screenshots/flashcards.png) |

> _To add your own screenshots, save your images as `screenshots/formula_extractor.png` and `screenshots/flashcards.png`._

---

## 🚀 Features

### 1. 📐 Formula Extractor
- Extracts mathematical laws, physics equations, and chemical relationships.
- Formats equations cleanly using valid LaTeX (`st.latex()`).
- Generates a structured table of variable symbols, definitions, and SI units.
- Explicitly lists critical validity limits and assumptions (e.g., ideal gases, non-relativistic speeds, constant acceleration).

### 2. 🃏 Flashcards & Anki Export
- Generates study flashcards from your notes with inline and block LaTeX equations.
- Export directly to Anki-compatible `.csv` format for spaced-repetition study.

### 3. 📥 Clean Multi-Format Input
- **Image & PDF Upload**: Upload photos of handwritten notes or textbook pages (with 90° clockwise rotation and multi-page PDF page selection).
- **Text Area**: Paste raw equations, derivations, or lecture notes directly.

---

## 🧠 Supported Models (Gemma 4 Only)

- **`gemma-4-26b-a4b-it` (Default)**: Fast Mixture-of-Experts (4B active parameters per token).
- **`gemma-4-31b-it` (Fallback)**: High-capacity dense 31B parameter model.

---

## 💻 Localhost Quickstart

### 1. Setup Virtual Environment & Dependencies
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Run on Localhost
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** (or your active local port) in your browser.

---

## 🧪 Testing

```bash
python -m pytest tests/test_cards.py -v
```
All 79 unit tests run completely offline with mocked API responses.

---

## 📁 Repository Structure

```
├── app.py               # Streamlit application UI & Gemma 4 fallback engine
├── cards.py             # Pure core logic: LaTeX prompt builders, parsers, and image helpers
├── requirements.txt     # Python dependencies
├── .env                 # Local API key storage (gitignored)
├── .gitignore           # Ignores .env, .venv, bytecode, and temporary files
├── README.md            # Documentation and setup instructions
├── screenshots/         # Application preview screenshots
└── tests/
    ├── test_cards.py    # Pytest test suite (79 tests)
    └── create_sample.py # Test image utility
```
