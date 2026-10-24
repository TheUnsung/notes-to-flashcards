# 🃏 Notes to Flashcards

> **Hacktoberfest Hack Day · Gemma 4 Track · OpenSTEM Theme**  
> _Open tools for Science, Technology, Engineering and Mathematics_

---

## What the project does

Upload a photo of handwritten notes, a textbook page, or a PDF document and
**Gemma 4** reads the page and generates study flashcards instantly.

- **Image & PDF support** – upload PNG, JPG, JPEG, or multi-page PDF documents
  (select specific pages to study).
- **Browse mode** – all cards shown in expandable panels with optional on-demand deep dives.  
- **Quiz mode** – one card at a time: reveal the answer, mark _I knew it_ or
  _Review again_, and get a final score.  
- **Counter-Question & Q&A** – select any card from a dropdown to challenge the concept,
  ask follow-up queries, and get step-by-step explanations from Gemma 4.
- **Anki export** – one-click CSV download ready to import into
  [Anki](https://apps.ankiweb.net/).

---

## Why open-weight AI is central

This project is built exclusively on **Gemma 4**, Google's open-weight multimodal
model family. No Gemini model, no other AI API, and no proprietary black box is
used anywhere in the codebase. Using open-weight models means:

- Any student or teacher can run inference locally or through a free API tier.
- The model weights are publicly available for research, fine-tuning, and auditing.
- It aligns with the **OpenSTEM** hackathon theme: open tools that lower the
  barrier to STEM education worldwide.

---

## Which Gemma 4 models are used

| Priority | Model ID | Notes |
|:-:|---|---|
| 1 (default) | `gemma-4-26b-a4b-it` | Faster default – MoE architecture with 4B active parameters for rapid token generation |
| 2 (fallback) | `gemma-4-31b-it` | High-capacity fallback – dense 31B parameter model |

The app tries the **preferred model** (user-selectable in the sidebar) first.
If it is busy or unavailable, it automatically falls back to the other model.

### Fallback & retry behaviour

| Error type | Action |
|---|---|
| 500 / 503 / INTERNAL / UNAVAILABLE | Retry the same model up to 5 times (2 s, 4 s, 8 s, 16 s, 32 s); automatically switches to fallback model if it exceeds 5 retries |
| 404 / model not found | Skip to the next model immediately |
| 400 / 401 / 403 / bad API key | Stop with a clear error message – no retries |
| All models exhausted | Show a clear error; image stays in place so the user can retry |

---

## Setup

### 1 – Clone the repo

```bash
git clone <your-repo-url>
cd notes-to-flashcards
```

### 2 – Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows CMD
.venv\Scripts\activate.bat
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### 4 – Set your Gemini API key

```bash
# macOS / Linux
export GEMINI_API_KEY=your_key_here

# Windows CMD
set GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/).

> **Never commit your key.** `.env` is listed in `.gitignore`.

---

## How to run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running tests

```bash
# Generate the sample test image first (only needed once)
python tests/create_sample.py

# Run the full test suite (no API key or network required)
pytest tests/ -v
```

---

## Project structure

```
.
├── app.py               # Streamlit UI + generate_with_fallback()
├── cards.py             # Pure functions: build_prompt, shrink_image,
│                        #   parse_cards, cards_to_csv
├── requirements.txt
├── .gitignore
├── README.md
└── tests/
    ├── create_sample.py # Script to generate sample_notes.png
    ├── sample_notes.png # Auto-generated test image
    └── test_cards.py    # pytest suite (no network required)
```

---

## Screenshots

<!-- Add screenshots here after running the app -->

| Upload & Generate | Browse Cards | Quiz Mode |
|:-:|:-:|:-:|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## Powered by

`gemma-4-31b-it` and `gemma-4-26b-a4b-it` via the Gemini API  
Only Gemma 4 open-weight models are used in this project.
