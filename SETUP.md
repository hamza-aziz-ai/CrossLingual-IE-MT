# Setup Guide

Cross-Lingual Information Extraction and Machine Translation System
NLP Applications (S2-25_AIMLCZG519) — Assignment 2, Part 1, Task A

This guide covers installing dependencies and running the web application
locally, as required by the assignment deliverables list (item 2, "Setup
Guide").

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.14 (pinned in `Pipfile`'s `[requires]`) | `pipenv install` provisions this automatically if `pyenv`/`pipenv --python` can find it; earlier 3.x versions are untested |
| RAM | 8 GB | 16 GB recommended — the MT model alone uses ~2.5 GB loaded |
| Disk space | ~6 GB free | For model weight downloads (MT + NER + RE models, first run only) |
| Internet connection | Required on first run | Downloads model weights from the HuggingFace Hub; not required afterwards (models are cached locally) |
| GPU | Optional | `Pipfile` defaults `torch` to a CUDA build (see "CPU vs. GPU" below) but the app **does not require a GPU** — every module auto-detects at runtime (`torch.cuda.is_available()`) and falls back to CPU automatically, just slower (~30-60s per sentence for translation on CPU vs. ~1-2s on a GPU) |

## 2. Installation

The project uses [`pipenv`](https://pipenv.pypa.io/) for dependency management —
one `Pipfile` describes both runtime and development dependencies, and
`pipenv` creates and manages the virtual environment for you (no separate
`python -m venv` step needed).

```bash
# 1. Install pipenv (one-time, if you don't already have it)
pip install --user pipenv

# 2. Clone/copy the project, then from the project root:
cd CrossLingual-IE-MT

# 3. Install runtime + development dependencies into a fresh virtualenv
pipenv install --dev
```

This single command also installs the spaCy `en_core_web_sm` model (pinned
directly as a wheel URL in the `Pipfile`) — there is no separate
`python -m spacy download` step to remember.

**First install only:** the bundled `Pipfile.lock` was generated in a Linux
CI-style environment. `pipenv` will detect that it needs re-resolving for
your platform/Python version and print something like `Pipfile.lock is
out of date, updating to match Pipfile...` before proceeding — this is
expected and not an error; let it finish (it re-locks once and caches the
result for subsequent installs).

**CPU vs. GPU:** the `Pipfile` pins `torch` to PyTorch's CUDA 13.2 wheel
index (`download.pytorch.org/whl/cu132`) by default, since the module
code already auto-detects and gracefully falls back to CPU at *runtime*
(`torch.cuda.is_available()` in `mt_engine.py` / `relation_extraction.py`)
regardless of which `torch` build is installed. That runtime fallback does
**not** help if the CUDA wheel itself fails to *install* on your machine
(no matching NVIDIA driver, older CUDA version, or no GPU at all — a plain
`pip install` of a CUDA wheel still downloads and installs fine even
without a GPU, but is a needless multi-gigabyte download). If you don't
have a compatible NVIDIA GPU, swap in the CPU-only build instead:

```bash
pipenv install --dev
pipenv run pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

If you have an NVIDIA GPU but a different CUDA version than 13.2, adjust
the `cu132` suffix accordingly — see
https://pytorch.org/get-started/locally/ for the correct tag, then
re-run the swap command above with that index URL instead.

**Running commands:** either prefix commands with `pipenv run` (e.g.
`pipenv run streamlit run frontend/app.py`), or start a shell inside the
environment once with `pipenv shell` and run commands directly for the rest
of that session. Convenience shortcuts are also defined in `Pipfile`'s
`[scripts]` section — see `pipenv run app`, `pipenv run test`, and
`pipenv run notebook` below.

## 3. Running the web application

```bash
pipenv run streamlit run frontend/app.py
# or, using the Pipfile shortcut:
pipenv run app
```

This opens the dashboard at `http://localhost:8501` in your default browser.

**First run:** the first time you click "Translate & Extract", the app
downloads the required model weights from the HuggingFace Hub:

| Model | Approx. size | Used for |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | ~2.4 GB | Machine Translation + word alignment |
| `ai4bharat/IndicNER` | ~1.1 GB | NER on Indic-language text |
| `Babelscape/rebel-large` | ~1.6 GB | Relation Extraction (falls back to a rule-based extractor automatically if this cannot be downloaded — see below) |

Models are cached under `~/.cache/huggingface/hub` (Linux/macOS) or
`%USERPROFILE%\.cache\huggingface\hub` (Windows) after the first download,
so subsequent runs start immediately.

## 4. Running the demo notebook

```bash
pipenv run jupyter notebook notebooks/demo_pipeline.ipynb
# or, using the Pipfile shortcut:
pipenv run notebook
```

Run all cells top to bottom (`Kernel > Restart & Run All`). The notebook is
also provided already-executed with real output cells (translations,
entity tables, the alignment heatmap, etc.) so it can be read without
re-running it, if you just want to review the results.

## 5. Running the module-level tests

A lightweight smoke test (no model downloads required beyond spaCy) is in
`tests/test_offline_modules.py` and covers text normalization, script
detection, the REBEL output parser, and temporal-expression extraction:

```bash
pipenv run pytest tests/ -v
# or, using the Pipfile shortcut:
pipenv run test
```

## 6. Capturing application screenshots

`tools/capture_screenshots.py` automates driving the running Streamlit app
through the full Input -> Translation & Alignment -> IE Extraction ->
JSON/CSV Download workflow with Playwright and saves numbered PNG
screenshots to `screenshots/`. To use it:

```bash
# 1. Install the Playwright browser binary (one-time)
pipenv run playwright install chromium

# 2. In one terminal, start the app:
pipenv run app

# 3. In a second terminal, run the capture script:
pipenv run python tools/capture_screenshots.py
```

See `tools/capture_screenshots.py`'s module docstring for what each
numbered screenshot captures, and `screenshots/README.md` for the expected
output.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Should not happen — `en_core_web_sm` is installed automatically by `pipenv install --dev` (pinned as a wheel URL in `Pipfile`). If it does, run `pipenv run python -m spacy download en_core_web_sm` manually. |
| `pipenv install` fails to resolve / `Locking Failed!` | Run `pipenv lock --clear` to discard any corrupted resolver cache and retry, then `pipenv install --dev` again. |
| `Pipfile.lock is out of date, updating to match Pipfile...` | Expected on first install (see Section 2) — not an error. |
| Translation is extremely slow (minutes per sentence) | Running on CPU is expected to be slow; this is normal without a GPU (check with `pipenv run python -c "import torch; print(torch.cuda.is_available())"`). Reduce input length, or confirm the CUDA `torch` build installed correctly if you do have an NVIDIA GPU (Section 2). |
| `pipenv install` downloads a multi-gigabyte CUDA `torch` wheel but you have no NVIDIA GPU | Expected — `Pipfile` defaults to the CUDA 13.2 build. Swap to the CPU-only build (Section 2) to avoid the large download; the app runs identically either way, just slower without a GPU. |
| `Not enough free disk space to download the file` warning during model download | Free up disk space (the NLLB model alone needs ~2.4 GB free); the download will still often complete if the *actual* available space is closer to the file size than the conservative estimate `huggingface_hub` uses. |
| Relations tab always shows `source: rule_based`, never `rebel` | `Babelscape/rebel-large` failed to download (offline, or insufficient disk/RAM) — the app has automatically fallen back to the rule-based extractor. This is expected, graceful degradation, not a bug; see `docs/System_Report.docx`. |
| `AttributeError: 'NllbTokenizer' object has no attribute 'lang_code_to_id'` | This is handled automatically by `mt_engine.py` (falls back to `convert_tokens_to_ids`); if you still see this error, run `pipenv update transformers` to pick up a version >= 4.41. |
| Streamlit shows a blank page / connection refused | Make sure you ran the command from the project root so `frontend/app.py`'s relative imports resolve; check the terminal running `pipenv run streamlit run frontend/app.py` for a traceback. |
| Devanagari/Tamil/Bengali text renders as boxes in the alignment heatmap | Should not happen — the app bundles Noto Sans fonts for these scripts under `assets/fonts/`. If it does, confirm `assets/fonts/` was copied along with the rest of the project (all three `.ttf` files must be present). |

## 8. Project structure

```
CrossLingual-IE-MT/
├── backend/                  # MT, NER, RE, Event/Temporal modules + pipeline orchestrator
│   ├── languages.py          # Supported language registry
│   ├── text_utils.py         # Normalization, sentence splitting, script detection
│   ├── fonts.py              # Indic-script font handling for matplotlib
│   ├── mt_engine.py          # NLLB-200 translation + cross-attention word alignment
│   ├── ner_module.py         # spaCy (English) + IndicNER (Indic) named entity recognition
│   ├── relation_extraction.py# REBEL + rule-based relation extraction
│   ├── event_temporal.py     # Event trigger + temporal expression extraction/linking
│   └── pipeline.py           # Orchestrates all modules; JSON/CSV export
├── frontend/
│   ├── app.py                # Streamlit web dashboard
│   └── ui_helpers.py         # HTML/dataframe/graphviz rendering helpers
├── notebooks/
│   └── demo_pipeline.ipynb   # Module-by-module demo with theory + real executed outputs
├── assets/fonts/              # Bundled Noto Sans Devanagari/Tamil/Bengali fonts
├── tools/
│   └── capture_screenshots.py# Automated screenshot capture (Playwright)
├── tests/
│   └── test_offline_modules.py
├── sample_data/                # Example input sentences for quick testing
├── docs/
│   ├── System_Report.md / .pdf
│   └── Task_B_Efficiency_and_Optimization_Plan.pdf
├── screenshots/               # Generated by tools/capture_screenshots.py
├── Pipfile                    # Dependency spec (pipenv) — source of truth for versions
├── Pipfile.lock                # Resolved/pinned dependency graph (regenerated on first install)
├── CLAUDE.md                   # Repo instructions for Claude Code
├── AGENTS.md                   # Repo instructions for ChatGPT Codex / other agents
├── SETUP.md                   # This file
└── README.md
```
