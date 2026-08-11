# Application Screenshots

This folder holds the step-by-step screenshots required by the assignment
deliverables list (item 4): **Input -> Translation & Alignment -> IE
Extraction -> JSON/CSV Download**.

## How these were generated

Run `tools/capture_screenshots.py` (see its docstring and `SETUP.md`
Section 6) against a locally running instance of the app:

```bash
streamlit run frontend/app.py        # terminal 1
python tools/capture_screenshots.py  # terminal 2
```

This produces:

| File | Shows |
|---|---|
| `01_initial_state.png` | Dashboard on first load, empty input |
| `02_input_entered.png` | Example sentence entered in the text area |
| `03_translation_view.png` | Translated output after clicking "Translate & Extract" |
| `04_alignment_heatmap.png` | Word-alignment cross-attention heatmap |
| `05_ner_view.png` | Named Entities tab, colour-coded highlights + table |
| `06_relations_view.png` | Relations tab, entity-relation graph + table |
| `07_events_view.png` | Events & Time tab, highlighted triggers + linked time expressions |
| `08_export_view.png` | Export tab with JSON/CSV download buttons |

## Why this project ships a *script* rather than static image files

The backend modules (MT, NER, RE, Event/Temporal — see
`notebooks/demo_pipeline.ipynb`) were verified to run correctly end-to-end
against the real pretrained models during development, in a sandboxed
Linux environment with no windowing system and no root/administrator
access to install the shared graphics libraries a real browser needs to
render pages (`libXdamage`, `libgbm`, `libnss3`, etc. — a headless Chromium
binary was downloaded but failed to launch for exactly this reason:
`error while loading shared libraries: libXdamage.so.1`). Rather than
fabricate placeholder screenshots that would not represent genuine
application behaviour, `tools/capture_screenshots.py` is provided so any
group member can generate authentic, reproducible screenshots of the real
running app on their own machine in a few minutes — see `SETUP.md` Section 6.
