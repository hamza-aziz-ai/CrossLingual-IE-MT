# Cross-Lingual Information Extraction and Machine Translation System

**NLP Applications (S2-25_AIMLCZG519) — Assignment 2, Part 1**
BITS Pilani Work Integrated Learning Programmes — M.Tech. Artificial Intelligence and Machine Learning

An integrated NLP pipeline that translates text between English and four Indic languages
(Hindi, Tamil, Marathi, Bengali) while extracting Named Entities, Relations, and Temporal
Events, exposed through an interactive Streamlit web dashboard.

**Repository:** [github.com/hamza-aziz-ai/CrossLingual-IE-MT](https://github.com/hamza-aziz-ai/CrossLingual-IE-MT)

## Group

| Student ID | Name | Contribution |
|---|---|---|
| 2024AC05133 | Hamza Aziz | 100% |
| 2024AC05114 | Kalpana S. | 100% |
| 2024AC05136 | Hardik Sharma | 100% |
| 2024AC05128 | Shubham Srivastava | 100% |
| 2024AC05109 | Nitin Verma | 100% |

## Quick start

The project uses [`pipenv`](https://pipenv.pypa.io/) for dependency management:

```bash
pip install --user pipenv
pipenv install --dev
pipenv run streamlit run frontend/app.py
```

`pipenv install --dev` installs everything, including the spaCy `en_core_web_sm`
model — no separate download step needed. See **[SETUP.md](SETUP.md)** for full
installation instructions (including CPU vs. GPU `torch`), troubleshooting, and how
to generate the application screenshots.

## Deliverables map

| Assignment requirement | Location |
|---|---|
| 1. Source Code | `backend/`, `frontend/` |
| 2. Setup Guide | [`SETUP.md`](SETUP.md) |
| 3. System Report | [`docs/System_Report.docx`](docs/System_Report.docx) / [`docs/System_Report.pdf`](docs/System_Report.pdf) |
| 4. Application Screenshots | [`screenshots/`](screenshots/) — generated via `tools/capture_screenshots.py`, see `screenshots/README.md` |
| 5. Task-B Document | [`docs/Task_B_Efficiency_and_Optimization_Plan.pdf`](docs/Task_B_Efficiency_and_Optimization_Plan.pdf) |
| Demo / module walkthrough (supplementary) | [`notebooks/demo_pipeline.ipynb`](notebooks/demo_pipeline.ipynb) — already executed with real outputs |
| Automated tests | `tests/test_offline_modules.py` (24 tests, `pipenv run pytest tests/ -v`) |

## Architecture at a glance

```
Indic input  --[NLLB-200: Indic→English]--> English text --[NER + RE + Event]--> IE results
English input --[NLLB-200: English→Indic]--> Indic text (display only)
              \--[NER + RE + Event on the original English]--> IE results
```

Translation runs on `facebook/nllb-200-distilled-600M`; Information Extraction runs on the
English side of the pair (Translate-then-Extract), using spaCy + IndicNER for NER,
REBEL (with an automatic rule-based fallback) for Relation Extraction, and spaCy +
`dateparser` for Event/Temporal Extraction. Full rationale, preprocessing steps, and
challenges encountered are documented in `docs/System_Report.docx`. The comparison against
a Direct Cross-Lingual Extraction architecture and a model-compression/deployment plan are
documented in `docs/Task_B_Efficiency_and_Optimization_Plan.pdf`.
