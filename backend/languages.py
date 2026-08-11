"""
languages.py
=============

Central registry of every language the system supports, mapping the
human-readable display name used in the Streamlit UI to the two codes
we need internally:

* ``flores_code``  -- the FLORES-200 language code required by the
  ``facebook/nllb-200-distilled-600M`` tokenizer (e.g. ``"hin_Deva"``).
  NLLB was trained on FLORES-200 codes, which encode both the language
  *and* the script (Devanagari, Tamil, Bengali, Latin, ...). Passing the
  wrong code silently produces garbage translations, so keeping this
  mapping in one place avoids that class of bug.
* ``iso_code``     -- a short ISO-639-1-style code used for UI labels,
  file names, and to select the correct NER model.

Why a registry instead of hard-coding strings everywhere?
-----------------------------------------------------------
Every other backend module (mt_engine, ner_module, pipeline) needs to
translate a UI-selected language name into the code a specific model
expects. Centralizing this mapping means we only fix a typo or add a
new language in one place, and every module stays in sync.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class LanguageInfo:
    """
    Immutable record describing one supported language.

    Attributes
    ----------
    display_name : str
        Human-readable name shown in the Streamlit dropdowns, e.g. "Hindi".
    iso_code : str
        Short code used for UI labels / file naming, e.g. "hi".
    flores_code : str
        FLORES-200 code required by the NLLB-200 tokenizer/model, e.g. "hin_Deva".
    is_indic : bool
        True for every language other than English. Used to route text to the
        correct NER backend (spaCy for English, IndicNER for Indic scripts) and
        to decide the direction of the Translate-then-Extract pipeline.
    unicode_block : Optional[Tuple[int, int]]
        (start, end) Unicode code-point range used for lightweight, offline
        script-based language auto-detection (see `text_utils.detect_script_language`).
        None for English, since Latin script alone is not a reliable Indic/English
        discriminator -- English is therefore the fallback when no Indic script
        is detected in the input text.
    """

    display_name: str
    iso_code: str
    flores_code: str
    is_indic: bool
    unicode_block: Optional[Tuple[int, int]] = None


# ---------------------------------------------------------------------------
# The four Indic languages named explicitly in the assignment brief (Hindi,
# Tamil, Marathi, Bengali) plus English as the pivot language. All four are
# covered by a *single* NLLB-200 checkpoint (no per-language model swapping
# needed), which keeps the deployment story simple -- see the System Report
# for why NLLB-200 was chosen over per-pair Helsinki-NLP OPUS-MT models.
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: Dict[str, LanguageInfo] = {
    "English": LanguageInfo(
        display_name="English",
        iso_code="en",
        flores_code="eng_Latn",
        is_indic=False,
        unicode_block=None,
    ),
    "Hindi": LanguageInfo(
        display_name="Hindi",
        iso_code="hi",
        flores_code="hin_Deva",
        is_indic=True,
        unicode_block=(0x0900, 0x097F),  # Devanagari
    ),
    "Marathi": LanguageInfo(
        display_name="Marathi",
        iso_code="mr",
        flores_code="mar_Deva",
        is_indic=True,
        unicode_block=(0x0900, 0x097F),  # Marathi also uses Devanagari
    ),
    "Tamil": LanguageInfo(
        display_name="Tamil",
        iso_code="ta",
        flores_code="tam_Taml",
        is_indic=True,
        unicode_block=(0x0B80, 0x0BFF),  # Tamil
    ),
    "Bengali": LanguageInfo(
        display_name="Bengali",
        iso_code="bn",
        flores_code="ben_Beng",
        is_indic=True,
        unicode_block=(0x0980, 0x09FF),  # Bengali
    ),
}


def get_language(display_name: str) -> LanguageInfo:
    """
    Look up a `LanguageInfo` record by its display name.

    Parameters
    ----------
    display_name : str
        One of the keys of `SUPPORTED_LANGUAGES`, e.g. "Hindi".

    Returns
    -------
    LanguageInfo

    Raises
    ------
    KeyError
        If the language is not supported. Raised (rather than returning None)
        so that an unsupported-language bug fails loudly at the call site
        instead of propagating a None deep into the MT/NER pipeline.
    """
    try:
        return SUPPORTED_LANGUAGES[display_name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise KeyError(
            f"Unsupported language '{display_name}'. Supported languages: {supported}"
        ) from exc


def flores_to_display_name(flores_code: str) -> str:
    """Reverse lookup: FLORES-200 code -> display name. Used when logging/exporting."""
    for name, info in SUPPORTED_LANGUAGES.items():
        if info.flores_code == flores_code:
            return name
    raise KeyError(f"No supported language registered for FLORES code '{flores_code}'")
