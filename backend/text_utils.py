"""
text_utils.py
==============

Shared text preprocessing / normalization helpers used by the MT, NER,
Relation-Extraction and Event/Temporal modules.

Theory
------
Neural MT and transformer-based IE models are sensitive to surface-level
noise: mixed Unicode normal forms (NFC vs NFD), stray zero-width joiners
that Indic keyboards/OCR frequently insert, inconsistent whitespace, and
un-split multi-sentence paragraphs (which hurts alignment quality and can
truncate at the model's max-token limit). This module performs the
"complete preprocessing (tokenization, normalization, script handling for
Indic languages)" step required by the assignment brief, independent of
any specific model.
"""

import re
import unicodedata
from typing import List

from .languages import SUPPORTED_LANGUAGES, LanguageInfo

# Zero-width characters that Indic-script text (especially text copy-pasted
# from WhatsApp/PDF exports) frequently contains. They are invisible but
# change tokenization, so we strip them during normalization.
_ZERO_WIDTH_CHARS = "".join(["​", "‌", "‍", "﻿"])
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")

# Collapse runs of whitespace (including tabs/newlines mid-sentence) to a
# single space; multiple blank lines are preserved as a single paragraph
# break so sentence segmentation below still works on longer inputs.
_MULTI_SPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

# A conservative sentence splitter for languages we do not have a spaCy
# sentencizer for out of the box (Hindi/Marathi/Tamil/Bengali). It splits on
# the Devanagari danda "।", double danda "॥", and standard Latin sentence
# punctuation, while avoiding splitting on decimal points / abbreviations
# followed by a lowercase letter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥.!?])\s+(?=[A-Zऀ-෿])")


def normalize_text(text: str) -> str:
    """
    Apply Unicode + whitespace normalization to raw user input.

    Why NFC and not NFD/NFKC
    -------------------------
    Devanagari, Tamil and Bengali all use combining vowel signs (matras).
    NFC (Normalization Form Canonical Composition) is the form both the
    NLLB tokenizer's SentencePiece model and IndicNER's XLM-R tokenizer
    were trained on, so composing to NFC prevents silent
    out-of-vocabulary/UNK-token issues that arise when a matra and its
    base consonant arrive as separate NFD code points. NFKC is avoided
    because it also performs *compatibility* folding (e.g. collapsing
    presentation-form ligatures), which can change Tamil/Bengali
    conjunct rendering in ways that alter meaning.

    Parameters
    ----------
    text : str
        Raw text as typed/pasted by the user.

    Returns
    -------
    str
        Normalized text: NFC-composed, zero-width characters removed,
        whitespace collapsed, and leading/trailing whitespace stripped.
    """
    if not text:
        return ""

    # Step 1: Unicode canonical composition (see docstring above).
    text = unicodedata.normalize("NFC", text)

    # Step 2: Remove invisible zero-width characters that do not carry
    # linguistic meaning but corrupt tokenization/alignment.
    text = _ZERO_WIDTH_RE.sub("", text)

    # Step 3: Collapse repeated horizontal whitespace to one space, and
    # collapse 3+ newlines down to a single paragraph break.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def split_sentences(text: str, language: LanguageInfo) -> List[str]:
    """
    Split a (already-normalized) block of text into sentences.

    Why this matters for MT
    ------------------------
    NLLB-200 (like most seq2seq MT models) was fine-tuned on single
    sentences/short segments. Feeding it a multi-sentence paragraph in one
    call still "works" mechanically, but translation quality and the
    word-alignment visualization degrade because cross-attention has to
    spread across a much longer, structurally mixed sequence. Splitting
    into sentences, translating each independently, and re-joining keeps
    both translation quality and the alignment heatmap interpretable.

    Parameters
    ----------
    text : str
        Normalized input text (see `normalize_text`).
    language : LanguageInfo
        The language the text is written in. English uses spaCy's
        statistical sentencizer (loaded lazily by the caller and passed
        as already-processed spans is *not* done here to avoid a hard
        dependency on spaCy inside this lightweight utility module);
        Indic languages fall back to the danda/punctuation regex above,
        since spaCy's small English pipeline is not trained for Indic
        sentence boundaries.

    Returns
    -------
    List[str]
        Non-empty sentence strings, in original order.
    """
    if not text:
        return []

    # A single regex-based splitter is used uniformly here (rather than
    # loading a spaCy sentencizer per language) to keep this module
    # dependency-free. The mt_engine module additionally guards against
    # over-long single "sentences" (e.g. text with no terminal punctuation)
    # by chunking on a max-token basis before calling the model.
    raw_sentences = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    return sentences if sentences else [text.strip()]


def detect_script_language(text: str) -> str:
    """
    Best-effort, offline language auto-detection based purely on the
    Unicode script of the characters present in `text`.

    Why script-based detection instead of a statistical language ID model
    -----------------------------------------------------------------------
    A dedicated language-ID model (e.g. fastText lid.176) is one more
    multi-hundred-MB download for a problem that, for our 5-language
    closed set, is solvable deterministically: Hindi/Marathi (Devanagari),
    Tamil (Tamil script) and Bengali (Bengali script) each occupy disjoint
    Unicode blocks, and anything without Indic script characters is
    treated as English. The one true ambiguity -- Hindi vs. Marathi, which
    share the Devanagari block -- cannot be resolved from script alone;
    we default to Hindi in that case and let the user override the
    dropdown, which is documented as a known limitation in the System
    Report.

    Parameters
    ----------
    text : str
        Raw or normalized input text.

    Returns
    -------
    str
        One of the display names in `SUPPORTED_LANGUAGES` ("English" by
        default when no Indic script characters are found).
    """
    if not text:
        return "English"

    script_counts = {name: 0 for name in SUPPORTED_LANGUAGES if SUPPORTED_LANGUAGES[name].is_indic}

    for ch in text:
        code_point = ord(ch)
        for name, info in SUPPORTED_LANGUAGES.items():
            if info.unicode_block is None:
                continue
            start, end = info.unicode_block
            if start <= code_point <= end:
                script_counts[name] += 1

    # Devanagari is shared by Hindi and Marathi; sum them under a single
    # "Devanagari" bucket before picking a winner, then map back to Hindi
    # as the safer default (Hindi has broader NLLB/IndicNER training data
    # than Marathi, so translation/NER quality is typically higher).
    devanagari_total = script_counts.get("Hindi", 0) + script_counts.get("Marathi", 0)
    tamil_total = script_counts.get("Tamil", 0)
    bengali_total = script_counts.get("Bengali", 0)

    best_script, best_count = max(
        [("Hindi", devanagari_total), ("Tamil", tamil_total), ("Bengali", bengali_total)],
        key=lambda pair: pair[1],
    )

    if best_count == 0:
        return "English"
    return best_script


def format_subword_token_for_display(token: str) -> str:
    """
    Cosmetic formatting for SentencePiece sub-word tokens before they are
    used as matplotlib tick labels or table cells.

    SentencePiece (used by the NLLB tokenizer) marks the start of a new
    word with the meta-character "▁" (U+2581, LOWER ONE EIGHTH BLOCK)
    rather than a plain space. Left as-is, this character (a) is not part
    of any natural-language alphabet and renders as a missing-glyph box
    in most fonts including the bundled Devanagari/Tamil/Bengali fonts
    (see `backend/fonts.py`), and (b) is visually confusing to a reader
    who does not know SentencePiece internals. Replacing it with a plain
    space is display-only -- it must never be applied to tokens before
    they are used for actual alignment-matrix indexing, only when
    rendering them to a human.

    Parameters
    ----------
    token : str
        A raw sub-word token as returned by the tokenizer.

    Returns
    -------
    str
        The token with "▁" replaced by a plain space (and outer
        whitespace stripped, since a token that is *only* "▁" would
        otherwise display as an empty-looking cell).
    """
    return token.replace("▁", " ").strip() or token


def truncate_for_display(text: str, max_chars: int = 4000) -> str:
    """
    Guard-rail against pasting extremely long documents into the UI.

    Why: the NLLB tokenizer silently truncates at its max position length
    (typically 512 sub-word tokens); translating only the first ~512
    tokens of a much longer document without warning the user would be
    misleading. We instead cap the *input* at a generous character budget
    and surface the truncation to the caller (the Streamlit app shows a
    warning) rather than let the model truncate invisibly mid-sentence.

    Parameters
    ----------
    text : str
    max_chars : int
        Character budget. 4000 chars is comfortably below the 512-subword
        NLLB limit for Latin-script text and well below it for Indic
        scripts (which tokenize to more sub-words per character).

    Returns
    -------
    str
        `text` unchanged if within budget, otherwise truncated at the
        nearest sentence boundary before `max_chars`.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_boundary = max(truncated.rfind("।"), truncated.rfind("."), truncated.rfind("\n"))
    if last_boundary > 0:
        truncated = truncated[: last_boundary + 1]
    return truncated
