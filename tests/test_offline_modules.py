"""
test_offline_modules.py
==========================

Lightweight smoke tests covering the parts of the backend that do **not**
require downloading multi-gigabyte model weights (MT/IndicNER/REBEL are
exercised manually / in the demo notebook instead, since a CI-style unit
test suite that re-downloads several GB of weights on every run is
impractical for a course assignment). This file specifically covers:

* `text_utils` — normalization, sentence splitting, script detection,
  sub-word display formatting.
* `languages` — registry lookups and error handling.
* `relation_extraction._parse_rebel_triplets` — pure-Python parser, tested
  against a fixed example string (no model call).
* `relation_extraction` rule-based fallback and `event_temporal` — both use
  only spaCy's small English pipeline (~13 MB, fast to install in CI) and
  `dateparser`, so they run "for real" here rather than as pure parser
  tests.

Run with: `python -m pytest tests/ -v`
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from backend.languages import get_language, flores_to_display_name, SUPPORTED_LANGUAGES
from backend.text_utils import (
    normalize_text,
    split_sentences,
    detect_script_language,
    truncate_for_display,
    format_subword_token_for_display,
)
from backend.relation_extraction import _parse_rebel_triplets, RelationExtractor
from backend.event_temporal import EventTemporalExtractor
from backend.ner_module import NERExtractor


# --------------------------------------------------------------------- #
# languages.py
# --------------------------------------------------------------------- #

def test_get_language_known():
    info = get_language("Hindi")
    assert info.flores_code == "hin_Deva"
    assert info.is_indic is True


def test_get_language_unknown_raises():
    with pytest.raises(KeyError):
        get_language("Klingon")


def test_flores_round_trip():
    for name, info in SUPPORTED_LANGUAGES.items():
        assert flores_to_display_name(info.flores_code) == name


# --------------------------------------------------------------------- #
# text_utils.py
# --------------------------------------------------------------------- #

def test_normalize_text_collapses_whitespace_and_strips():
    assert normalize_text("  Hello   world  \n\n\n test  ") == "Hello world \n\n test"


def test_normalize_text_empty():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_split_sentences_english():
    sentences = split_sentences(
        "Tim Cook announced the new iPhone. Apple will launch next week.",
        get_language("English"),
    )
    assert sentences == [
        "Tim Cook announced the new iPhone.",
        "Apple will launch next week.",
    ]


def test_detect_script_language_english():
    assert detect_script_language("Hello world, this is English text.") == "English"


def test_detect_script_language_hindi():
    assert detect_script_language("नमस्ते दुनिया") == "Hindi"


def test_detect_script_language_tamil():
    assert detect_script_language("வணக்கம் உலகம்") == "Tamil"


def test_detect_script_language_bengali():
    assert detect_script_language("ওহে বিশ্ব") == "Bengali"


def test_detect_script_language_empty_defaults_english():
    assert detect_script_language("") == "English"


def test_truncate_for_display_short_text_unchanged():
    text = "short text"
    assert truncate_for_display(text, max_chars=100) == text


def test_truncate_for_display_long_text_truncated():
    text = "A" * 5000
    result = truncate_for_display(text, max_chars=100)
    assert len(result) <= 100


def test_format_subword_token_for_display_strips_sentencepiece_marker():
    assert format_subword_token_for_display("▁Hello") == "Hello"
    assert format_subword_token_for_display("world") == "world"


# --------------------------------------------------------------------- #
# relation_extraction.py — REBEL parser (pure function, no model needed)
# --------------------------------------------------------------------- #

def test_parse_rebel_triplets_single():
    output = "<triplet> Tim Cook <subj> Apple <obj> employer"
    triplets = _parse_rebel_triplets(output)
    assert triplets == [{"head": "Tim Cook", "type": "employer", "tail": "Apple"}]


def test_parse_rebel_triplets_multiple():
    output = (
        "<triplet> Tim Cook <subj> Apple <obj> employer "
        "<triplet> Apple <subj> California <obj> headquarters location"
    )
    triplets = _parse_rebel_triplets(output)
    assert len(triplets) == 2
    assert triplets[0]["head"] == "Tim Cook"
    assert triplets[1]["tail"] == "California"


def test_parse_rebel_triplets_empty_string():
    assert _parse_rebel_triplets("") == []


# --------------------------------------------------------------------- #
# relation_extraction.py — rule-based fallback (uses spaCy en_core_web_sm)
# --------------------------------------------------------------------- #

def test_rule_based_relation_extraction_finds_svo():
    extractor = RelationExtractor(use_rebel=False)
    relations = extractor.extract("Apple acquired a small startup.")
    assert len(relations) >= 1
    assert relations[0].head.lower().endswith("apple")
    assert relations[0].relation == "acquire"
    assert "startup" in relations[0].tail.lower()
    assert relations[0].source == "rule_based"


def test_relation_extraction_empty_text():
    extractor = RelationExtractor(use_rebel=False)
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []


# --------------------------------------------------------------------- #
# event_temporal.py
# --------------------------------------------------------------------- #

def test_event_extraction_links_nearest_time():
    extractor = EventTemporalExtractor()
    events = extractor.extract("The team met yesterday to discuss the roadmap.")
    assert len(events) >= 1
    met_events = [e for e in events if e.trigger_lemma == "meet"]
    assert met_events, "Expected a 'meet' event trigger"
    assert met_events[0].time_expression == "yesterday"


def test_event_extraction_no_time_expression():
    extractor = EventTemporalExtractor()
    events = extractor.extract("The engineers reviewed the code.")
    assert len(events) >= 1
    assert events[0].time_expression is None


def test_event_extraction_empty_text():
    extractor = EventTemporalExtractor()
    assert extractor.extract("") == []


# --------------------------------------------------------------------- #
# ner_module.py — English backend (spaCy)
# --------------------------------------------------------------------- #

def test_ner_english_finds_person_and_org():
    extractor = NERExtractor()
    entities = extractor.extract("Tim Cook works at Apple.", "English")
    labels = {e.label for e in entities}
    assert "PERSON" in labels
    assert "ORGANIZATION" in labels


def test_ner_empty_text_returns_empty_list():
    extractor = NERExtractor()
    assert extractor.extract("", "English") == []
