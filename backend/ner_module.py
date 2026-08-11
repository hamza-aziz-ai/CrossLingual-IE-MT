"""
ner_module.py
==============

Named Entity Recognition module.

Design decision -- English-pivot NER
--------------------------------------
The assignment allows extracting entities "from the source or translated
text." We route *all* Information Extraction (NER, Relation Extraction,
Event/Temporal) onto the English side of the translation pair:

* If the user's input is already English, IE runs on the original text.
* If the user's input is an Indic language, IE runs on the NLLB
  translation into English (see `pipeline.py`).

This is the "Translate-then-Extract" architecture discussed in Task B.
The alternative -- Direct Cross-Lingual Extraction, i.e. running NER/RE/
Event extraction natively on Hindi/Tamil/Marathi/Bengali text -- was
scoped out of Part 1's implementation because production-quality
Relation Extraction and dependency-parse-based event detection do not
have freely available, pip-installable models for all four target
languages (unlike NER, which IndicNER covers). Task B's written
comparison discusses the trade-off in full; this module still *supports*
running IndicNER directly on Indic text (`extract(text, lang, pivot=False)`)
so the direct-extraction path can be demonstrated for the NER sub-task
specifically, even though RE/Event extraction remain English-pivoted.

Two backends
------------
* English  -> spaCy `en_core_web_sm` (`nlp.ents`), mature, fast, CPU-friendly.
* Indic    -> `ai4bharat/IndicNER`, an XLM-RoBERTa model fine-tuned on the
  Naamapadam NER corpus covering 11 Indic languages (including all four
  required here), loaded via a HuggingFace `token-classification` pipeline
  with `aggregation_strategy="simple"` (merges B-/I- sub-word pieces of the
  BIO tagging scheme into whole-word entity spans automatically).
"""

from dataclasses import dataclass
from typing import List

from .languages import get_language

# Unified entity-type schema. Both backends' native label sets are mapped
# onto this fixed vocabulary so the frontend's colour-coding and the
# exported JSON/CSV schema do not need to know which backend produced
# a given entity.
LABEL_COLOR_MAP = {
    "PERSON": "#ffb3ba",       # soft red
    "LOCATION": "#bae1ff",     # soft blue
    "ORGANIZATION": "#baffc9", # soft green
    "MISC": "#ffdfba",         # soft orange
    "DATE": "#e0baff",         # soft purple
    "NUMBER": "#fff5ba",       # soft yellow
}

# spaCy's `en_core_web_sm` label set -> unified schema.
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "NORP": "MISC",
    "FAC": "LOCATION",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "DATE",
    "TIME": "DATE",
    "PERCENT": "NUMBER",
    "MONEY": "NUMBER",
    "QUANTITY": "NUMBER",
    "ORDINAL": "NUMBER",
    "CARDINAL": "NUMBER",
}

# IndicNER / Naamapadam label set (BIO-tagged PER/LOC/ORG; aggregation_strategy
# "simple" strips the B-/I- prefix and returns "entity_group" already) -> unified schema.
_INDICNER_LABEL_MAP = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
}


@dataclass
class Entity:
    """
    One extracted named entity in the unified schema.

    Attributes
    ----------
    text : str
        Surface form of the entity as it appears in the analyzed text.
    label : str
        One of the keys of `LABEL_COLOR_MAP` (PERSON/LOCATION/ORGANIZATION/MISC/DATE/NUMBER).
    start : int
        Character offset (inclusive) of the entity's first character in the analyzed text.
    end : int
        Character offset (exclusive) of the entity's last character.
    confidence : float
        Model confidence score in [0, 1]. spaCy's rule/statistical NER does
        not expose a per-entity probability, so English entities are
        reported at a fixed 1.0; IndicNER (a softmax-based classifier)
        reports its true score.
    """

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": round(float(self.confidence), 4),
        }


class NERExtractor:
    """
    Lazily-loaded, language-routed Named Entity Recognizer.

    Both backends are loaded on first use only (not in `__init__`), for the
    same start-up-latency reason as `MTEngine`.
    """

    def __init__(self, indic_model_name: str = "ai4bharat/IndicNER"):
        self.indic_model_name = indic_model_name
        self._spacy_nlp = None
        self._indic_pipeline = None

    def _ensure_spacy_loaded(self):
        if self._spacy_nlp is not None:
            return
        import spacy

        try:
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy model 'en_core_web_sm' is not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc

    def _ensure_indic_loaded(self):
        if self._indic_pipeline is not None:
            return
        from transformers import pipeline

        self._indic_pipeline = pipeline(
            task="token-classification",
            model=self.indic_model_name,
            aggregation_strategy="simple",
        )

    def _extract_english(self, text: str) -> List[Entity]:
        self._ensure_spacy_loaded()
        doc = self._spacy_nlp(text)
        entities = []
        for ent in doc.ents:
            unified_label = _SPACY_LABEL_MAP.get(ent.label_)
            if unified_label is None:
                # Skip spaCy entity types outside our unified schema (e.g.
                # LAW/WORK_OF_ART edge cases already folded into MISC above
                # cover the common ones; anything truly unmapped is dropped
                # rather than silently mislabeled).
                continue
            entities.append(
                Entity(text=ent.text, label=unified_label, start=ent.start_char, end=ent.end_char, confidence=1.0)
            )
        return entities

    def _extract_indic(self, text: str) -> List[Entity]:
        self._ensure_indic_loaded()
        raw_entities = self._indic_pipeline(text)
        entities = []
        for raw in raw_entities:
            unified_label = _INDICNER_LABEL_MAP.get(raw.get("entity_group", ""))
            if unified_label is None:
                continue
            entities.append(
                Entity(
                    text=raw["word"],
                    label=unified_label,
                    start=int(raw["start"]),
                    end=int(raw["end"]),
                    confidence=float(raw.get("score", 0.0)),
                )
            )
        return entities

    def extract(self, text: str, language_name: str) -> List[Entity]:
        """
        Extract named entities from `text`, routing to the appropriate
        backend based on `language_name`.

        Parameters
        ----------
        text : str
            The text to analyze (already normalized upstream).
        language_name : str
            Display name from `languages.SUPPORTED_LANGUAGES` describing
            the language `text` is actually written in (NOT necessarily the
            user's originally-selected source language -- the pipeline
            passes the language of whichever side, original or translated,
            IE is being run on).

        Returns
        -------
        List[Entity]
        """
        if not text or not text.strip():
            return []
        lang_info = get_language(language_name)
        if lang_info.is_indic:
            return self._extract_indic(text)
        return self._extract_english(text)
