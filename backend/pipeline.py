"""
pipeline.py
============

Orchestrates the Machine Translation, NER, Relation Extraction and
Event/Temporal modules into a single request/response object, and
provides the JSON/CSV export used by the Streamlit "download" button.

Architecture: Translate-then-Extract
--------------------------------------
As discussed in each sub-module's docstring and formally compared in the
Task-B document, this pipeline always runs Information Extraction on the
**English** side of the translation pair:

    Indic input  --[MT: Indic -> English]--> English text --[NER+RE+Event]--> IE results
    English input --[MT: English -> Indic]--> Indic text (display only)
                   \\-----------------------[NER+RE+Event]--> IE results  (on the original English)

This keeps every IE sub-module on the language where the strongest,
best-documented tooling exists (spaCy's full pipeline, REBEL, dependency
parsing), at the cost of exposing translation errors to the IE stage
(error propagation) -- a trade-off analyzed quantitatively/qualitatively
in Task B.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import csv
import io
import json

from .languages import get_language
from .mt_engine import MTEngine, TranslationResult
from .ner_module import NERExtractor, Entity
from .relation_extraction import RelationExtractor, Relation, link_relations_to_entities
from .event_temporal import EventTemporalExtractor, Event


@dataclass
class ModuleLatencies:
    """
    Per-module wall-clock timings (seconds), captured for the empirical
    half of the Task-B "Scalability & Inference Optimization" discussion
    (e.g. to justify *which* module benefits most from quantization).
    """

    translation: float = 0.0
    ner: float = 0.0
    relation_extraction: float = 0.0
    event_extraction: float = 0.0

    @property
    def total(self) -> float:
        return self.translation + self.ner + self.relation_extraction + self.event_extraction


@dataclass
class PipelineResult:
    """
    The single object returned by `CrossLingualPipeline.run()`, containing
    everything the Streamlit UI needs to render every view (Translation,
    Alignment, NER, Relations, Events) plus everything the JSON/CSV export
    needs.
    """

    original_text: str
    source_language: str
    target_language: str
    translated_text: str
    translation_result: TranslationResult
    ie_source_language: str  # language the IE modules actually ran on (always "English" in this pipeline)
    ie_source_text: str      # the English text IE ran on (== original_text if source_language == "English")
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    latencies: ModuleLatencies = field(default_factory=ModuleLatencies)

    def to_export_dict(self) -> dict:
        """
        Build the JSON-serializable "IE knowledge base" export required by
        the assignment brief ("download button to export the translated
        text and the extracted IE knowledge base in JSON or CSV format").
        """
        return {
            "original_text": self.original_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "translated_text": self.translated_text,
            "ie_source_language": self.ie_source_language,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "events": [ev.to_dict() for ev in self.events],
            "latencies_seconds": asdict(self.latencies),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_export_dict(), indent=2, ensure_ascii=False).encode("utf-8")

    def to_csv_bytes(self) -> bytes:
        """
        Flatten entities, relations and events into one long-format CSV
        (a `record_type` column disambiguates rows), since a knowledge
        base with three different schemas cannot be a single flat table
        without either padding columns or a long-format layout. Long
        format was chosen so the CSV opens cleanly in Excel/Sheets without
        dozens of mostly-empty cells.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["record_type", "field_1", "field_2", "field_3", "field_4", "field_5"])

        writer.writerow(["translation", self.source_language, self.target_language, self.translated_text, "", ""])

        for e in self.entities:
            writer.writerow(["entity", e.text, e.label, e.start, e.end, e.confidence])

        for r in self.relations:
            writer.writerow(["relation", r.head, r.relation, r.tail, r.source, ""])

        for ev in self.events:
            writer.writerow(["event", ev.trigger_text, ev.time_expression or "", ev.normalized_time or "", ev.sentence, ""])

        return buffer.getvalue().encode("utf-8")


class CrossLingualPipeline:
    """
    High-level facade combining `MTEngine`, `NERExtractor`,
    `RelationExtractor` and `EventTemporalExtractor` behind one `run()`
    call. This is the object the Streamlit app constructs once (via
    `st.cache_resource`) and reuses across requests, so each individual
    module's lazy model loading only happens the first time that module is
    actually needed.
    """

    def __init__(self):
        self.mt_engine = MTEngine()
        self.ner_extractor = NERExtractor()
        self.relation_extractor = RelationExtractor()
        self.event_extractor = EventTemporalExtractor()

    def run(self, text: str, source_language: str, target_language: str) -> PipelineResult:
        """
        Run the full Translate-then-Extract pipeline on one input.

        Parameters
        ----------
        text : str
            Raw user input.
        source_language : str
            Display name, e.g. "English" or "Hindi".
        target_language : str
            Display name of the desired translation output language.

        Returns
        -------
        PipelineResult
        """
        latencies = ModuleLatencies()

        # ---- 1. Machine Translation -----------------------------------
        t0 = time.perf_counter()
        translation_result = self.mt_engine.translate(text, source_language, target_language)
        latencies.translation = time.perf_counter() - t0

        # ---- 2. Decide which text Information Extraction runs on -------
        # English-pivot: if the user's source language is already English,
        # run IE on the original (untranslated) text to avoid *adding*
        # translation-induced error into the IE stage unnecessarily; if the
        # source is Indic, IE must run on the English translation, since
        # that is the side with mature NER/RE/Event tooling (see module
        # docstrings for the full rationale).
        source_lang_info = get_language(source_language)
        if source_lang_info.is_indic:
            ie_text = translation_result.translated_text
        else:
            ie_text = translation_result.source_text

        # ---- 3. Named Entity Recognition -------------------------------
        t0 = time.perf_counter()
        entities = self.ner_extractor.extract(ie_text, "English")
        latencies.ner = time.perf_counter() - t0

        # ---- 4. Relation Extraction ------------------------------------
        t0 = time.perf_counter()
        relations = self.relation_extractor.extract(ie_text)
        relations = link_relations_to_entities(relations, entities)
        latencies.relation_extraction = time.perf_counter() - t0

        # ---- 5. Event & Temporal Extraction -----------------------------
        t0 = time.perf_counter()
        events = self.event_extractor.extract(ie_text)
        latencies.event_extraction = time.perf_counter() - t0

        return PipelineResult(
            original_text=translation_result.source_text,
            source_language=source_language,
            target_language=target_language,
            translated_text=translation_result.translated_text,
            translation_result=translation_result,
            ie_source_language="English",
            ie_source_text=ie_text,
            entities=entities,
            relations=relations,
            events=events,
            latencies=latencies,
        )
