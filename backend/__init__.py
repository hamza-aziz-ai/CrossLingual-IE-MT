"""
Backend package for the Cross-Lingual Information Extraction and Machine
Translation System (NLP Applications Assignment-2, Part 1).

Sub-modules
-----------
languages           : Registry of supported languages and their codes.
text_utils          : Shared text preprocessing/normalization helpers.
mt_engine           : Neural Machine Translation module (NLLB-200) + word alignment.
ner_module          : Named Entity Recognition (English + Indic languages).
relation_extraction : Relation Extraction (REBEL + rule-based fallback).
event_temporal      : Event trigger detection + temporal expression extraction/linking.
pipeline            : Orchestrates all modules into a single request/response object.
"""

__version__ = "1.0.0"
