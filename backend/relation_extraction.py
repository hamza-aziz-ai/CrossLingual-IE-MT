"""
relation_extraction.py
========================

Relation Extraction module: identifies and classifies semantic
relationships between entities, e.g. ``[Barack Obama] -- (born_in) -->
[Honolulu]``.

Two backends, primary + fallback
-----------------------------------
1. **Primary -- REBEL** (`Babelscape/rebel-large`): a BART-based
   sequence-to-sequence model fine-tuned to directly generate
   ``<triplet> head <subj> tail <obj> relation`` formatted strings from raw
   text, covering 220+ Wikidata relation types. This gives genuinely
   *semantic* relation labels (e.g. "employer", "place of birth", "spouse")
   rather than just "the verb between two entities."
2. **Fallback -- rule-based Subject-Verb-Object (SVO) extraction** via
   spaCy dependency parsing. If REBEL cannot be loaded (offline machine,
   insufficient disk/RAM, or the download fails) the pipeline still
   produces relation triples, at the cost of coarser, syntax-derived
   relation labels (the verb lemma itself, e.g. "acquire" rather than
   "acquirer_of"). This keeps the web app functional end-to-end even in a
   constrained environment, which matters for a group project graded on a
   live demo.

Both backends are run only on the English-side text, per the
English-pivot design documented in `ner_module.py`.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Relation:
    """
    One extracted (head, relation, tail) triple.

    Attributes
    ----------
    head : str
        Surface text of the subject/head entity, e.g. "Barack Obama".
    relation : str
        Relation label. REBEL produces a Wikidata-style relation phrase
        (e.g. "place of birth"); the rule-based fallback produces a verb
        lemma (e.g. "born").
    tail : str
        Surface text of the object/tail entity, e.g. "Honolulu".
    source : str
        Which backend produced this triple: "rebel" or "rule_based". Shown
        in the UI so the report/demo can be transparent about provenance.
    head_entity_label : Optional[str]
        Unified NER label of `head` if it could be matched to a detected
        entity span (see `link_relations_to_entities`), else None.
    tail_entity_label : Optional[str]
        Same, for `tail`.
    """

    head: str
    relation: str
    tail: str
    source: str
    head_entity_label: Optional[str] = None
    tail_entity_label: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "source": self.source,
            "head_entity_label": self.head_entity_label,
            "tail_entity_label": self.tail_entity_label,
        }


# Light verbs / auxiliaries that make poor relation labels on their own
# (e.g. "is", "was", "has") when they end up as a dependency ROOT with a
# nominal complement -- these are still kept if no better verb exists, but
# are de-prioritized versus content verbs during fallback extraction.
_COPULA_LEMMAS = {"be", "have", "do"}


def _parse_rebel_triplets(generated_text: str) -> List[dict]:
    """
    Parse REBEL's special ``<triplet>/<subj>/<obj>`` token format into a
    list of ``{"head":, "type":, "tail":}`` dicts.

    This is the reference parsing routine published in the REBEL model
    card (Babelscape/rebel-large), reproduced here verbatim (module-level
    naming adapted to this file's style) because the model's raw output is
    not valid to parse any other way -- it is a flat token stream, not
    JSON/XML.

    Parameters
    ----------
    generated_text : str
        Raw decoded output of the REBEL seq2seq model for one input text.

    Returns
    -------
    List[dict]
        Each dict has keys "head", "type" (the relation), and "tail".
    """
    triplets = []
    relation, subject, object_ = "", "", ""
    text = generated_text.strip()
    current = "x"
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
    return triplets


class RelationExtractor:
    """
    Lazily-loaded relation extractor with an automatic REBEL -> rule-based
    fallback chain.
    """

    def __init__(
        self,
        rebel_model_name: str = "Babelscape/rebel-large",
        use_rebel: bool = True,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        device : Optional[str]
            "cuda", "cpu", or None to auto-select (CUDA if available, else
            CPU) -- same convention as `MTEngine`, for the same reason (the
            group's laptops vary in GPU availability).
        """
        self.rebel_model_name = rebel_model_name
        self.use_rebel = use_rebel
        self._device = device
        self._rebel_tokenizer = None
        self._rebel_model = None
        self._rebel_load_failed = False
        self._spacy_nlp = None

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

    def _ensure_rebel_loaded(self) -> bool:
        """
        Attempt to load the REBEL pipeline once. Returns True on success,
        False if loading failed (e.g. no internet / insufficient disk on
        first run), in which case the extractor permanently falls back to
        the rule-based backend for the remainder of the process lifetime
        (re-attempting a multi-GB download on every request would make a
        broken environment unusably slow rather than just degraded).
        """
        if self._rebel_model is not None:
            return True
        if self._rebel_load_failed or not self.use_rebel:
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._rebel_tokenizer = AutoTokenizer.from_pretrained(self.rebel_model_name)
            self._rebel_model = AutoModelForSeq2SeqLM.from_pretrained(self.rebel_model_name)
            self._rebel_model.to(self._device)
            self._rebel_model.eval()
            return True
        except Exception:
            # Broad except is intentional here: `transformers` can raise many
            # different exception types depending on *why* loading failed
            # (network error, disk quota, corrupt cache, missing optional
            # dependency); all of them should trigger the same fallback
            # behaviour rather than crashing the request.
            self._rebel_load_failed = True
            return False

    def _extract_with_rebel(self, text: str) -> List[Relation]:
        """
        Run REBEL via raw model.generate() rather than the `transformers`
        text2text-generation `pipeline()` helper. The pipeline's postprocess
        step hardcodes `skip_special_tokens=True`, which strips REBEL's
        `<triplet>/<subj>/<obj>` markup tokens that `_parse_rebel_triplets`
        depends on -- silently producing zero triplets for every input. The
        model card's documented usage (decode with
        `skip_special_tokens=False`) is reproduced here instead.
        """
        import torch

        inputs = self._rebel_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(
            self._device
        )
        with torch.no_grad():
            generated = self._rebel_model.generate(
                **inputs, max_length=256, length_penalty=0, num_beams=3
            )
        generated_text = self._rebel_tokenizer.decode(generated[0], skip_special_tokens=False)
        raw_triplets = _parse_rebel_triplets(generated_text)
        return [
            Relation(head=t["head"], relation=t["type"], tail=t["tail"], source="rebel")
            for t in raw_triplets
            if t["head"] and t["tail"]
        ]

    def _extract_with_rules(self, text: str) -> List[Relation]:
        """
        Dependency-parse-based Subject-Verb-Object extraction, used as the
        offline-safe fallback.

        Algorithm
        ---------
        For every sentence, for every token whose dependency label is a
        verbal ROOT or conjunct of the root (`dep_ in {"ROOT", "conj"}` and
        `pos_ == "VERB"`):
          1. Subject  = the first child with `dep_ in {"nsubj", "nsubjpass"}`.
          2. Object   = the first child with `dep_ in {"dobj", "attr", "oprd"}`,
             or -- if the verb takes a prepositional object instead (e.g.
             "moved to Delhi") -- the object of its first `prep` child.
          3. If both a subject and an object are found, emit a Relation
             with the verb's lemma as the relation label.

        This mirrors the classical "Stanford OpenIE"-style SVO heuristic
        taught in NLP courses -- explainable and dependency-free of any
        extra model download, at the cost of recall (only surface SVO
        patterns are caught) and relation-label granularity (a lemma, not
        a semantic relation type).
        """
        self._ensure_spacy_loaded()
        doc = self._spacy_nlp(text)
        relations: List[Relation] = []

        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB" or token.dep_ not in ("ROOT", "conj"):
                    continue

                subject_token = next(
                    (child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")), None
                )
                if subject_token is None:
                    continue

                object_token = next(
                    (child for child in token.children if child.dep_ in ("dobj", "attr", "oprd")), None
                )
                if object_token is None:
                    prep_token = next((child for child in token.children if child.dep_ == "prep"), None)
                    if prep_token is not None:
                        object_token = next(
                            (child for child in prep_token.children if child.dep_ == "pobj"), None
                        )

                if object_token is None:
                    continue

                # Expand subject/object tokens to their full noun-phrase span
                # (e.g. "the CEO" rather than just "CEO") using spaCy's
                # dependency subtree, restricted to a contiguous chunk to
                # avoid pulling in unrelated clauses.
                subject_span = doc[subject_token.left_edge.i: subject_token.right_edge.i + 1]
                object_span = doc[object_token.left_edge.i: object_token.right_edge.i + 1]

                relations.append(
                    Relation(
                        head=subject_span.text,
                        relation=token.lemma_,
                        tail=object_span.text,
                        source="rule_based",
                    )
                )

        return relations

    def extract(self, text: str) -> List[Relation]:
        """
        Extract relation triples from `text` (assumed to be English --
        callers pass the translated-to-English text for Indic inputs; see
        `pipeline.py`).

        Tries REBEL first; transparently falls back to the rule-based
        extractor if REBEL cannot be loaded. Both backends can be combined
        by a caller that wants maximum recall, but by default only the
        primary available backend's output is returned to keep the UI's
        relation table free of near-duplicate triples from two different
        labeling schemes.

        Parameters
        ----------
        text : str
            English text to analyze.

        Returns
        -------
        List[Relation]
        """
        if not text or not text.strip():
            return []

        if self._ensure_rebel_loaded():
            try:
                return self._extract_with_rebel(text)
            except Exception:
                # A load-time success does not guarantee every inference call
                # succeeds (e.g. transient CUDA OOM on a very long input);
                # fall back per-call rather than crashing the request.
                return self._extract_with_rules(text)

        return self._extract_with_rules(text)


def link_relations_to_entities(relations: List[Relation], entities) -> List[Relation]:
    """
    Annotate each relation's head/tail with the unified NER label of the
    best-matching detected entity, so the frontend's relation graph can
    colour nodes consistently with the NER view.

    Matching heuristic
    -------------------
    Case-insensitive substring match in both directions (relation span
    contains the entity text, or vice versa) -- REBEL and the rule-based
    extractor do not always return spans with identical boundaries to the
    NER model's spans (e.g. "the President" vs. "President"), so exact
    string equality would miss most real matches.

    Parameters
    ----------
    relations : List[Relation]
    entities : List[ner_module.Entity]

    Returns
    -------
    List[Relation]
        The same `Relation` objects, mutated in place with
        `head_entity_label` / `tail_entity_label` populated where a match
        was found, and also returned for convenience.
    """

    def _best_label_for(span_text: str) -> Optional[str]:
        span_lower = span_text.lower().strip()
        if not span_lower:
            return None
        for entity in entities:
            entity_lower = entity.text.lower().strip()
            if entity_lower and (entity_lower in span_lower or span_lower in entity_lower):
                return entity.label
        return None

    for relation in relations:
        relation.head_entity_label = _best_label_for(relation.head)
        relation.tail_entity_label = _best_label_for(relation.tail)

    return relations
