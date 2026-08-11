"""
ui_helpers.py
===============

Pure rendering/formatting helpers for the Streamlit app (`app.py`).
Kept separate from `app.py` so the HTML-building and dataframe-shaping
logic can be unit-tested without importing Streamlit itself, and so
`app.py` stays focused on page layout/control flow.
"""

import html
from typing import List

import pandas as pd

from backend.ner_module import Entity, LABEL_COLOR_MAP


def build_ner_highlighted_html(text: str, entities: List[Entity]) -> str:
    """
    Render `text` as HTML with each detected entity span wrapped in a
    coloured `<mark>` tag (colour keyed by `LABEL_COLOR_MAP`), for the
    "Color-coded highlights for Persons, Locations, Organizations, etc."
    requirement.

    Implementation note
    ---------------------
    Entities are sorted by start offset and inserted back-to-front-safe by
    walking the text left to right and copying the untouched gaps between
    entities verbatim (HTML-escaped) -- this avoids the classic bug of
    naive string-replace-based highlighting, which breaks when the same
    surface text appears both inside and outside an entity span (e.g. the
    word "Apple" as a fruit vs. the company).

    Parameters
    ----------
    text : str
        The plain text the entities were extracted from.
    entities : List[Entity]
        Entities with `start`/`end` character offsets into `text`.

    Returns
    -------
    str
        An HTML fragment safe to pass to `st.markdown(..., unsafe_allow_html=True)`.
    """
    if not entities:
        return f"<div style='line-height:1.9'>{html.escape(text)}</div>"

    sorted_entities = sorted(entities, key=lambda e: e.start)
    pieces = []
    cursor = 0
    for entity in sorted_entities:
        if entity.start < cursor:
            # Overlapping spans can occur when two different backends'
            # outputs are merged; skip an entity that starts before the
            # cursor rather than emit malformed nested <mark> tags.
            continue
        pieces.append(html.escape(text[cursor:entity.start]))
        color = LABEL_COLOR_MAP.get(entity.label, "#eeeeee")
        entity_text = html.escape(text[entity.start:entity.end])
        pieces.append(
            f"<mark style='background-color:{color}; border-radius:4px; padding:1px 3px;' "
            f"title='{entity.label} ({entity.confidence:.2f})'>{entity_text}"
            f"<sub style='font-size:0.65em; opacity:0.7;'> {entity.label}</sub></mark>"
        )
        cursor = entity.end
    pieces.append(html.escape(text[cursor:]))
    return f"<div style='line-height:2.4'>{''.join(pieces)}</div>"


def build_event_highlighted_html(text: str, events) -> str:
    """
    Render `text` with event triggers bolded/highlighted and linked
    temporal expressions underlined, for the "Highlighted event triggers
    paired with extracted temporal expressions" requirement.

    Simpler than `build_ner_highlighted_html` because events/temporal
    expressions are matched by substring search against the sentence they
    were extracted from (their offsets are sentence-relative, not
    document-relative, by design -- see `event_temporal.py`), so highlighting
    is done per-sentence rather than against one shared offset space.
    """
    if not events:
        return f"<div style='line-height:1.9'>{html.escape(text)}</div>"

    rendered_sentences = []
    seen_sentences = []
    for event in events:
        if event.sentence in seen_sentences:
            continue
        seen_sentences.append(event.sentence)

        sentence_events = [e for e in events if e.sentence == event.sentence]
        marked = html.escape(event.sentence)
        for e in sentence_events:
            trigger_escaped = html.escape(e.trigger_text)
            marked = marked.replace(
                trigger_escaped,
                f"<mark style='background-color:#ffd6a5; border-radius:4px; padding:1px 3px;' "
                f"title='event trigger'>{trigger_escaped}</mark>",
                1,
            )
            if e.time_expression:
                time_escaped = html.escape(e.time_expression)
                marked = marked.replace(
                    time_escaped,
                    f"<u style='text-decoration-color:#1f77b4; text-decoration-thickness:2px;' "
                    f"title='temporal expression'>{time_escaped}</u>",
                    1,
                )
        rendered_sentences.append(f"<p style='line-height:1.9'>{marked}</p>")

    return "".join(rendered_sentences)


def entities_to_dataframe(entities: List[Entity]) -> pd.DataFrame:
    """Convert extracted entities into a display-ready pandas DataFrame."""
    if not entities:
        return pd.DataFrame(columns=["Text", "Label", "Start", "End", "Confidence"])
    return pd.DataFrame(
        [
            {
                "Text": e.text,
                "Label": e.label,
                "Start": e.start,
                "End": e.end,
                "Confidence": round(e.confidence, 3),
            }
            for e in entities
        ]
    )


def relations_to_dataframe(relations) -> pd.DataFrame:
    """Convert extracted relations into a display-ready pandas DataFrame."""
    if not relations:
        return pd.DataFrame(columns=["Head", "Relation", "Tail", "Source"])
    return pd.DataFrame(
        [
            {"Head": r.head, "Relation": r.relation, "Tail": r.tail, "Source": r.source}
            for r in relations
        ]
    )


def events_to_dataframe(events) -> pd.DataFrame:
    """Convert extracted events into a display-ready pandas DataFrame."""
    if not events:
        return pd.DataFrame(columns=["Event Trigger", "Time Expression", "Normalized Time", "Sentence"])
    return pd.DataFrame(
        [
            {
                "Event Trigger": e.trigger_text,
                "Time Expression": e.time_expression or "—",
                "Normalized Time": e.normalized_time or "—",
                "Sentence": e.sentence,
            }
            for e in events
        ]
    )


def relations_to_graphviz(relations) -> str:
    """
    Build a Graphviz DOT string visualizing relations as a directed graph
    ``[Entity A] -- (Relation) --> [Entity B]``, for `st.graphviz_chart`.

    Node colours reuse `LABEL_COLOR_MAP` where a relation's head/tail could
    be linked to a known entity type (see
    `relation_extraction.link_relations_to_entities`), so the graph view
    stays visually consistent with the NER highlight colours.

    Parameters
    ----------
    relations : List[relation_extraction.Relation]

    Returns
    -------
    str
        A DOT-language graph description.
    """
    if not relations:
        return "digraph G { empty [label=\"No relations extracted\", shape=plaintext]; }"

    lines = ["digraph G {", '  rankdir="LR";', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']

    def _node_id(name: str) -> str:
        return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)[:40]

    seen_nodes = {}
    for r in relations:
        for span, label in [(r.head, r.head_entity_label), (r.tail, r.tail_entity_label)]:
            node_id = _node_id(span)
            if node_id in seen_nodes:
                continue
            color = LABEL_COLOR_MAP.get(label, "#f0f0f0")
            safe_label = span.replace('"', "'")
            lines.append(f'  {node_id} [label="{safe_label}", fillcolor="{color}"];')
            seen_nodes[node_id] = True

    for r in relations:
        head_id = _node_id(r.head)
        tail_id = _node_id(r.tail)
        safe_relation = r.relation.replace('"', "'")
        lines.append(f'  {head_id} -> {tail_id} [label="{safe_relation}", fontsize=10];')

    lines.append("}")
    return "\n".join(lines)
