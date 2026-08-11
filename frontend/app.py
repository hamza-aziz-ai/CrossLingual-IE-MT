"""
app.py
=======

Streamlit web dashboard for the Cross-Lingual Information Extraction and
Machine Translation System (NLP Applications Assignment-2, Part 1,
Section 1 "Web Interface").

Run with:
    streamlit run frontend/app.py

Layout
------
Sidebar   : source/target language selectors + IE component toggles.
Main area : text input + "Translate & Extract" button, then four tabs:
            Translation & Alignment | Named Entities | Relations |
            Events & Time, plus an Export panel with JSON/CSV downloads.

Streamlit-specific design notes
----------------------------------
* `st.cache_resource` wraps `get_pipeline()` so the (potentially several
  GB of) MT/NER/RE models are loaded into memory once per server process,
  not once per user interaction -- without this, every button click would
  re-instantiate `CrossLingualPipeline`, and every module's *lazy* loading
  inside would still re-trigger on first use each time.
* `st.session_state` holds the last `PipelineResult` so that toggling a
  sidebar checkbox (show/hide alignment, NER, etc.) re-renders instantly
  from the already-computed result instead of re-running the whole
  MT+IE pipeline.
"""

import sys
from pathlib import Path

# Allow `import backend...` when Streamlit is launched as
# `streamlit run frontend/app.py` from the project root (Streamlit does
# not add the project root to sys.path automatically the way a package
# install would).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from backend.languages import SUPPORTED_LANGUAGES
from backend.ner_module import LABEL_COLOR_MAP
from backend.fonts import apply_script_aware_fonts
from backend.pipeline import CrossLingualPipeline
from backend.text_utils import detect_script_language, truncate_for_display, format_subword_token_for_display
from frontend.ui_helpers import (
    build_event_highlighted_html,
    build_ner_highlighted_html,
    entities_to_dataframe,
    events_to_dataframe,
    relations_to_dataframe,
    relations_to_graphviz,
)

st.set_page_config(
    page_title="Cross-Lingual IE + MT System",
    page_icon="🌐",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> CrossLingualPipeline:
    """
    Construct the shared `CrossLingualPipeline` exactly once per Streamlit
    server process. `st.cache_resource` (rather than `st.cache_data`) is
    used because the cached object holds live model weights / GPU tensors,
    which must not be pickled/copied the way `st.cache_data` would.
    """
    return CrossLingualPipeline()


def render_sidebar():
    """
    Render the sidebar language pickers and IE-view toggle controls.

    Returns
    -------
    dict
        {"source_language": str, "target_language": str,
         "show_alignment": bool, "show_ner": bool,
         "show_relations": bool, "show_events": bool}
    """
    st.sidebar.header("⚙️ Configuration")

    language_names = list(SUPPORTED_LANGUAGES.keys())

    default_source_idx = language_names.index(st.session_state.get("last_source_language", "English"))
    source_language = st.sidebar.selectbox(
        "Source language", language_names, index=default_source_idx, key="source_language_select"
    )

    target_options = [name for name in language_names if name != source_language]
    default_target = st.session_state.get("last_target_language", "Hindi")
    if default_target not in target_options:
        default_target = target_options[0]
    target_language = st.sidebar.selectbox(
        "Target language", target_options, index=target_options.index(default_target), key="target_language_select"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Show / Hide")
    show_alignment = st.sidebar.checkbox("Word alignment heatmap", value=True)
    show_ner = st.sidebar.checkbox("Named Entities (NER)", value=True)
    show_relations = st.sidebar.checkbox("Relation Extraction", value=True)
    show_events = st.sidebar.checkbox("Events & Temporal Expressions", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Models: facebook/nllb-200-distilled-600M (MT) · spaCy en_core_web_sm + "
        "ai4bharat/IndicNER (NER) · Babelscape/rebel-large + rule-based fallback (RE) · "
        "spaCy dependency parse + dateparser (Events/Time)."
    )

    return {
        "source_language": source_language,
        "target_language": target_language,
        "show_alignment": show_alignment,
        "show_ner": show_ner,
        "show_relations": show_relations,
        "show_events": show_events,
    }


def render_alignment_heatmap(alignment_result):
    """
    Render one sentence's source-x-target cross-attention matrix as a
    matplotlib heatmap (the "visual word-alignment table/heatmap or
    cross-lingual attention visualization" requirement).
    """
    matrix = alignment_result.attention_matrix
    if matrix.size == 0:
        st.info("No alignment data available for this sentence.")
        return

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(alignment_result.source_tokens)),
                                     max(4, 0.4 * len(alignment_result.target_tokens))))
    display_source_tokens = [format_subword_token_for_display(t) for t in alignment_result.source_tokens]
    display_target_tokens = [format_subword_token_for_display(t) for t in alignment_result.target_tokens]

    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(display_source_tokens)))
    ax.set_xticklabels(display_source_tokens, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(display_target_tokens)))
    ax.set_yticklabels(display_target_tokens, fontsize=8)
    # Devanagari/Tamil/Bengali sub-word tokens need an explicit bundled
    # font -- matplotlib's default font has no glyphs for these scripts
    # (see backend/fonts.py docstring).
    apply_script_aware_fonts(ax.get_xticklabels(), display_source_tokens)
    apply_script_aware_fonts(ax.get_yticklabels(), display_target_tokens)
    ax.set_xlabel("Source sub-word tokens")
    ax.set_ylabel("Target sub-word tokens")
    ax.set_title("Cross-attention word alignment (decoder → encoder)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_alignment_table(alignment_result):
    """Render the compact hard-alignment table: one best-matching source token per target token."""
    rows = []
    for tgt_idx, src_idx in enumerate(alignment_result.hard_alignment):
        if tgt_idx >= len(alignment_result.target_tokens):
            continue
        src_idx = int(src_idx)
        source_token = (
            alignment_result.source_tokens[src_idx] if src_idx < len(alignment_result.source_tokens) else "—"
        )
        rows.append({
            "Target token": format_subword_token_for_display(alignment_result.target_tokens[tgt_idx]),
            "Aligned source token": format_subword_token_for_display(source_token),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    st.title("🌐 Cross-Lingual Information Extraction & Machine Translation System")
    st.caption(
        "NLP Applications (S2-25_AIMLCZG519) — Assignment 2, Part 1, Task A. "
        "Translate between English and an Indic language while extracting Named "
        "Entities, Relations, and Temporal Events."
    )

    config = render_sidebar()

    default_text = st.session_state.get("input_text", "")
    input_text = st.text_area(
        "Enter source text",
        value=default_text,
        height=140,
        placeholder="Type or paste text in English or the selected source language...",
        key="input_text_area",
    )

    detected = detect_script_language(input_text) if input_text else None
    if detected and detected != config["source_language"] and input_text.strip():
        st.caption(f"🔍 Script-based auto-detection suggests this text may be **{detected}** — "
                   f"double-check the *Source language* dropdown if that looks wrong.")

    run_clicked = st.button("🚀 Translate & Extract", type="primary")

    if run_clicked:
        if not input_text or not input_text.strip():
            st.warning("Please enter some text first.")
            return

        text_to_process = truncate_for_display(input_text)
        if text_to_process != input_text:
            st.warning("Input was truncated to the first ~4000 characters before processing.")

        pipeline = get_pipeline()
        with st.spinner("Running translation + information extraction pipeline... "
                         "(first run downloads model weights and can take several minutes)"):
            try:
                result = pipeline.run(text_to_process, config["source_language"], config["target_language"])
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

        st.session_state["pipeline_result"] = result
        st.session_state["last_source_language"] = config["source_language"]
        st.session_state["last_target_language"] = config["target_language"]

    result = st.session_state.get("pipeline_result")
    if result is None:
        st.info("Enter text above and click **Translate & Extract** to run the pipeline.")
        return

    tab_translation, tab_ner, tab_relations, tab_events, tab_export = st.tabs(
        ["🔤 Translation & Alignment", "🏷️ Named Entities", "🔗 Relations", "📅 Events & Time", "⬇️ Export"]
    )

    with tab_translation:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Source ({result.source_language})")
            st.write(result.original_text)
        with col2:
            st.subheader(f"Translation ({result.target_language})")
            st.write(result.translated_text)

        if result.translation_result.truncated:
            st.warning("One or more sentences exceeded the model's token limit and were truncated.")

        if config["show_alignment"]:
            st.markdown("---")
            st.subheader("Word Alignment")
            sentence_options = [f"Sentence {i + 1}" for i in range(len(result.translation_result.sentence_alignments))]
            if sentence_options:
                selected = st.selectbox("Choose a sentence to visualize", sentence_options, key="alignment_sentence_select")
                idx = sentence_options.index(selected)
                alignment_result = result.translation_result.sentence_alignments[idx]
                heat_tab, table_tab = st.tabs(["Heatmap", "Alignment table"])
                with heat_tab:
                    render_alignment_heatmap(alignment_result)
                with table_tab:
                    render_alignment_table(alignment_result)

    with tab_ner:
        if config["show_ner"]:
            st.subheader(f"Named Entities (extracted from the {result.ie_source_language} text)")
            legend_html = " &nbsp; ".join(
                f"<span style='background-color:{color}; padding:2px 8px; border-radius:4px;'>{label}</span>"
                for label, color in LABEL_COLOR_MAP.items()
            )
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("　")
            st.markdown(build_ner_highlighted_html(result.ie_source_text, result.entities), unsafe_allow_html=True)
            st.markdown("---")
            st.dataframe(entities_to_dataframe(result.entities), use_container_width=True, hide_index=True)
        else:
            st.info("Named Entities view is hidden — enable it from the sidebar.")

    with tab_relations:
        if config["show_relations"]:
            st.subheader("Extracted Relations")
            if result.relations:
                backend_used = result.relations[0].source
                st.caption(f"Backend used: **{backend_used}** "
                           f"({'REBEL semantic relation extraction' if backend_used == 'rebel' else 'rule-based dependency-parse fallback'})")
            st.graphviz_chart(relations_to_graphviz(result.relations))
            st.dataframe(relations_to_dataframe(result.relations), use_container_width=True, hide_index=True)
        else:
            st.info("Relations view is hidden — enable it from the sidebar.")

    with tab_events:
        if config["show_events"]:
            st.subheader("Events & Temporal Expressions")
            st.markdown(build_event_highlighted_html(result.ie_source_text, result.events), unsafe_allow_html=True)
            st.markdown("---")
            st.dataframe(events_to_dataframe(result.events), use_container_width=True, hide_index=True)
        else:
            st.info("Events & Time view is hidden — enable it from the sidebar.")

    with tab_export:
        st.subheader("Download results")
        st.caption("Exports the translation plus the full extracted IE knowledge base "
                   "(entities, relations, events).")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download JSON",
                data=result.to_json_bytes(),
                file_name="cross_lingual_ie_export.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "⬇️ Download CSV",
                data=result.to_csv_bytes(),
                file_name="cross_lingual_ie_export.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("Module latencies (this run)")
        st.dataframe(
            [
                {"Module": "Translation", "Seconds": round(result.latencies.translation, 3)},
                {"Module": "NER", "Seconds": round(result.latencies.ner, 3)},
                {"Module": "Relation Extraction", "Seconds": round(result.latencies.relation_extraction, 3)},
                {"Module": "Event/Temporal Extraction", "Seconds": round(result.latencies.event_extraction, 3)},
                {"Module": "TOTAL", "Seconds": round(result.latencies.total, 3)},
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
