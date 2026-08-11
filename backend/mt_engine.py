"""
mt_engine.py
=============

Neural Machine Translation module built on Meta's NLLB-200
(``facebook/nllb-200-distilled-600M``) via HuggingFace Transformers, as
permitted by the assignment brief ("Neural/Indic MT using frameworks like
IndicTrans / HuggingFace Transformers").

Why NLLB-200 and not IndicTrans2 or per-pair OPUS-MT models
--------------------------------------------------------------
1. Single checkpoint, all four required languages, both directions:
   NLLB-200 was trained on 200 languages including all four Indic
   languages in the brief (Hindi, Tamil, Marathi, Bengali) and English,
   in every direction, from one set of weights. IndicTrans2 needs two
   *separate* checkpoints (en->indic and indic->en) plus the
   ``IndicTransToolkit`` preprocessing package (sentence normalization via
   IndicNLP + FLORES tagging + BPE via a custom tokenizer with
   ``trust_remote_code=True``); OPUS-MT (Helsinki-NLP) ships a different
   checkpoint *per language pair*, and does not have public en<->mr or
   en<->ta checkpoints. NLLB-200 is therefore the option that most
   directly satisfies "translate between English and an Indic language"
   for an arbitrary one of the four languages without multiplying the
   number of models the group has to download, version and keep in sync.
2. Standard `AutoModelForSeq2SeqLM` / `AutoTokenizer` API (no custom
   remote code), which keeps the dependency surface small and the code
   auditable.
3. `output_attentions=True` on `.generate()` exposes encoder-decoder
   cross-attention tensors directly, which is what Section 1's "visual
   word-alignment table/heatmap or cross-lingual attention visualization"
   requirement needs -- no separate aligner (fast_align/SimAlign) required.

Trade-off (documented further in the System Report / Task-B document):
NLLB-200-distilled-600M is heavier (≈2.4 GB fp32) than a single OPUS-MT
pair (~300 MB) and slightly behind IndicTrans2 on Indic-specific BLEU,
in exchange for the one-model-does-everything simplicity above. Task B
discusses quantization/distillation to shrink this for deployment.
"""

import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .languages import LanguageInfo, get_language
from .text_utils import normalize_text, split_sentences

# Model identifier. Swap to "facebook/nllb-200-1.3B" for higher quality at
# the cost of ~3x the memory/latency, or to a 4-bit quantized variant for
# the deployment scenario discussed in the Task-B optimization plan.
DEFAULT_MT_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB's tokenizer truncates beyond this many sub-word tokens; sentences
# longer than this are still sent through (HF truncates safely) but a
# warning is attached to the result so the UI can surface it.
MAX_SOURCE_TOKENS = 200


@dataclass
class AlignmentResult:
    """
    Word-level alignment between one source sentence and its translation,
    derived from the MT model's encoder-decoder (cross) attention.

    Attributes
    ----------
    source_tokens : List[str]
        Sub-word tokens of the source sentence (special tokens stripped).
    target_tokens : List[str]
        Sub-word tokens of the generated translation (special tokens stripped).
    attention_matrix : np.ndarray, shape (len(target_tokens), len(source_tokens))
        Row i = attention distribution of target token i over all source
        tokens, averaged across attention heads and the last N decoder
        layers, and re-normalized to sum to 1 per row. This is the matrix
        rendered as a heatmap in the Streamlit "Translation & Alignment" view.
    hard_alignment : List[int]
        For each target token, the index of the source token it attends to
        most strongly (``argmax`` of `attention_matrix` per row). This is
        the compact "alignment table" view (one source word per target word).
    """

    source_tokens: List[str]
    target_tokens: List[str]
    attention_matrix: np.ndarray
    hard_alignment: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.hard_alignment and self.attention_matrix.size:
            self.hard_alignment = list(np.argmax(self.attention_matrix, axis=1))


@dataclass
class TranslationResult:
    """
    Full output of one `MTEngine.translate()` call for a single input text
    (which may itself span multiple sentences -- see `sentence_results`).

    Attributes
    ----------
    source_text : str
        The (normalized) input text as translated.
    translated_text : str
        Full translation, sentences re-joined with a single space.
    source_language : str
        Display name of the detected/selected source language.
    target_language : str
        Display name of the selected target language.
    sentence_alignments : List[AlignmentResult]
        One `AlignmentResult` per source sentence, in order, so the UI can
        render either a single concatenated heatmap or a per-sentence view.
    latency_seconds : float
        Wall-clock time spent inside the model's forward/generate calls,
        used as empirical evidence in the Task-B optimization discussion.
    truncated : bool
        True if any sentence exceeded `MAX_SOURCE_TOKENS` and was truncated
        by the tokenizer.
    """

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    sentence_alignments: List[AlignmentResult]
    latency_seconds: float
    truncated: bool = False


class MTEngine:
    """
    Thin, lazily-initialized wrapper around an NLLB-200 seq2seq model that
    provides sentence-level translation plus cross-attention-derived word
    alignment.

    Model loading is deferred to first use (`_ensure_loaded`) rather than
    `__init__`, so that constructing an `MTEngine` (e.g. inside
    `st.cache_resource` at Streamlit app start-up) is cheap, and the ~2.4 GB
    of weights are only pulled into memory the first time a translation is
    actually requested.
    """

    def __init__(self, model_name: str = DEFAULT_MT_MODEL_NAME, device: Optional[str] = None):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace Hub identifier of the NLLB checkpoint to load.
        device : Optional[str]
            "cuda", "cpu", or None to auto-select (CUDA if available, else CPU).
            Explicit control is exposed because a CPU box can still run this
            model (slower), which matters for the group's varying laptops.
        """
        self.model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        """
        Lazily import torch/transformers and load the tokenizer + model.

        Imports are performed inside this method (rather than at module
        top-level) so that importing `mt_engine` does not require torch to
        be installed unless MT is actually used -- e.g. the temporal/event
        module's unit tests do not need a 2 GB torch install to run.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # `attn_implementation="eager"` is required (not just preferred) for
        # word alignment: HuggingFace's default "sdpa" attention backend
        # (PyTorch's fused scaled-dot-product-attention kernel) does not
        # materialize per-head attention weights at all, so
        # `output_attentions=True` silently returns *empty* tuples under
        # sdpa instead of raising -- discovered empirically while wiring up
        # `_extract_alignment` (it failed with an IndexError on an all-empty
        # `cross_attentions` tuple). "eager" trades a small amount of speed
        # for actually exposing the attention tensors the alignment
        # heatmap needs.
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, attn_implementation="eager"
        )
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch

    def _forced_bos_token_id(self, tgt_flores_code: str) -> int:
        """
        Resolve the token id that must be forced as the first generated
        token so NLLB knows which target language to produce.

        Why the try/except
        --------------------
        Older `transformers` releases exposed `tokenizer.lang_code_to_id`
        as a dict; newer releases (>=4.35 or so) register language codes as
        regular special tokens and removed that attribute in favour of
        `tokenizer.convert_tokens_to_ids(code)`. Supporting both keeps the
        code working across the range of `transformers` versions group
        members may have installed locally, rather than failing with an
        opaque `AttributeError` on someone's machine (see Setup Guide
        troubleshooting section).
        """
        lang_code_to_id = getattr(self._tokenizer, "lang_code_to_id", None)
        if lang_code_to_id and tgt_flores_code in lang_code_to_id:
            return lang_code_to_id[tgt_flores_code]
        token_id = self._tokenizer.convert_tokens_to_ids(tgt_flores_code)
        if token_id == self._tokenizer.unk_token_id:
            raise ValueError(
                f"Target language code '{tgt_flores_code}' is not recognized by "
                f"tokenizer for model '{self.model_name}'."
            )
        return token_id

    def _extract_alignment(self, cross_attentions, source_token_strs, generated_ids) -> AlignmentResult:
        """
        Convert the raw cross-attention tensors returned by `.generate()`
        into a dense (target_len x source_len) alignment matrix.

        `cross_attentions` structure (HuggingFace `generate` with
        `output_attentions=True, return_dict_in_generate=True`):
        a tuple of length `num_generated_tokens`, where each element is
        itself a tuple of length `num_decoder_layers`, each a tensor of
        shape (batch=1, num_heads, 1, source_len) for that one decoding
        step (beam_size=1 assumed -- see `translate()`).

        We average over the last 2 decoder layers (empirically the most
        semantically aligned in encoder-decoder Transformers; the very
        last layer alone can be sharper but noisier for short sentences)
        and over all attention heads, producing one probability
        distribution over source tokens per generated target token.

        Parameters
        ----------
        cross_attentions : tuple
            As returned by `model.generate(..., output_attentions=True)`.
        source_token_strs : List[str]
            Source sub-word tokens (special tokens already stripped).
        generated_ids : torch.LongTensor
            The generated token ids (including the forced BOS / special
            tokens), used to recover target sub-word strings and to know
            how many generation steps correspond to real content tokens.

        Returns
        -------
        AlignmentResult
        """
        target_token_strs = self._tokenizer.convert_ids_to_tokens(generated_ids)

        # The first generated position is the forced target-language token
        # (e.g. "hin_Deva") and the final position is the end-of-sentence
        # token; neither carries a meaningful word alignment, so they are
        # excluded from the visualized matrix.
        special_tokens = set(self._tokenizer.all_special_tokens)
        keep_positions = [
            i for i, tok in enumerate(target_token_strs)
            if tok not in special_tokens and not (tok.startswith("__") and tok.endswith("__"))
        ]

        n_source = len(source_token_strs)
        n_target = len(keep_positions)
        matrix = np.zeros((n_target, n_source), dtype=np.float32)

        num_layers = len(cross_attentions[0]) if cross_attentions else 0
        layers_to_average = list(range(max(0, num_layers - 2), num_layers))

        for row, step_idx in enumerate(keep_positions):
            if step_idx >= len(cross_attentions):
                continue
            step_layers = cross_attentions[step_idx]
            # Stack the chosen layers' (heads, 1, source_len) attentions,
            # average over layers and heads -> (source_len,) distribution.
            layer_tensors = [step_layers[l][0] for l in layers_to_average]  # each: (heads, 1, source_len)
            stacked = self._torch.stack(layer_tensors, dim=0)  # (n_layers, heads, 1, source_len)
            averaged = stacked.mean(dim=(0, 1, 2)).detach().cpu().numpy()  # (source_len,)
            averaged = averaged[:n_source]
            row_sum = averaged.sum()
            matrix[row, :] = averaged / row_sum if row_sum > 0 else averaged

        target_tokens_clean = [target_token_strs[i] for i in keep_positions]
        return AlignmentResult(
            source_tokens=source_token_strs,
            target_tokens=target_tokens_clean,
            attention_matrix=matrix,
        )

    def translate_sentence(self, sentence: str, src_lang: LanguageInfo, tgt_lang: LanguageInfo) -> tuple:
        """
        Translate a single sentence and compute its word alignment.

        Returns
        -------
        (str, AlignmentResult, bool)
            Translated sentence text, its alignment against the source, and
            a `truncated` flag (True if the source exceeded MAX_SOURCE_TOKENS).
        """
        self._ensure_loaded()

        self._tokenizer.src_lang = src_lang.flores_code
        inputs = self._tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_TOKENS
        ).to(self._device)

        truncated = inputs["input_ids"].shape[1] >= MAX_SOURCE_TOKENS
        forced_bos_token_id = self._forced_bos_token_id(tgt_lang.flores_code)

        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=1,  # greedy decoding: keeps cross-attention indexing simple (1 hypothesis)
                max_new_tokens=MAX_SOURCE_TOKENS,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0].tolist()
        translated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        source_token_strs = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        special_tokens = set(self._tokenizer.all_special_tokens)
        source_token_strs = [t for t in source_token_strs if t not in special_tokens]

        alignment = self._extract_alignment(output.cross_attentions, source_token_strs, generated_ids)
        return translated_text.strip(), alignment, truncated

    def translate(self, text: str, src_lang_name: str, tgt_lang_name: str) -> TranslationResult:
        """
        Public entry point: normalize, sentence-split, translate each
        sentence, and re-assemble a full `TranslationResult`.

        Parameters
        ----------
        text : str
            Raw source text (any of the supported languages).
        src_lang_name : str
            Display name from `languages.SUPPORTED_LANGUAGES`, e.g. "English".
        tgt_lang_name : str
            Display name of the desired output language.

        Returns
        -------
        TranslationResult
        """
        src_lang = get_language(src_lang_name)
        tgt_lang = get_language(tgt_lang_name)

        if src_lang.flores_code == tgt_lang.flores_code:
            raise ValueError("Source and target language must be different.")

        normalized = normalize_text(text)
        sentences = split_sentences(normalized, src_lang)

        start = time.perf_counter()
        translated_sentences = []
        alignments = []
        any_truncated = False
        for sentence in sentences:
            translated, alignment, truncated = self.translate_sentence(sentence, src_lang, tgt_lang)
            translated_sentences.append(translated)
            alignments.append(alignment)
            any_truncated = any_truncated or truncated
        latency = time.perf_counter() - start

        return TranslationResult(
            source_text=normalized,
            translated_text=" ".join(translated_sentences),
            source_language=src_lang.display_name,
            target_language=tgt_lang.display_name,
            sentence_alignments=alignments,
            latency_seconds=latency,
            truncated=any_truncated,
        )
