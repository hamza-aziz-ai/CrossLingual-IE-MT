"""
fonts.py
=========

Matplotlib rendering fix for Indic-script tick labels in the word-alignment
heatmap.

Why this module exists
-------------------------
Matplotlib's bundled default font (DejaVu Sans) has no Devanagari, Tamil,
or Bengali glyphs. Rendering an alignment heatmap for a Hindi/Marathi,
Tamil, or Bengali translation therefore produces empty "missing glyph"
boxes for every target-language tick label unless a script-appropriate
font is supplied explicitly (this was caught during development: see
`docs/System_Report.md`, Challenges section). Three open-licensed
(SIL Open Font License) Noto fonts are bundled under `assets/fonts/` for
exactly this purpose, so the heatmap renders correctly with **no internet
access required at run time** -- fonts are shipped with the repository,
not downloaded on demand.
"""

from pathlib import Path
from typing import Optional

from matplotlib.font_manager import FontProperties

ASSETS_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# (unicode_start, unicode_end) -> bundled font filename. Mirrors the
# script ranges in `languages.py`'s `unicode_block` field; kept as a
# separate table (rather than importing from `languages.py`) because this
# module is about *rendering*, not language identification, and Marathi
# shares Devanagari with Hindi so both map to the same font file.
_SCRIPT_FONT_MAP = {
    (0x0900, 0x097F): "NotoSansDevanagari-Regular.ttf",  # Hindi, Marathi
    (0x0B80, 0x0BFF): "NotoSansTamil-Regular.ttf",       # Tamil
    (0x0980, 0x09FF): "NotoSansBengali-Regular.ttf",     # Bengali
}

_font_property_cache = {}


def get_font_property_for_text(text: str) -> Optional[FontProperties]:
    """
    Return a matplotlib `FontProperties` pointing at the bundled font that
    covers the first Indic script detected in `text`, or `None` if `text`
    contains no Indic-script characters (in which case matplotlib's
    default font, which does cover Latin script, is sufficient).

    Parameters
    ----------
    text : str
        A single tick-label string (e.g. one NLLB sub-word token, which
        may include the SentencePiece "▁" word-boundary marker).

    Returns
    -------
    Optional[FontProperties]
        Usable directly as `Text.set_fontproperties(...)` on a matplotlib
        tick label.
    """
    for ch in text:
        code_point = ord(ch)
        for (start, end), font_filename in _SCRIPT_FONT_MAP.items():
            if start <= code_point <= end:
                if font_filename not in _font_property_cache:
                    font_path = ASSETS_FONTS_DIR / font_filename
                    if not font_path.exists():
                        # Degrade gracefully: if the bundled font asset is
                        # missing for some reason, fall back to the default
                        # font rather than raising and breaking the whole
                        # heatmap render.
                        _font_property_cache[font_filename] = None
                    else:
                        _font_property_cache[font_filename] = FontProperties(fname=str(font_path))
                return _font_property_cache[font_filename]
    return None


def apply_script_aware_fonts(tick_labels, token_strings) -> None:
    """
    Apply the correct bundled font to each matplotlib tick label whose
    corresponding token contains Indic-script characters.

    Parameters
    ----------
    tick_labels : List[matplotlib.text.Text]
        The list returned by `ax.get_xticklabels()` / `ax.get_yticklabels()`.
    token_strings : List[str]
        The token strings in the same order used to set those tick labels
        (i.e. what was passed to `ax.set_xticklabels(...)`).
    """
    for label, token in zip(tick_labels, token_strings):
        font_property = get_font_property_for_text(token)
        if font_property is not None:
            label.set_fontproperties(font_property)
