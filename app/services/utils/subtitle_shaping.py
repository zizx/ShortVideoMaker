"""Shaping adapter for Arabic-script subtitles.

Arabic-script languages such as Arabic, Persian and Urdu must first pick
contextual glyph forms and join letters, then lay out right-to-left, to display
correctly; vowel marks must also be positioned on their letters by OpenType
rules. Pillow shapes only when built with Raqm, and common installs such as
Windows lack Raqm, so letters break apart, the order reverses and vowel marks
land in the wrong place.

Here, subtitles containing Arabic-script characters are measured and drawn by
khatt (HarfBuzz shaping + FreeType rasterization, bundled in the wheel with no
system library dependency); khatt never calls Pillow's text layout, so results
are identical whether or not Pillow has Raqm, and no double shaping occurs.
Other languages still go through MoviePy's native TextClip, unchanged.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

import khatt
from khatt.fonts import font_path as khatt_font_path
from khatt.video import TextClip as ShapedTextClip
from loguru import logger

# Unicode blocks related to the Arabic script: base, supplement, Extended A/B
# and presentation forms A/B (excluding the U+FEFF byte order mark).
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFC),
)

# Direction marks: invisible; they only fix a line's direction as a paragraph.
RLM = "‏"
LRM = "‎"

# Fonts bundled with the project contain no Arabic glyphs. When the chosen font
# lacks them, khatt's bundled Noto Naskh Arabic (SIL OFL 1.1) takes over; it
# covers Arabic, Persian and Urdu at the same time.
FALLBACK_FONT = "noto-naskh"


def _is_arabic_char(char: str) -> bool:
    # Only letters, marks and symbols (L/M/S categories) count: the Arabic
    # comma, question mark and Arabic-Indic digits are also shared by Thaana,
    # Syriac and N'Ko, so they cannot identify Arabic script on their own.
    return unicodedata.category(char)[0] in "LMS" and any(
        start <= ord(char) <= end for start, end in _ARABIC_SCRIPT_RANGES
    )


def needs_shaping(text: str | None) -> bool:
    """Whether the text contains Arabic-script letters that need shaping."""
    return any(_is_arabic_char(char) for char in text or "")


def is_rtl_paragraph(text: str) -> bool:
    """Decide paragraph direction from the first strong character (bidi P2/P3)."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def direction_mark(text: str) -> str:
    """A direction mark matching the paragraph direction of ``text``.

    khatt treats every line as its own paragraph and decides direction from the
    first strong character at the line start. Automatically wrapped lines must
    keep the original paragraph's direction, otherwise lines starting with
    English or digits reverse their word order, so this mark is prefixed to
    each line.
    """
    return RLM if is_rtl_paragraph(text) else LRM


def resolve_font(font_path: str, text: str) -> tuple[str, bool]:
    """Return the font file actually used to draw ``text`` and whether it fell back.

    Falling back happens only when the chosen font lacks Arabic-script glyphs;
    if the missing characters are merely emoji or other symbols, the user's
    font is kept and khatt skips them with a warning as it already did.
    """
    try:
        missing = khatt.missing_characters(text, font_path)
    except (KeyError, ValueError) as exc:
        # When the font file is missing or khatt cannot read it, keep subtitles
        # readable.
        logger.warning(f"cannot check subtitle font glyphs, using {FALLBACK_FONT}: {exc}")
        return str(khatt_font_path(FALLBACK_FONT)), True
    if any(_is_arabic_char(char) for char in missing):
        return str(khatt_font_path(FALLBACK_FONT)), True
    return font_path, False


def fallback_fonts(shaped_font: str, font_path: str) -> list[str]:
    """Fonts used for characters ``shaped_font`` cannot draw in shaped subtitles.

    After a fallback font is used, characters the fallback lacks within the same
    subtitle (for example the Chinese after the Arabic text) are still drawn by
    the user's font: khatt falls back run by run, so nothing is dropped.
    """
    if not font_path or shaped_font == font_path:
        return []
    try:
        khatt_font_path(font_path)
    except (KeyError, ValueError):
        return []  # The chosen font is unreadable; resolve_font already warned
    return [font_path]


def video_font(font_path: str, subtitles: Iterable[str]) -> str:
    """Pick one font for all Arabic-script subtitles of a video.

    Choosing a font per subtitle could make subtitles in the same video flip
    between two fonts, so one choice is made from all subtitles at once; the
    font fallback and undrawable characters are each reported once.
    """
    text = "".join(subtitle for subtitle in subtitles if needs_shaping(subtitle))
    if not text:
        return font_path
    font, fell_back = resolve_font(font_path, text)
    if fell_back:
        logger.warning(
            f"subtitle font has no Arabic-script glyphs, using {FALLBACK_FONT} "
            f"for Arabic-script subtitles: {font_path}"
        )
    skipped = khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )
    if skipped:
        # For example emoji: neither the fallback nor the chosen font has these
        # glyphs, so drawing skips them.
        logger.warning(
            "characters in Arabic-script subtitles that the subtitle font cannot "
            f"draw are skipped: {''.join(skipped[:20])}"
        )
    return font


def supports_text(font_path: str, text: str) -> bool:
    """Whether the font actually drawing ``text`` can render every character."""
    font, _ = resolve_font(font_path, text)
    return not khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )


def text_width(
    text: str, font_path: str, font_size: int, fallbacks: Iterable[str] = ()
) -> int:
    """Measure the width in pixels of one shaped line of text.

    ``fallbacks`` is described in :func:`fallback_fonts`.
    """
    left, _, right, _ = khatt.text_bbox(
        text, font=font_path, size_px=int(font_size), fallback_fonts=list(fallbacks)
    )
    return right - left


def text_clip(native, *, shaped_font: str | None = None, **kwargs):
    """Create a subtitle TextClip: khatt draws Arabic-script text, ``native`` the rest.

    ``native`` is the MoviePy ``TextClip`` used by the caller; non-Arabic text is
    handed to it exactly as before. Other arguments match MoviePy 2.2's
    ``TextClip``. ``shaped_font`` is the font that draws Arabic-script text (see
    :func:`video_font`); when omitted it is derived from ``font`` and this text,
    and characters it cannot draw are rendered with ``font``.
    """
    text = kwargs.get("text")
    if not needs_shaping(text):
        return native(**kwargs)
    font = kwargs.get("font") or ""
    if shaped_font is None:
        shaped_font, _ = resolve_font(font, text)
    return ShapedTextClip(
        **{**kwargs, "font": shaped_font},
        fallback_fonts=fallback_fonts(shaped_font, font),
    )
