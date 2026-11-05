"""Regression tests for Arabic-script subtitle shaping.

These tests inspect the actual rendered glyphs, not just strings: vowel marks
must sit on their letters, letters must join, text must be laid out right to
left, and regardless of whether Pillow has Raqm, Pillow's text layout is never
used (so no double shaping happens).
"""

import functools
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from khatt.fonts import font_path as khatt_font_path
from khatt.video import TextClip as ShapedTextClip
from loguru import logger
from moviepy import ColorClip, TextClip
from PIL import ImageDraw, ImageFont

from app.services import video as vd
from app.services.utils import subtitle_shaping
from app.services.video import wrap_text

FONT_DIR = Path(__file__).resolve().parents[2] / "resource/fonts"
# The project's default subtitle font has no Arabic glyphs and covers the font
# fallback path.
DEFAULT_FONT = str(FONT_DIR / "STHeitiMedium.ttc")
# The font the maintainer used to reproduce Arabic and Chinese in one subtitle.
YAHEI_FONT = str(FONT_DIR / "MicrosoftYaHeiBold.ttc")
MIXED_CUE = "مرحبا بالعالم 中文测试"
ARABIC_FONT = str(khatt_font_path("amiri"))
FALLBACK_FONT = str(khatt_font_path(subtitle_shaping.FALLBACK_FONT))
# Directory of fonts bundled with khatt, so generate_video can select them
# directly.
KHATT_FONT_DIR = str(khatt_font_path("amiri").parent)
RLM, LRM = subtitle_shaping.RLM, subtitle_shaping.LRM

# khatt raises TextOverflowWarning when text exceeds the canvas and content is
# clipped: treat that as a failure here.
pytestmark = pytest.mark.filterwarnings("error::khatt.TextOverflowWarning")


def shaped_clip(text, font=DEFAULT_FONT, **kwargs):
    return subtitle_shaping.text_clip(
        TextClip, text=text, font=font, font_size=100, color="white", **kwargs
    )


def ink_box(clip):
    """The region of the subtitle clip that actually has ink (left, top, right,
    bottom)."""
    ink = clip.mask.get_frame(0) > 0.5
    ys, xs = np.nonzero(ink)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ink_width(clip):
    left, _, right, _ = ink_box(clip)
    return right - left


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مرحبا بالعالم", True),
        ("سلام دنیا", True),  # Persian
        ("ٹھیک ہے", True),  # Urdu
        ("Breaking news: عاجل", True),
        ("ﻻ", True),  # presentation form (lam-alef ligature)
        ("﷽", True),  # Basmala ligature (symbol category)
        ("Hello world", False),
        ("你好，世界", False),
        ("नमस्ते", False),
        ("", False),
        (None, False),
        ("\ufeffBOM", False),  # byte order mark is not Arabic script
        ("שלום עולם", False),  # Hebrew
        ("ދިވެހި، ބަސް", False),  # Dhivehi (Thaana) uses an Arabic comma
        ("ܫܠܡܐ؟", False),  # Syriac uses an Arabic question mark
        ("ߒߞߏ؛", False),  # N'Ko uses an Arabic semicolon
        ("١٢٣ ٤٥٦", False),  # Arabic-Indic digits only
    ],
)
def test_only_arabic_script_text_is_shaped(text, expected):
    assert subtitle_shaping.needs_shaping(text) is expected


def test_font_without_arabic_glyphs_falls_back():
    font, fell_back = subtitle_shaping.resolve_font(DEFAULT_FONT, "مرحبا")
    assert fell_back
    assert Path(font).name == "NotoNaskhArabic-Regular.ttf"


def test_font_with_arabic_glyphs_is_kept():
    assert subtitle_shaping.resolve_font(ARABIC_FONT, "مرحبا") == (ARABIC_FONT, False)


def test_missing_emoji_alone_does_not_switch_font():
    # Missing only an emoji keeps the user's chosen font; do not switch to the
    # fallback font.
    assert subtitle_shaping.resolve_font(ARABIC_FONT, "مرحبا 😀") == (ARABIC_FONT, False)


def test_non_arabic_text_goes_to_moviepy_unchanged():
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return "native clip"

    kwargs = dict(text="Hello", font=DEFAULT_FONT, font_size=60, color="white", stroke_width=2)
    assert subtitle_shaping.text_clip(native, **kwargs) == "native clip"
    assert calls == [kwargs]


def test_arabic_text_is_drawn_by_khatt():
    clip = shaped_clip("مرحبا بالعالم", stroke_color="black", stroke_width=3)
    assert isinstance(clip, ShapedTextClip)
    assert ink_width(clip) > 0


def test_never_uses_pillow_text_layout(monkeypatch):
    # Whether or not Pillow has Raqm, Arabic-script subtitles never go through
    # Pillow's text layout, so there is no double shaping and results do not
    # vary by environment.
    def forbidden(*args, **kwargs):
        raise AssertionError("Pillow text layout must not be used for Arabic subtitles")

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", forbidden)
    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", forbidden)
    assert ink_width(shaped_clip("بِتَ مرحبا")) > 0


@pytest.mark.parametrize("font", [DEFAULT_FONT, ARABIC_FONT])
def test_vowel_marks_stay_on_their_letters(font):
    # Maintainer reproduction case: without shaping Kasra extends past the
    # rightmost glyph (107 -> 145 px at 100px).
    plain = ink_box(shaped_clip("بت", font=font))
    marked = ink_box(shaped_clip("بِتَ", font=font))
    assert abs((marked[2] - marked[0]) - (plain[2] - plain[0])) <= 2
    # Vowel marks sit above/below the letters, so the ink only grows
    # vertically.
    assert marked[1] < plain[1]
    assert marked[3] > plain[3]


def ink_runs(clip):
    """Count contiguous ink columns from left to right: a joined word is one
    run."""
    columns = (clip.mask.get_frame(0) > 0.5).any(axis=0)
    starts = columns & ~np.concatenate(([False], columns[:-1]))
    return int(starts.sum())


@pytest.mark.parametrize("word", ["سسس", "ببب"])
def test_letters_join(word):
    # Without shaping (Pillow without Raqm) these letters are three separate
    # glyphs; after shaping they join into one run.
    assert ink_runs(shaped_clip(word)) == 1
    assert ink_runs(shaped_clip(" ".join(word))) == 3


def test_right_to_left_order():
    # Alef is a tall vertical stroke and Beh is a low glyph: in right-to-left
    # order Alef must be on the right.
    ink = shaped_clip("ا ب").mask.get_frame(0) > 0.5
    ys, xs = np.nonzero(ink)
    middle = (xs.min() + xs.max()) / 2
    right_height = np.ptp(ys[xs > middle])
    left_height = np.ptp(ys[xs <= middle])
    assert right_height > left_height


def strip_marks(line):
    return line.lstrip(RLM + LRM)


ARABIC_PARAGRAPH = "تمنح هذه الشهادة إلى الطالب المتميز تقديرا لجهوده الكبيرة في العام الدراسي"


def test_wrap_text_measures_shaped_arabic():
    wrapped, height = wrap_text(ARABIC_PARAGRAPH, 600, font=DEFAULT_FONT, fontsize=60)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    # Wrap only at spaces; words are never split.
    assert " ".join(strip_marks(line) for line in lines).split() == ARABIC_PARAGRAPH.split()
    font, _ = subtitle_shaping.resolve_font(DEFAULT_FONT, ARABIC_PARAGRAPH)
    assert all(subtitle_shaping.text_width(line, font, 60) <= 600 for line in lines)
    assert height > 0


def test_arabic_wraps_by_shaped_width_even_without_raqm(monkeypatch):
    # Windows Pillow has no Raqm: widths measured by Pillow for Arabic are
    # disconnected, reversed letter widths. Wrapping must still use the shaped
    # width: each line stays within the limit and greedily fills as much as
    # possible.
    basic = functools.partial(ImageFont.truetype, layout_engine=ImageFont.Layout.BASIC)
    monkeypatch.setattr(vd.ImageFont, "truetype", basic)
    max_width = 420
    wrapped, height = wrap_text(ARABIC_PARAGRAPH, max_width, font=ARABIC_FONT, fontsize=48)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert " ".join(strip_marks(line) for line in lines) == ARABIC_PARAGRAPH
    for line, following in zip(lines, lines[1:] + [None]):
        assert subtitle_shaping.text_width(line, ARABIC_FONT, 48) <= max_width
        if following is not None:
            longer = f"{line} {strip_marks(following).split()[0]}"
            assert subtitle_shaping.text_width(longer, ARABIC_FONT, 48) > max_width
    assert height == len(lines) * sum(ImageFont.truetype(ARABIC_FONT, 48).getmetrics())


def test_wrapped_lines_keep_their_paragraphs_direction():
    # Auto-wrapped lines keep their paragraph direction: each line of a
    # right-to-left paragraph starts with RLM, and each line of a left-to-right
    # paragraph containing Arabic starts with LRM.
    rtl, _ = wrap_text("مرحبا بالعالم Khatt 2026 جديد " * 4, 300, font=ARABIC_FONT, fontsize=48)
    ltr, _ = wrap_text("Hello مرحبا world بالعالم " * 4, 300, font=ARABIC_FONT, fontsize=48)
    assert len(rtl.split("\n")) > 1 and len(ltr.split("\n")) > 1
    for line in rtl.split("\n"):
        assert line.startswith(RLM) and subtitle_shaping.is_rtl_paragraph(line)
    for line in ltr.split("\n"):
        assert line.startswith(LRM) and not subtitle_shaping.is_rtl_paragraph(line)


def test_hand_broken_lines_keep_their_own_direction():
    # Bilingual subtitles: one Arabic line and one English translation line,
    # each keeping its own direction.
    wrapped, _ = wrap_text("مرحبا بالعالم\nHello, world!", 900, font=ARABIC_FONT, fontsize=48)
    arabic_line, english_line = wrapped.split("\n")
    assert arabic_line == RLM + "مرحبا بالعالم"
    assert strip_marks(english_line) == "Hello, world!"
    assert not subtitle_shaping.is_rtl_paragraph(english_line)


def test_translation_lines_are_measured_as_khatt_draws_them(monkeypatch):
    # Subtitles containing Arabic letters are drawn entirely by khatt, so
    # English translation lines are also measured with khatt.
    measured = []
    original = subtitle_shaping.text_width

    def spy(text, *args):
        measured.append(strip_marks(text))
        return original(text, *args)

    monkeypatch.setattr(subtitle_shaping, "text_width", spy)
    wrap_text("مرحبا\nHello, world!", 900, font=ARABIC_FONT, fontsize=48)
    assert "Hello, world!" in measured


def test_lines_are_measured_in_the_direction_they_render():
    # The second line wrapped out of a right-to-left paragraph starts with
    # English. Mixed-line width differs by direction, so measurement must use
    # the rendering direction (right-to-left), not the direction inferred from
    # the leading English word.
    word = "الاستـ" + "ـ" * 16 + "ـقبال"  # stretched with Kashida to fill line one
    line = "Khatt 1.5 مرحبا بكم"
    rtl_width = subtitle_shaping.text_width(RLM + line, ARABIC_FONT, 48)
    ltr_width = subtitle_shaping.text_width(LRM + line, ARABIC_FONT, 48)
    assert rtl_width < ltr_width  # premise: left-to-right measures wider
    wrapped, _ = wrap_text(f"{word} {line}", rtl_width, font=ARABIC_FONT, fontsize=48)
    assert wrapped.split("\n") == [RLM + word, RLM + line]  # line two just fits


def test_arabic_words_are_never_split_across_lines():
    # CJK line-start punctuation adjustment moves the last character of the
    # previous line to the next line; Arabic words must not be split that way,
    # or the letter joining breaks.
    text = "مرحبا بالعالم الجميل ! نعم , هذا اختبار . للتأكد ؟ من النتيجة"
    words = set(text.split())
    for max_width in range(120, 600, 15):
        wrapped, _ = wrap_text(text, max_width, font=ARABIC_FONT, fontsize=48)
        assert {w for line in wrapped.split("\n") for w in strip_marks(line).split()} <= words


def test_non_arabic_wrapping_gets_no_direction_marks():
    for text in ("Hello world from MoneyPrinterTurbo " * 4, "你好世界，这是一个很长的中文字幕句子。" * 3):
        wrapped, _ = wrap_text(text, 300, font=DEFAULT_FONT, fontsize=48)
        assert RLM not in wrapped and LRM not in wrapped


def test_webui_font_check_judges_the_font_that_draws_arabic():
    # Pure Arabic: the fallback font supplies the glyphs, so no more false
    # "font unsupported" reports.
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "مرحبا بالعالم")
    # Arabic mixed with Chinese: the fallback font draws Arabic and the
    # selected font draws Chinese; both can be drawn.
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "مرحبا 你好")
    # Detection for other languages is unchanged.
    vietnamese_only = str(FONT_DIR / "BeVietnamPro-Medium.ttf")
    assert not vd.subtitle_font_supports_text(vietnamese_only, "你好世界")
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "你好世界")


class FakeAudio:
    duration = 3

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        pass

    def with_effects(self, _effects):
        return self


def render_subtitled_video(subtitles, frame_at=None, font_dir=None, **param_overrides):
    """Run the real ColorClip through generate_video's subtitle flow and return
    the frame at the given time plus the renderer calls."""
    params = vd.VideoParams(
        video_subject="test",
        video_aspect="9:16",
        subtitle_enabled=True,
        font_name=param_overrides.pop("font_name", "STHeitiMedium.ttc"),
        font_size=60,
        stroke_width=2,
        bgm_type="",
        **param_overrides,
    )
    srt = "".join(
        f"{index}\n00:00:0{index - 1},000 --> 00:00:0{index},000\n{text}\n\n"
        for index, text in enumerate(subtitles, start=1)
    )
    frames = []

    def write(clip, *_args, **_kwargs):
        # generate_video closes the composite video after returning, so capture
        # the frame while writing.
        if frame_at is not None:
            frames.append(clip.get_frame(frame_at))

    with tempfile.TemporaryDirectory() as tmp_dir:
        subtitle_path = Path(tmp_dir) / "subtitle.srt"
        subtitle_path.write_text(srt, encoding="utf-8")
        source = ColorClip((1080, 1920), color=(20, 30, 60), duration=len(subtitles))
        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source),
            patch.object(vd, "AudioFileClip", return_value=FakeAudio()),
            patch.object(vd, "TextClip", wraps=TextClip) as moviepy_renderer,
            patch.object(
                subtitle_shaping, "ShapedTextClip", wraps=ShapedTextClip
            ) as khatt_renderer,
            patch.object(vd, "_write_videofile_with_codec_fallback", side_effect=write),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            patch.object(vd.utils, "font_dir", return_value=font_dir or str(FONT_DIR)),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path=str(subtitle_path),
                output_file=str(Path(tmp_dir) / "final.mp4"),
                params=params,
            )
    return (frames[0] if frames else None), moviepy_renderer, khatt_renderer


def subtitle_calls(renderer):
    """The calls where create_text_clip actually draws a subtitle (with color
    and size).

    Excludes SubtitlesClip's probe calls through make_textclip, and the
    measurement calls the no-background style makes to find the natural height
    of Arabic subtitles.
    """
    return [
        call for call in renderer.call_args_list if "size" in call.kwargs and "color" in call.kwargs
    ]


STYLES = [
    {},
    {"text_background_color": "#000000"},
    {"text_background_color": "#000000", "rounded_subtitle_background": True},
]
STYLE_IDS = ["no-background", "background", "rounded-background"]


@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_only_arabic_subtitles_go_through_khatt(overrides):
    subtitles = ["Hello world", "مرحبا بالعالم", "你好世界"]
    frame, moviepy_renderer, khatt_renderer = render_subtitled_video(
        subtitles, frame_at=1.5, **overrides
    )
    khatt_calls = subtitle_calls(khatt_renderer)
    moviepy_calls = subtitle_calls(moviepy_renderer)
    assert [call.kwargs["text"] for call in khatt_calls] == [RLM + "مرحبا بالعالم"]
    assert all(call.kwargs["font"] == FALLBACK_FONT for call in khatt_calls)
    # Calls for other languages are exactly as before: original font, original
    # text, no direction marks. (On Windows generate_video replaces backslashes
    # in font paths with forward slashes, so compare via Path.)
    assert [call.kwargs["text"] for call in moviepy_calls] == ["Hello world", "你好世界"]
    assert all(Path(call.kwargs["font"]) == FONT_DIR / "STHeitiMedium.ttc" for call in moviepy_calls)
    assert (frame >= 250).all(axis=2).any()  # that second drew white Arabic text


def test_rounded_background_fits_the_shaped_arabic_text():
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا بالعالم"], text_background_color="#000000", rounded_subtitle_background=True
    )
    (call,) = subtitle_calls(khatt_renderer)
    pad_x = int(60 * 0.4)
    expected = subtitle_shaping.text_width(call.kwargs["text"], FALLBACK_FONT, 60) + 2 * pad_x
    assert call.kwargs["size"][0] == expected  # shaped width, not STHeiti missing-glyph width


@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_quranic_pause_marks_are_not_clipped(overrides):
    # Amiri's pause marks extend above the font's top edge; the fixed height of
    # the no-background style must not clip them (khatt emits TextOverflowWarning
    # when clipping, which this file treats as a failure). This verse needs
    # 152px at 60px with a 2px stroke, while the formula gives only 147px.
    render_subtitled_video(
        ["وَٱلضُّحَىٰ ۖ وَٱلَّيْلِ إِذَا سَجَىٰ ۗ"],
        font_dir=KHATT_FONT_DIR,
        font_name="Amiri-Regular.ttf",
        **overrides,
    )


def test_make_textclip_routes_arabic_through_khatt():
    # SubtitlesClip probes the first subtitle with make_textclip.
    _, moviepy_renderer, khatt_renderer = render_subtitled_video(["مرحبا", "Hello"])
    probes = [call for call in khatt_renderer.call_args_list if "size" not in call.kwargs]
    assert probes and all(call.kwargs["font"] == FALLBACK_FONT for call in probes)
    # MoviePy itself probes with "T" to check for a transparent mask; other
    # than that, no Arabic should be handed to MoviePy.
    moviepy_probes = [c for c in moviepy_renderer.call_args_list if "size" not in c.kwargs]
    assert {call.kwargs["text"] for call in moviepy_probes} <= {"T"}


def test_one_arabic_font_for_the_whole_video(monkeypatch):
    # Rubik has Arabic letters but not the Urdu ے: Arabic subtitles for the
    # whole video switch to the fallback font rather than switching between the
    # two fonts per cue; wrapping is measured with that font too.
    measured_fonts = set()
    original = subtitle_shaping.text_width

    def spy(text, font, size, *args):
        measured_fonts.add(font)
        return original(text, font, size, *args)

    monkeypatch.setattr(subtitle_shaping, "text_width", spy)
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا", "پاکستان ے"], font_dir=KHATT_FONT_DIR, font_name="Rubik-Regular.ttf"
    )
    calls = subtitle_calls(khatt_renderer)
    assert len(calls) == 2
    assert {call.kwargs["font"] for call in calls} == {FALLBACK_FONT}
    assert measured_fonts == {FALLBACK_FONT}


def test_user_font_with_arabic_glyphs_is_used():
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا", "سلام"], font_dir=KHATT_FONT_DIR, font_name="Rubik-Regular.ttf"
    )
    assert {Path(call.kwargs["font"]) for call in subtitle_calls(khatt_renderer)} == {
        Path(KHATT_FONT_DIR) / "Rubik-Regular.ttf"
    }


def warnings_while(render):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        render()
    finally:
        logger.remove(sink)
    return messages


def test_font_fallback_is_logged_once_per_video():
    messages = warnings_while(
        lambda: [render_subtitled_video(["مرحبا", "بالعالم", "سلام"]) for _ in range(2)]
    )
    fallback = f"using {subtitle_shaping.FALLBACK_FONT}"
    assert sum(fallback in message for message in messages) == 2  # once per video


@pytest.mark.filterwarnings("ignore::khatt.UnsupportedCharacterWarning")
def test_characters_no_font_can_draw_are_reported():
    # Characters that neither the fallback font nor the selected font can draw
    # (an emoji here) are skipped: the log must say so.
    messages = warnings_while(lambda: render_subtitled_video(["مرحبا 😀 你好"]))
    skipped = [message for message in messages if "skipped" in message]
    assert skipped and all("😀" in message and "你" not in message for message in skipped)


@pytest.mark.filterwarnings("error::khatt.UnsupportedCharacterWarning")
def test_mixed_arabic_and_chinese_cue_keeps_the_chinese():
    # The maintainer reported in #1698 that Chinese in the same subtitle was
    # skipped by the fallback font. Now Chinese is drawn by the selected font,
    # and the rendered Chinese is identical to drawing it with that font alone.
    clip = shaped_clip(MIXED_CUE, font=YAHEI_FONT)
    chinese = ShapedTextClip(text="中文测试", font=YAHEI_FONT, font_size=100, color="white")
    chinese_width = ink_width(chinese)
    # In a right-to-left paragraph Chinese is to the left of Arabic: the far
    # left of the frame is exactly these characters.
    left = clip.mask.get_frame(0)[:, :chinese_width] > 0.5
    alone = chinese.mask.get_frame(0) > 0.5
    rows = np.nonzero(left.any(axis=1))[0]
    alone_rows = np.nonzero(alone.any(axis=1))[0]
    assert np.array_equal(
        left[rows.min() : rows.max() + 1],
        alone[alone_rows.min() : alone_rows.max() + 1, :chinese_width],
    )
    # Arabic is still shaped by the fallback font, unchanged in width, and
    # immediately to the right of the Chinese.
    arabic = shaped_clip("مرحبا بالعالم", font=YAHEI_FONT)
    assert ink_width(clip) > chinese_width + ink_width(arabic)


@pytest.mark.filterwarnings("error::khatt.UnsupportedCharacterWarning")
@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_mixed_cue_in_a_video_draws_the_chinese_with_the_selected_font(overrides):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        frame, _, khatt_renderer = render_subtitled_video(
            [MIXED_CUE], frame_at=0.5, font_name="MicrosoftYaHeiBold.ttc", **overrides
        )
    finally:
        logger.remove(sink)
    (call,) = subtitle_calls(khatt_renderer)
    assert call.kwargs["font"] == FALLBACK_FONT
    assert [Path(font) for font in call.kwargs["fallback_fonts"]] == [Path(YAHEI_FONT)]
    assert not any("skipped" in message for message in messages)
    assert (frame >= 250).all(axis=2).any()
    if overrides.get("rounded_subtitle_background"):
        # The rounded background width also accounts for Chinese.
        text_w = subtitle_shaping.text_width(
            call.kwargs["text"], FALLBACK_FONT, 60, call.kwargs["fallback_fonts"]
        )
        assert call.kwargs["size"][0] == text_w + 2 * int(60 * 0.4)


def test_wrapping_a_mixed_cue_measures_the_chinese():
    wrapped, _ = wrap_text(MIXED_CUE * 3, 500, font=YAHEI_FONT, fontsize=60)
    for line in wrapped.split("\n"):
        width = subtitle_shaping.text_width(line, FALLBACK_FONT, 60, [YAHEI_FONT])
        assert width <= 500


def test_wrap_text_leaves_other_languages_alone(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("non-Arabic subtitles must be measured as before")

    monkeypatch.setattr(subtitle_shaping, "text_width", forbidden)
    wrapped, _ = wrap_text("Hello world " * 20, 600, font=DEFAULT_FONT, fontsize=60)
    assert len(wrapped.split("\n")) > 1
