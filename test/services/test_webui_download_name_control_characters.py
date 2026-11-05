"""Task-history download file names must strip control and bidirectional
formatting characters."""
import ast
import os
import re
from collections.abc import Mapping
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
DOWNLOAD_NAME_HELPERS = {"_build_video_download_name"}
DOWNLOAD_NAME_CONSTANTS = {
    "_DOWNLOAD_FILENAME_INVALID_PATTERN",
    "_WINDOWS_RESERVED_FILENAMES",
}


def _load_download_name_helpers():
    """
    Load the download-name pure functions in isolation from the WebUI entry point.

    Importing Main.py directly would run the whole page render. Only the target
    constants and functions are compiled here, and they still execute the merged
    real implementation, so assertions cover production behavior rather than a
    copy kept in the test.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in DOWNLOAD_NAME_CONSTANTS
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in DOWNLOAD_NAME_HELPERS:
            selected_nodes.append(node)

    namespace = {"os": os, "re": re, "Mapping": Mapping}
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


build_video_download_name = _load_download_name_helpers()[
    "_build_video_download_name"
]

# List the characters that must be removed explicitly instead of copying the
# production range derivation: if the implementation gets a range wrong the
# assertion fails outright instead of missing it too. C0 controls and U+0085,
# U+2028, U+2029 are already handled by whitespace collapsing, but they are still
# checked one by one here so later changes cannot put them back into file names.
C0_CONTROL_CHARACTERS = ("\x00", "\x08", "\x1b", "\x1f")
C1_CONTROL_CHARACTERS = ("\x7f", "\x80", "\x85", "\x9f")
BIDIRECTIONAL_CONTROL_CHARACTERS = (
    "‎",
    "‏",
    "‪",
    "‫",
    "‬",
    "‭",
    "‮",
    "⁦",
    "⁧",
    "⁨",
    "⁩",
)
LINE_SEPARATOR_CHARACTERS = (" ", " ")


def test_download_name_drops_bidirectional_override_from_subject():
    """U+202E renders the following characters in reverse, faking another
    believable-looking extension."""
    download_name = build_video_download_name("holiday‮4pm", 1, 1)

    assert "‮" not in download_name
    assert download_name == "holiday 4pm.mp4"


def test_download_name_drops_every_unsafe_character_from_subject():
    """No character rejected by the upload entry point may reach a download file
    name."""
    unsafe_characters = (
        C0_CONTROL_CHARACTERS
        + C1_CONTROL_CHARACTERS
        + BIDIRECTIONAL_CONTROL_CHARACTERS
        + LINE_SEPARATOR_CHARACTERS
    )
    for character in unsafe_characters:
        download_name = build_video_download_name(f"trip{character}note", 1, 1)

        assert character not in download_name, f"U+{ord(character):04X} survived"
        assert download_name == "trip note.mp4"


def test_download_name_keeps_ordinary_multilingual_subjects():
    """Stripping controls must not rewrite ordinary subjects, including Chinese
    text, accented letters and punctuation."""
    assert build_video_download_name("上海一日游", 1, 1) == "上海一日游.mp4"
    assert build_video_download_name("Café déjà vu", 2, 3) == "Café déjà vu-2.mp4"
    assert (
        build_video_download_name("A day: in / Shanghai?", 2, 3)
        == "A day in Shanghai-2.mp4"
    )
