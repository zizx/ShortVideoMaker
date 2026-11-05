import ast
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models import const


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
TASK_HISTORY_HELPERS = {
    "_find_final_task_video",
    "_build_video_download_name",
    "_build_restore_upload_requirements",
    "_get_unmet_restore_upload_requirements",
}
TASK_HISTORY_CONSTANTS = {
    "_FINAL_VIDEO_PATTERN",
    "_DOWNLOAD_FILENAME_INVALID_PATTERN",
    "_WINDOWS_RESERVED_FILENAMES",
    "VOICE_MODE_TTS",
    "VOICE_MODE_UPLOAD",
    "VOICE_MODE_NONE",
}


def _load_task_history_helpers():
    """
    Load the Streamlit-independent task history pure functions in isolation from
    the WebUI entry point.

    Importing Main.py directly would run the whole page rendering. The test
    compiles only the target constants and functions, which both exercises the
    real merged implementation and avoids splitting out a production module with
    just a few functions for unit tests.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in TASK_HISTORY_CONSTANTS
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in TASK_HISTORY_HELPERS:
            selected_nodes.append(node)

    namespace = {"os": os, "re": re, "Mapping": Mapping}
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


TASK_HISTORY_NAMESPACE = _load_task_history_helpers()
find_final_task_video = TASK_HISTORY_NAMESPACE["_find_final_task_video"]
build_video_download_name = TASK_HISTORY_NAMESPACE["_build_video_download_name"]
windows_reserved_filenames = TASK_HISTORY_NAMESPACE["_WINDOWS_RESERVED_FILENAMES"]
build_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_build_restore_upload_requirements"
]
get_unmet_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_get_unmet_restore_upload_requirements"
]


def test_find_final_task_video_ignores_intermediate_files(tmp_path):
    """Task history may only treat the final video as complete; composition
    intermediate files must not be used."""
    for file_name in (
        "combined-1.mp4",
        "temp-clip-1.mp4",
        "final-1TEMP_MPY_wvf_snd.mp4",
    ):
        (tmp_path / file_name).touch()

    assert find_final_task_video(str(tmp_path)) == ""


def test_find_final_task_video_returns_first_numbered_output(tmp_path):
    """Multi-output tasks match the runtime result: the final video with the
    smallest index plays by default."""
    (tmp_path / "final-10.mp4").touch()
    (tmp_path / "final-2.mp4").touch()
    (tmp_path / "final-1.mp4").touch()

    assert find_final_task_video(str(tmp_path)) == str(tmp_path / "final-1.mp4")


def test_build_video_download_name_uses_subject_and_output_index():
    assert (
        build_video_download_name("A day: in / Shanghai?", 2, 3)
        == "A day in Shanghai-2.mp4"
    )


def test_build_video_download_name_handles_empty_and_long_subjects():
    assert build_video_download_name("  ...  ", 1, 1) == "video.mp4"
    assert len(build_video_download_name("a" * 100, 1, 1)) == 84


def test_build_video_download_name_avoids_windows_reserved_names():
    # Use an explicit list based on the official rules instead of copying the
    # production range/comprehension into the test; if the implementation gets
    # the range wrong, the set-equality assertion fails immediately rather than
    # missing the bug along with the implementation.
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "COM¹",
        "COM²",
        "COM³",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
        "LPT¹",
        "LPT²",
        "LPT³",
    }
    assert windows_reserved_filenames == reserved_names

    for reserved_name in reserved_names:
        assert (
            build_video_download_name(reserved_name.lower(), 1, 1)
            == f"_{reserved_name.lower()}.mp4"
        )
        assert (
            build_video_download_name(f"{reserved_name}.topic", 2, 3)
            == f"_{reserved_name}.topic-2.mp4"
        )

    assert build_video_download_name("con .topic", 1, 1) == "_con .topic.mp4"


def test_build_video_download_name_does_not_overmatch_similar_names():
    """Only real Windows reserved names are handled; adjacent but valid ordinary
    subjects must not be affected."""

    for safe_name in ("COM0", "COM10", "LPT0", "LPT10", "COM⁴", "LPT⁴"):
        assert build_video_download_name(safe_name, 1, 1) == f"{safe_name}.mp4"


def test_restore_requirements_block_missing_uploaded_files():
    params = {
        "video_source": "local",
        "custom_audio_file": "/old-task/custom-audio.wav",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
    }
    requirements = build_restore_upload_requirements(params)

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="local",
        voice_name=params["voice_name"],
        has_local_materials=False,
        has_custom_audio=False,
    ) == {"local_materials", "custom_audio"}


def test_restore_requirements_allow_explicit_replacements():
    requirements = build_restore_upload_requirements(
        {
            "video_source": "local",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert not get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="en-US-JennyNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
    )


def test_restore_requirements_require_file_in_upload_voice_mode():
    """When restoring an uploaded-voiceover task, staying in upload mode
    requires selecting the audio file again."""
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
        voice_mode="upload",
    ) == {"custom_audio"}


def test_restore_requirements_allow_replacing_upload_with_other_voice_modes():
    """When the user actively switches to automatic voiceover or no voiceover,
    the historical upload file is no longer required."""
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    for voice_mode in ("tts", "none"):
        assert not get_unmet_restore_upload_requirements(
            requirements,
            video_source="pexels",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            has_local_materials=False,
            has_custom_audio=False,
            voice_mode=voice_mode,
        )


@pytest.mark.parametrize("payload", [[], {"params": []}, {"script": []}])
def test_history_scan_skips_non_object_script_payload(tmp_path, payload):
    """One malformed historical artifact must not break the task panel."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "broken-task"
    task_dir.mkdir(parents=True)
    (task_dir / "script.json").write_text(json.dumps(payload), encoding="utf-8")

    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_safe_load_task_script", "_scan_history_tasks"}
    ]
    namespace = {
        "os": os,
        "json": json,
        "logger": MagicMock(),
        "utils": SimpleNamespace(task_dir=lambda: str(tasks_root)),
        "const": const,
        "_find_final_task_video": lambda _task_path: "",
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)

    history = namespace["_scan_history_tasks"]()

    assert len(history) == 1
    assert history[0]["task_id"] == "broken-task"
    assert history[0]["subject"] == "broken-task"


def test_find_final_task_video_ignores_directories_and_broken_links(tmp_path):
    (tmp_path / "final-1.mp4").mkdir()
    (tmp_path / "final-2.mp4").symlink_to(tmp_path / "missing.mp4")
    assert find_final_task_video(str(tmp_path)) == ""
    (tmp_path / "final-3.mp4").touch()
    assert find_final_task_video(str(tmp_path)) == str(tmp_path / "final-3.mp4")
