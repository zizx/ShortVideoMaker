import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config

# Importing app.services.material from webui/Main.py also imports moviepy, which
# resolves the FFmpeg executable at import time (picking the platform binary via
# sys.platform); numpy likewise decides whether to call os.uname() based on
# sys.platform. These resolutions must happen before the fixture below, or
# simulating a headless server on Windows would look for a Linux FFmpeg build or
# raise AttributeError outright.
from app.services import material as _material  # noqa: F401
from app.services import state as sm
from app.utils import utils

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


@pytest.fixture
def headless_task_app(tmp_path, monkeypatch):
    # Use an isolated task directory so the developer's real generation records
    # are neither read nor modified. The test file only needs to be registered by
    # Streamlit as a media resource and is not decoded, so the unit test does not
    # need FFmpeg to produce assets and can reliably cover the headless UI branch.
    tasks_dir = tmp_path / "storage" / "tasks"
    task_dir = tasks_dir / "headless-test"
    task_dir.mkdir(parents=True)
    video_file = task_dir / "final-1.mp4"
    video_file.write_bytes(b"test video payload")

    monkeypatch.setattr(utils, "task_dir", lambda: str(tasks_dir))
    monkeypatch.setattr(sm.state, "get_all_tasks", lambda *_args, **_kwargs: ([], 0))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    # AppTest re-executes the page script several times; config saving must stay
    # isolated for the whole test lifetime so widget initialization never writes
    # into the developer's config.toml.
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.run()
        yield app, video_file


def _button_by_key_prefix(app, key_prefix):
    return next(button for button in app.button if str(button.key).startswith(key_prefix))


def test_headless_play_renders_and_closes_browser_preview(headless_task_app):
    app, video_file = headless_task_app

    _button_by_key_prefix(app, "play_task_all_headless-test").click()
    app.run()

    assert not app.exception
    assert app.session_state["task_preview_video_file"] == str(video_file.resolve())
    assert len(app.get("video")) == 1

    _button_by_key_prefix(app, "close_task_video_preview").click()
    app.run()

    assert "task_preview_video_file" not in app.session_state
    assert len(app.get("video")) == 0


def test_headless_open_folder_shows_host_mapped_path(headless_task_app):
    app, _ = headless_task_app

    _button_by_key_prefix(app, "open_task_all_headless-test").click()
    app.run()

    assert not app.exception
    # _open_task_path() builds the relative part with os.path.relpath(), so the
    # separator follows the platform running the test. The headless branch is
    # entered by monkeypatching sys.platform, and on Windows the toast still uses
    # backslashes, so assertions must not hard-code POSIX separators.
    expected_folder = os.path.join("tasks", "headless-test")
    assert any(
        f"./storage/{expected_folder}" in toast.value for toast in app.get("toast")
    )
