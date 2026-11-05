"""Run the real Streamlit page to verify Kokoro voice refresh, disconnects and config changes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import voice


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setattr(config, "ui", dict(config.ui, voice_mode="tts", tts_server="kokoro",
                                          voice_name="kokoro:zf_xiaobei"))
    monkeypatch.setattr(config, "kokoro", dict(base_url="http://localhost:8880/v1",
                                              api_key="", model_id="kokoro", voices=[]))
    monkeypatch.setattr(config, "save_config", Mock())
    monkeypatch.setattr(config, "try_save_config", Mock(return_value=True))
    page = AppTest.from_file(str(Path(__file__).parents[2] / "webui/Main.py"), default_timeout=30)
    page.session_state["ui_language"] = "zh"
    return page


def selected_voice(page):
    return next(item for item in page.selectbox
                if str(item.key).startswith("speech_synthesis_select_kokoro"))


def test_online_offline_recovery_retains_selection(monkeypatch, ui):
    """A real rerun goes through success, a disconnect and recovery without resetting the voice."""
    get = Mock(side_effect=[["kokoro:af_heart", "kokoro:zf_xiaobei"], [],
                           ["kokoro:af_heart", "kokoro:zf_xiaobei"]])
    monkeypatch.setattr(voice, "get_kokoro_voices", get)
    ui.run()
    assert selected_voice(ui).value == "kokoro:zf_xiaobei"
    ui.run()
    assert get.call_count == 1  # Do not hit the service again while the cache is valid.
    ui.session_state["kokoro_voice_catalog"]["checked_at"] = -100
    ui.run()
    assert selected_voice(ui).value == "kokoro:zf_xiaobei"
    assert config.ui["voice_name"] == "kokoro:zf_xiaobei"
    assert any("暂时无法获取 Kokoro" in str(w.value) for w in ui.warning)
    ui.session_state["kokoro_voice_catalog"]["checked_at"] = -100
    ui.run()
    assert selected_voice(ui).value == "kokoro:zf_xiaobei"
    assert not any("暂时无法获取 Kokoro" in str(w.value) for w in ui.warning)
    assert get.call_count == 3
    assert not ui.exception


def test_first_open_offline_preserves_saved_voice(monkeypatch, ui):
    monkeypatch.setattr(voice, "get_kokoro_voices", Mock(return_value=[]))
    ui.run()
    assert selected_voice(ui).value == "kokoro:zf_xiaobei"
    assert config.ui["voice_name"] == "kokoro:zf_xiaobei"
    assert not ui.exception


@pytest.mark.parametrize("input_key, value", [
    ("kokoro_base_url_input", "http://localhost:9999/v1"),
    ("kokoro_api_key_input", "new-test-key"),
])
def test_endpoint_or_credential_change_invalidates_cache(monkeypatch, ui, input_key, value):
    get = Mock(return_value=["kokoro:zf_xiaobei"])
    monkeypatch.setattr(voice, "get_kokoro_voices", get)
    ui.run()
    next(item for item in ui.text_input if item.key == input_key).set_value(value).run()
    assert get.call_count == 2
    assert not ui.exception


def test_manual_voices_and_clearing_them(monkeypatch, ui):
    """Manually entered voices apply at once; clearing them triggers discovery, and the newer
    object list is shown as compact IDs."""
    config.kokoro["voices"] = ["zf_xiaobei", "af_heart"]
    get = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {
        "voices": [{"id": "zf_xiaobei"}, {"id": "af_heart"}],
    }))
    monkeypatch.setattr(voice.requests, "get", get)
    ui.run()
    get.assert_not_called()
    next(item for item in ui.text_input if item.key == "kokoro_voices_input").set_value("").run()
    get.assert_called_once()
    assert selected_voice(ui).options == ["zf_xiaobei", "af_heart"]
    assert selected_voice(ui).value == "kokoro:zf_xiaobei"
    assert not ui.exception
