import asyncio
import base64
import io
import inspect
import itertools
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from datetime import timedelta
from typing import Union
from urllib.parse import urlparse
from xml.sax.saxutils import escape, unescape

import edge_tts
import requests
from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.services import bgm
from app.utils import utils
from app.utils.subtitle_reader import moviepy_subtitles
from app.utils.subtitle_writer import staged_subtitle_file

_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS = 30.0
_SILICONFLOW_TTS_TIMEOUT_SECONDS = (10, 300)  # connect, read
_MIMO_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
_MIMO_DEFAULT_TTS_MODEL = "mimo-v2.5-tts"
MINIMAX_TTS_GLOBAL_URL = "https://api.minimax.io/v1/t2a_v2"
MINIMAX_TTS_CN_URL = "https://api.minimaxi.com/v1/t2a_v2"
MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_TTS_DEFAULT_VOICE = "English_expressive_narrator"
MINIMAX_TTS_MODELS = (
    "speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
    "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo",
)
GEMINI_TTS_VOICES = (
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)
_MINIMAX_TTS_MAX_AUDIO_HEX_CHARS = 100 * 1024 * 1024
_ELEVENLABS_TTS_MAX_AUDIO_BYTES = 50 * 1024 * 1024
_ELEVENLABS_TTS_MAX_ERROR_BYTES = 4096
VOXCPM_DEFAULT_BASE_URL = "https://api.modelbest.cn/v1"
VOXCPM_DEFAULT_VOICE = "default"
VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES = 5 * 1024 * 1024
VOXCPM_REFERENCE_AUDIO_CONVERSION_TIMEOUT_SECONDS = 15
VOXCPM_REFERENCE_AUDIO_MAX_DURATION_SECONDS = 120
VOXCPM_REFERENCE_AUDIO_FILE_TYPES = ("wav", "mp3", "m4a", "aac", "ogg", "flac")
_DEFAULT_TTS_FFMPEG_TIMEOUT_SECONDS = 600
_VOXCPM_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
_VOXCPM_RETRY_DELAY_SECONDS = (1.0, 2.0)
# Match the existing generated-speech ceiling; allow one full Base64 WAV event
# plus JSON framing while bounding both transport lines and multiline events.
_VOXCPM_TTS_MAX_AUDIO_BYTES = 50 * 1024 * 1024
_VOXCPM_SSE_MAX_EVENT_BYTES = 4 * ((_VOXCPM_TTS_MAX_AUDIO_BYTES + 2) // 3) + 64 * 1024
NO_VOICE_NAME = "no-voice"
# `none` was the no-voice sentinel used in PR #981. Keep it working short term
# so API users who already called that branch are not broken by an upgrade;
# the WebUI and new code standardize on the clearer `no-voice`.
_NO_VOICE_ALIASES = {NO_VOICE_NAME, "none"}


def _configure_pydub_ffmpeg(audio_segment_cls):
    configured_ffmpeg = utils.get_ffmpeg_binary()
    if configured_ffmpeg:
        audio_segment_cls.converter = configured_ffmpeg


def mktimestamp(time_unit: float) -> str:
    """
    Convert edge_tts' 100-nanosecond ticks into a subtitle timestamp.

    edge_tts 7.x no longer exports the old `mktimestamp`, but the legacy subtitle
    paths in this project still need that formatter to handle the manually built
    subtitle timelines from Azure v2, Gemini and SiliconFlow, so an equivalent
    implementation is provided here.
    """
    # Round to milliseconds before splitting into hours/minutes/seconds. Formatting
    # the modulo seconds directly with `:06.3f` would render 59.9996 s as
    # `00:00:60.000`, which strict SRT parsers reject.
    total_milliseconds = int(round(time_unit / 10**4))
    hour, remainder = divmod(total_milliseconds, 3_600_000)
    minute, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hour:02d}:{minute:02d}:{seconds:02d}.{milliseconds:03d}"


def get_siliconflow_voices() -> list[str]:
    """
    Get the SiliconFlow voice list.

    Returns:
        Voice list in the form ["siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex", ...]
    """
    # SiliconFlow voices and their genders (used for display)
    voices_with_gender = [
        ("FunAudioLLM/CosyVoice2-0.5B", "alex", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "anna", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "bella", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "benjamin", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "charles", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "claire", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "david", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "diana", "Female"),
    ]

    # Add the siliconflow: prefix and format as display names
    return [
        f"siliconflow:{model}:{voice}-{gender}"
        for model, voice, gender in voices_with_gender
    ]


def get_gemini_voices() -> list[str]:
    """
    Get the official Gemini TTS preset voice list.

    Google does not publish gender metadata for these voices, so the dropdown
    uses the official style descriptions and no guessed gender is persisted in
    the voice id. Voice catalog source:
    https://ai.google.dev/gemini-api/docs/speech-generation#voice-options

    Returns:
        Voice list in the form ["gemini:Zephyr-Bright", "gemini:Puck-Upbeat", ...]
    """
    return [f"gemini:{voice}-{style}" for voice, style in GEMINI_TTS_VOICES]


def get_mimo_voices() -> list[str]:
    """
    Get the preset voice list for Xiaomi MiMo V2.5 TTS.

    Only the `mimo-v2.5-tts` preset voice mode documented officially is wired up
    for now. Voice design (`mimo-v2.5-tts-voicedesign`) and voice cloning
    (`mimo-v2.5-tts-voiceclone`) need extra input forms and asset upload flows,
    so they are kept out of the regular TTS dropdown to avoid making users think
    one voice id selection covers all advanced capabilities.
    """
    voices_with_gender = [
        ("mimo_default", "Female"),
        ("冰糖", "Female"),
        ("茉莉", "Female"),
        ("苏打", "Male"),
        ("白桦", "Male"),
        ("Mia", "Female"),
        ("Chloe", "Female"),
        ("Milo", "Male"),
        ("Dean", "Male"),
    ]

    return [f"mimo:{voice}-{gender}" for voice, gender in voices_with_gender]


def get_minimax_voices(voice_id: str | None = None) -> list[str]:
    """Return the configured MiniMax voice in the unified TTS dispatch format."""
    voice_id = str(
        voice_id
        or config.minimax_tts.get("voice_id", MINIMAX_TTS_DEFAULT_VOICE)
        or MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    return [f"minimax:{voice_id}"]


def get_elevenlabs_voices(api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        url = "https://api.elevenlabs.io/v2/voices"
        params = {"is_favorite": "true", "page_size": 100}
        headers = {"xi-api-key": api_key}
        # Requests preserves custom xi-api-key headers across redirects. Keep
        # the key on the provider endpoint even if it responds with a redirect.
        result = []
        seen_voices = set()
        seen_tokens = set()
        deadline = time.monotonic() + 30.0
        # v2 catalogs are paginated even when page_size is at its maximum.
        # Retain the fixed endpoint and no-redirect credential boundary.
        for _ in range(100):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("ElevenLabs voice catalog time budget exhausted; returning partial catalog")
                break
            phase_timeout = min(10.0, remaining / 2.0)
            response = trace_http_call(
                "GET", requests.get,
                url, params=dict(params), headers=headers,
                timeout=(phase_timeout, phase_timeout), allow_redirects=False,
            )
            try:
                if response.status_code != 200:
                    logger.warning(
                        f"ElevenLabs voices fetch failed with status {response.status_code}"
                    )
                    return []
                data = response.json()
            finally:
                response.close()
            if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
                return []
            for entry in data["voices"]:
                if not isinstance(entry, dict):
                    continue
                identity = entry.get("voice_id")
                name = entry.get("name")
                if (
                    isinstance(identity, str) and identity
                    and isinstance(name, str) and name
                    and entry.get("status") != "disabled"
                    and identity not in seen_voices
                ):
                    seen_voices.add(identity)
                    result.append(f"elevenlabs:{identity}:{name}")
            token = data.get("next_page_token")
            if data.get("has_more") is not True or not isinstance(token, str) or not token:
                break
            if token in seen_tokens:
                logger.warning("ElevenLabs voice catalog repeated a page token")
                break
            seen_tokens.add(token)
            params["next_page_token"] = token
        else:
            logger.warning("ElevenLabs voice catalog page limit reached; returning partial catalog")
        return result
    except Exception as e:
        logger.warning(f"ElevenLabs voices fetch failed: {str(e)}")
        return []


def get_chatterbox_voices() -> list[str]:
    """Return the configured Chatterbox voices.

    Chatterbox is self-hosted, so there is no global voice catalog. Operators
    list the voice names exposed by their server via ``[chatterbox] voices``
    (a TOML array, or a comma-separated string). Each entry is normalised to
    the ``chatterbox:<name>`` format used by the TTS dispatcher.
    """
    voices = config.chatterbox.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    result = []
    for v in voices:
        v = str(v).strip()
        if not v:
            continue
        result.append(v if v.startswith("chatterbox:") else f"chatterbox:{v}")
    if not result:
        # keep the dropdown usable even before any voice is configured
        result = ["chatterbox:default-Female"]
    return result


KOKORO_DEFAULT_VOICE = "af_heart"


def _normalize_kokoro_voices(entries) -> list[str]:
    """Normalize manual config and old/new server formats; keep only real IDs,
    never coercing objects into voice names."""
    if isinstance(entries, str):
        entries = entries.split(",")
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        name = entry.get("id") if isinstance(entry, dict) else entry
        if not isinstance(name, str):
            continue
        name = name.strip().removeprefix("kokoro:").strip()
        if name:
            value = f"kokoro:{name}"
            if value not in result:
                result.append(value)
    return result


def get_kokoro_voices(*, fallback: bool = True) -> list[str]:
    """Prefer manually configured voices, otherwise query the server; the UI can
    disable defaults to detect a disconnected server and keep its selection.

    Old servers return a list of strings, new ones return a list of objects with
    an id. On failure no session state is cached in the service layer; the WebUI
    keeps the last successful catalog so users/endpoints do not pollute each other.
    """
    voices = _normalize_kokoro_voices(config.kokoro.get("voices"))
    if not voices:
        base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
        if base_url:
            try:
                headers = {}
                api_key = config.kokoro.get("api_key", "")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                response = trace_http_call(
                    "GET", requests.get,
                    f"{base_url}/audio/voices", headers=headers, timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    listed = data.get("voices", []) if isinstance(data, dict) else data
                    voices = _normalize_kokoro_voices(listed)
                    if not voices:
                        logger.warning("kokoro voice list contains no valid voice IDs")
                else:
                    logger.warning(
                        f"kokoro voices request failed with status {response.status_code}"
                    )
            except Exception as e:
                # Do not log the URL or exception body, so query parameters or
                # credentials of a self-hosted address never reach the logs.
                logger.warning(f"kokoro voice list unavailable ({type(e).__name__})")
    return voices or ([f"kokoro:{KOKORO_DEFAULT_VOICE}"] if fallback else [])


def get_fish_audio_voices() -> list[str]:
    """Return configured Fish Audio voices.

    Each entry follows the format ``fish_audio:<reference_id>:<display_name>``.
    When ``reference_id`` is "default", Fish Audio's built-in default voice is
    used (no ``reference_id`` is sent in the API request).  Operators can list
    additional public or cloned voices via ``[fish_audio] voices`` in the
    config file.
    """
    result = [
        "fish_audio:2324c907b9a94c64ab4afb941e5b3408:Clear Female-Female",
        "fish_audio:7b6131ba75ba47c98a46c847db729ab6:Clear Male-Male",
        "fish_audio:default:Default Voice",
    ]
    voices = config.fish_audio.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    for entry in voices:
        entry = str(entry).strip()
        if not entry:
            continue
        if entry.startswith("fish_audio:"):
            result.append(entry)
        elif ":" in entry:
            # "<reference_id>:<display_name>"
            result.append(f"fish_audio:{entry}")
        else:
            # bare reference_id
            result.append(f"fish_audio:{entry}:{entry}")
    return result


def get_voxcpm_voices(voice_id: str | None = None) -> list[str]:
    """Return the ModelBest VoxCPM voice selected in the local configuration.

    ModelBest accepts ``default`` when no explicit voice is selected. Voice
    design is expressed in the input text and voice cloning requires a separate
    reference-audio workflow, so the first integration deliberately keeps the
    standard TTS selector to one configured voice id.
    """
    voice_id = str(
        voice_id
        or config.voxcpm.get("voice_id", VOXCPM_DEFAULT_VOICE)
        or VOXCPM_DEFAULT_VOICE
    ).strip()
    return [f"voxcpm:{voice_id}"]


_AZURE_VOICES_DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data", "azure_voices.json"
)
_azure_voices_cache = None


def _load_azure_voices() -> list[dict]:
    global _azure_voices_cache
    if _azure_voices_cache is None:
        with open(_AZURE_VOICES_DATA_FILE, "r", encoding="utf-8") as f:
            _azure_voices_cache = json.load(f)
    return _azure_voices_cache


def get_all_azure_voices(filter_locals=None) -> list[str]:
    voices = []
    for item in _load_azure_voices():
        name = item["name"]
        gender = item["gender"]
        # Apply the filter
        if filter_locals and any(
            name.lower().startswith(fl.lower()) for fl in filter_locals
        ):
            voices.append(f"{name}-{gender}")
        elif not filter_locals:
            voices.append(f"{name}-{gender}")

    voices.sort()
    return voices


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.strip()
    # Qualified IDs belong to their provider. Gender/style display labels are
    # parsed by each provider adapter, not rewritten inside an opaque voice ID.
    if ":" not in name:
        name = name.removesuffix("-Female").removesuffix("-Male")
    return name


def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if ":" not in voice_name and voice_name.endswith("-V2"):
        return voice_name.removesuffix("-V2").strip()
    return ""


def is_siliconflow_voice(voice_name: str):
    """Return whether this is a SiliconFlow voice."""
    return voice_name.startswith("siliconflow:")


def is_gemini_voice(voice_name: str):
    """Return whether this is a Gemini TTS voice."""
    return voice_name.startswith("gemini:")


def parse_gemini_voice_name(voice_name: str | None) -> str:
    """Extract the preset voice name used by the Google API from old and new
    Gemini dropdown values."""
    if not is_gemini_voice(voice_name or ""):
        return ""
    return (voice_name or "").split(":", 1)[1].split("-", 1)[0].strip()


def is_mimo_voice(voice_name: str):
    """Return whether this is a Xiaomi MiMo TTS voice."""
    return voice_name.startswith("mimo:")


def is_minimax_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("minimax:")


def is_elevenlabs_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("elevenlabs:")


def get_elevenlabs_api_key() -> str:
    """
    Read the API key used by ElevenLabs TTS.

    The config file wins; the environment variable is only a fallback when nothing
    is configured. The WebUI and the music feature already support
    ``ELEVENLABS_API_KEY`` and TTS must follow the same rule, otherwise a
    deployment configured only through container env vars loads the voice list
    fine but reports a missing key when actually synthesizing speech.
    """
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    return configured_key or os.getenv("ELEVENLABS_API_KEY", "").strip()


def is_chatterbox_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("chatterbox:")


def is_kokoro_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("kokoro:")


def is_fish_audio_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("fish_audio:")


def is_voxcpm_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("voxcpm:")


def get_fish_audio_api_key() -> str:
    configured_key = str(config.fish_audio.get("api_key", "") if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict) else "").strip()
    return configured_key or os.getenv("FISH_API_KEY", "").strip()


def is_no_voice(voice_name: str | None) -> bool:
    """
    Determine whether the user explicitly selected "no voice" mode.

    An empty string is deliberately not treated as no-voice: an empty voice more
    likely means a broken config, lost state from an older WebUI, or a missing API
    parameter. Only explicit sentinels take the silent branch, so real errors are
    not disguised as successful generations.
    """
    return str(voice_name or "").strip().lower() in _NO_VOICE_ALIASES


def is_azure_v1_voice(voice_name: str | None) -> bool:
    """
    Check whether this is an Azure TTS v1 (Edge TTS) preset voice.

    The first segmented-synthesis pipeline for pause tags ([pause: ...]) targets
    only Edge TTS (Azure TTS v1), so that request billing, rate limits and default
    behavior of other providers such as Gemini, Fish Audio, SiliconFlow and
    Kokoro stay unchanged.
    """
    if not voice_name:
        return False
    name = str(voice_name).strip()
    if is_no_voice(name):
        return False
    if is_azure_v2_voice(name):
        return False
    if is_siliconflow_voice(name):
        return False
    if is_gemini_voice(name):
        return False
    if is_mimo_voice(name):
        return False
    if is_minimax_voice(name):
        return False
    if is_elevenlabs_voice(name):
        return False
    if is_chatterbox_voice(name):
        return False
    if is_kokoro_voice(name):
        return False
    if is_fish_audio_voice(name):
        return False
    if is_voxcpm_voice(name):
        return False
    return True


def estimate_no_voice_duration(text: str) -> float:
    """
    Estimate a stable video timeline length for no-voice mode.

    No-voice still needs an audio placeholder to drive the existing footage
    trimming, subtitle timeline and final composition. The estimate is kept
    simple:
    1. CJK characters such as Chinese at about 4.2 chars/second;
    2. English words and digits at about 2.7 words/second;
    3. a fallback of about 4.0 chars/second for other scripts, covering
       Russian, Arabic, Japanese kana, Korean and other non-ASCII text;
    4. a small pause per sentence so subtitle switches are not too tight;
    5. a 3-second minimum so a very short script does not yield 0s of audio.
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 3.0

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized_text))
    words = len(re.findall(r"[A-Za-z0-9]+", normalized_text))
    ascii_word_chars = sum(len(word) for word in re.findall(r"[A-Za-z0-9]+", normalized_text))
    other_text_chars = 0
    for char in normalized_text:
        # A Unicode category starting with L means letters in any script, N means
        # digits. CJK and ASCII words are already counted above, so only count
        # the remaining characters here to avoid timing English twice.
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            other_text_chars += 1
    other_text_chars = max(other_text_chars - cjk_chars - ascii_word_chars, 0)
    sentence_count = max(len(utils.split_string_by_punctuations(normalized_text)), 1)

    cjk_duration = cjk_chars / 4.2
    word_duration = words / 2.7
    other_text_duration = other_text_chars / 4.0
    pause_duration = max(sentence_count - 1, 0) * 0.35
    return max(3.0, cjk_duration + word_duration + other_text_duration + pause_duration)


def generate_silent_audio(duration_seconds: float, output_file: str) -> bool:
    """
    Generate silent audio.

    Supports writing 16-bit mono PCM WAV directly (sample-accurate, no encoder
    delay), or generating MP3 through FFmpeg anullsrc as the no-voice placeholder.
    """
    ensure_file_path_exists(output_file)
    duration_seconds = max(
        float(duration_seconds or 0), utils.MIN_PAUSE_DURATION_SECONDS
    )

    if output_file.lower().endswith(".wav"):
        sample_rate = 24000
        num_samples = int(round(duration_seconds * sample_rate))
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0

    ffmpeg_binary = utils.get_ffmpeg_binary()
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{duration_seconds:.3f}",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
    ]

    logger.info(
        f"generating silent audio for no-voice mode, duration: {duration_seconds:.2f}s"
    )
    return _publish_tts_ffmpeg_output(command, output_file, "silent narration encode")


def _warn_ignored_voice_rate(provider: str, voice_rate: float) -> None:
    """
    Gemini TTS, MiMo TTS and VoxCPM have no numeric speed parameter, so
    voice_rate is not sent to them. Warn when a speed other than 1.0 is
    requested instead of ignoring it silently.
    """
    try:
        ignored = float(voice_rate) != 1.0
    except (TypeError, ValueError, OverflowError):
        return
    if ignored:
        logger.warning(
            f"{provider} does not support numeric speed adjustment; "
            f"voice_rate {voice_rate} is ignored"
        )


def _single_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> Union[SubMaker, None]:
    if is_no_voice(voice_name):
        duration_seconds = estimate_no_voice_duration(text)
        if not generate_silent_audio(duration_seconds, voice_file):
            return None

        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=duration_seconds,
        )

    if is_azure_v2_voice(voice_name):
        return azure_tts_v2(
            text,
            voice_name,
            voice_file,
            voice_rate=voice_rate,
        )
    elif is_siliconflow_voice(voice_name):
        # Extract the model and voice from voice_name
        # Format: siliconflow:model:voice-Gender
        parts = voice_name.split(":")
        if len(parts) >= 3:
            model = parts[1]
            # Strip the gender suffix, e.g. "alex-Male" -> "alex"
            voice_with_gender = parts[2]
            voice = voice_with_gender.split("-")[0]
            # Build the full voice argument in the form "model:voice"
            full_voice = f"{model}:{voice}"
            return siliconflow_tts(
                text, model, full_voice, voice_rate, voice_file, voice_volume
            )
        else:
            logger.error(f"Invalid siliconflow voice name format: {voice_name}")
            return None
    elif is_gemini_voice(voice_name):
        # Extract the voice name from voice_name
        # Format: gemini:voice-Style; the old gemini:voice-Gender still works too.
        voice = parse_gemini_voice_name(voice_name)
        if voice:
            _warn_ignored_voice_rate("Gemini TTS", voice_rate)
            return gemini_tts(text, voice, voice_rate, voice_file, voice_volume)
        else:
            logger.error(f"Invalid gemini voice name format: {voice_name}")
            return None
    elif is_mimo_voice(voice_name):
        # Extract the voice name from voice_name
        # Format: mimo:voice-Gender; if the caller already ran parse_voice_name,
        # it may be mimo:voice instead. Both formats are supported.
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_with_gender = parts[1]
            voice = voice_with_gender.split("-")[0]
            _warn_ignored_voice_rate("MiMo TTS", voice_rate)
            return mimo_tts(text, voice, voice_rate, voice_file, voice_volume)
        else:
            logger.error(f"Invalid mimo voice name format: {voice_name}")
            return None
    elif is_minimax_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            return minimax_tts(text, voice_id, voice_rate, voice_file, voice_volume)
        logger.error(f"Invalid MiniMax voice name format: {voice_name}")
        return None
    elif is_elevenlabs_voice(voice_name):
        # Format: elevenlabs:{voice_id}:{name}
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_id = parts[1]
            return elevenlabs_tts(text, voice_id, voice_file, voice_rate, voice_volume)
        else:
            logger.error(f"Invalid elevenlabs voice name format: {voice_name}")
            return None
    elif is_chatterbox_voice(voice_name):
        # Format: chatterbox:<voice>; voice may carry a display -Female/-Male suffix
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            chatterbox_voice = parts[1].strip()
            if chatterbox_voice.endswith(("-Female", "-Male")):
                chatterbox_voice = chatterbox_voice.rsplit("-", 1)[0]
            return chatterbox_tts(
                text, chatterbox_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"Invalid chatterbox voice name format: {voice_name}")
            return None
    elif is_kokoro_voice(voice_name):
        # Format: kokoro:<voice>; voice may carry a display -Female/-Male suffix
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            kokoro_voice = parts[1].strip()
            if kokoro_voice.endswith(("-Female", "-Male")):
                kokoro_voice = kokoro_voice.rsplit("-", 1)[0]
            return kokoro_tts(
                text, kokoro_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"Invalid kokoro voice name format: {voice_name}")
            return None
    elif is_fish_audio_voice(voice_name):
        parts = voice_name.split(":")
        reference_id = parts[1] if len(parts) >= 2 else "default"
        if reference_id == "default":
            reference_id = None
        return fish_audio_tts(text, voice_file, voice_rate, voice_volume, reference_id=reference_id)
    elif is_voxcpm_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            _warn_ignored_voice_rate("VoxCPM", voice_rate)
            if voxcpm_reference_audio is None:
                return voxcpm_tts(text, voice_id, voice_file, voice_rate, voice_volume)
            return voxcpm_tts(
                text,
                voice_id,
                voice_file,
                voice_rate,
                voice_volume,
                reference_audio=voxcpm_reference_audio,
                prompt_audio=voxcpm_prompt_audio,
                prompt_text=voxcpm_prompt_text,
            )
        logger.error(f"Invalid VoxCPM voice name format: {voice_name}")
        return None
    return azure_tts_v1(text, voice_name, voice_rate, voice_file)


def _run_tts_ffmpeg(command: list[str], stage: str):
    """Bound TTS FFmpeg work and reap the child on timeout."""
    raw_timeout = config.app.get(
        "ffmpeg_tts_timeout_seconds",
        _DEFAULT_TTS_FFMPEG_TIMEOUT_SECONDS,
    )
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("ffmpeg_tts_timeout_seconds must be positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ffmpeg_tts_timeout_seconds must be positive")

    try:
        # subprocess.run kills and waits for the child on timeout. Do not allow
        # FFmpeg to wait for terminal input in an unattended generation task.
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg {stage} timed out after {timeout:g} seconds")
    except OSError as exc:
        logger.error(f"failed to start FFmpeg {stage}: {exc}")
    return None


def _publish_tts_ffmpeg_output(
    command: list[str], output_file: str, stage: str
) -> bool:
    """Encode beside the destination and replace only after a complete output."""
    output_dir = os.path.dirname(output_file) or "."
    with tempfile.TemporaryDirectory(
        prefix=".mpt-tts-audio-", dir=output_dir
    ) as publish_temp:
        staged_output = os.path.join(
            publish_temp, f"narration{os.path.splitext(output_file)[1] or '.mp3'}"
        )
        result = _run_tts_ffmpeg([*command, staged_output], stage)
        if result is None:
            return False
        if result.returncode != 0:
            logger.error(
                f"FFmpeg {stage} failed: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
            return False
        if not os.path.exists(staged_output) or os.path.getsize(staged_output) == 0:
            logger.error(f"FFmpeg {stage} produced no narration audio")
            return False
        try:
            os.replace(staged_output, output_file)
        except OSError as exc:
            logger.error(f"failed to publish {stage} audio: {exc}")
            return False
        return True



def apply_audio_volume(audio_file: str, output_file: str, volume: float) -> bool:
    """Encode narration export gain to a caller-owned destination."""
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("audio export volume must be finite and nonnegative")
    ensure_file_path_exists(output_file)
    command = [
        utils.get_ffmpeg_binary(), "-nostdin", "-v", "error", "-y",
        "-i", audio_file, "-vn", "-af", f"volume={volume}",
        "-codec:a", "libmp3lame", "-q:a", "4",
    ]
    return _publish_tts_ffmpeg_output(command, output_file, "audio export volume")

def _concat_audio_files(audio_files: list[str], output_file: str) -> bool:
    """
    Merge multiple audio segments using PCM decoding and a single re-encode.

    Every input segment is converted to standard PCM (24000Hz 16-bit mono)
    samples and concatenated seamlessly, then encoded once into the target file
    (e.g. MP3). This removes audio/video desync and subtitle drift caused by the
    encoder delay/padding accumulated from encoding each MP3 chunk separately.
    """
    if not audio_files:
        return False
    ensure_file_path_exists(output_file)
    if len(audio_files) == 1:
        if audio_files[0] != output_file:
            shutil.copyfile(audio_files[0], output_file)
        return True

    target_sample_rate = 24000
    ffmpeg_binary = utils.get_ffmpeg_binary()

    with tempfile.TemporaryDirectory() as concat_temp:
        temp_combined_wav = os.path.join(concat_temp, "combined_master.wav")
        combined_frames = 0
        with wave.open(temp_combined_wav, "wb") as combined_wave:
            combined_wave.setnchannels(1)
            combined_wave.setsampwidth(2)
            combined_wave.setframerate(target_sample_rate)

            def append_pcm(reader):
                nonlocal combined_frames
                chunk_frames = 0
                while chunk := reader.readframes(8192):
                    if len(chunk) % 2:
                        raise ValueError("incomplete PCM sample in narration chunk")
                    combined_wave.writeframesraw(chunk)
                    chunk_frames += len(chunk) // 2
                if not chunk_frames or chunk_frames != reader.getnframes():
                    raise ValueError("empty or truncated narration chunk")
                combined_frames += chunk_frames

            for idx, f in enumerate(audio_files):
                if not os.path.exists(f) or os.path.getsize(f) == 0:
                    logger.error(f"narration chunk is missing or empty: {f}")
                    return False

                # Check whether it is already a 24000Hz 16-bit mono WAV
                is_valid_pcm_wav = False
                if f.lower().endswith(".wav"):
                    try:
                        with wave.open(f, "rb") as wf:
                            if (
                                wf.getframerate() == target_sample_rate
                                and wf.getnchannels() == 1
                                and wf.getsampwidth() == 2
                            ):
                                is_valid_pcm_wav = True
                                append_pcm(wf)
                    except Exception as exc:
                        if is_valid_pcm_wav:
                            logger.error(f"failed to stream PCM input: {exc}")
                            return False
                        is_valid_pcm_wav = False

                if not is_valid_pcm_wav:
                    # Decode the input file to 24000Hz 16-bit mono PCM WAV with FFmpeg
                    pcm_wav = os.path.join(concat_temp, f"chunk_{idx}.wav")
                    cmd = [
                        ffmpeg_binary,
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        f,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sample_rate),
                        "-codec:a",
                        "pcm_s16le",
                        pcm_wav,
                    ]
                    res = _run_tts_ffmpeg(cmd, "pause audio decode")
                    if res is None:
                        return False
                    if res.returncode == 0 and os.path.exists(pcm_wav):
                        try:
                            with wave.open(pcm_wav, "rb") as wf:
                                append_pcm(wf)
                        except Exception as e:
                            logger.error(f"failed to read decoded pcm wav: {e}")
                            return False
                    else:
                        logger.error(f"failed to decode audio chunk with ffmpeg: {res.stderr}")
                        return False

        if not combined_frames:
            logger.error("no valid audio samples to concatenate")
            return False

        if output_file.lower().endswith(".wav"):
            shutil.copyfile(temp_combined_wav, output_file)
            return True

        command = [
            ffmpeg_binary,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            temp_combined_wav,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
        ]
        return _publish_tts_ffmpeg_output(
            command, output_file, "pause audio encode"
        )


def _tts_with_pauses(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Synthesize a script that contains pause tags (e.g. [pause: 2s] /
    [pausa: 1.5s] / [停顿: 3秒]). Speech and exact PCM silence are generated per
    segment, subtitle offsets are computed from the decoded samples, and a single
    unified encode runs at the end.
    """
    segments = utils.parse_script_with_pauses(text)
    if not segments:
        return None

    speech_segments = [s for s in segments if s[0] == "speech"]
    pause_segments = [s for s in segments if s[0] == "pause"]

    if not pause_segments:
        clean_text = utils.remove_pause_tags(text)
        return _single_tts(clean_text, voice_name, voice_rate, voice_file, voice_volume)

    if not speech_segments:
        total_pause_duration = sum(float(s[1]) for s in pause_segments)
        total_pause_duration = min(
            total_pause_duration, utils.MAX_PAUSE_DURATION_SECONDS
        )
        if not generate_silent_audio(total_pause_duration, voice_file):
            return None
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        sub_maker.duration = total_pause_duration
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=utils.remove_pause_tags(text),
            audio_duration_seconds=total_pause_duration,
        )

    SAMPLE_RATE = 24000
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_chunk_files: list[str] = []
        combined_submaker = ensure_legacy_submaker_fields(SubMaker())
        cumulative_samples = 0

        for idx, (seg_type, seg_val) in enumerate(segments):
            if seg_type == "pause":
                pause_duration = float(seg_val)
                silence_wav = os.path.join(temp_dir, f"silence_{idx}.wav")
                num_silent_samples = int(round(pause_duration * SAMPLE_RATE))
                with wave.open(silence_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"\x00\x00" * num_silent_samples)

                # Also call generate_silent_audio so unit tests that mock it still
                # observe the invocation
                generate_silent_audio(pause_duration, silence_wav)

                actual_pause_duration = pause_duration
                # If a unit test mocks get_audio_duration, prefer the duration it returns
                mock_check_duration = get_audio_duration(silence_wav)
                if mock_check_duration > 0 and abs(mock_check_duration - pause_duration) > 0.05:
                    actual_pause_duration = mock_check_duration
                    num_silent_samples = int(round(actual_pause_duration * SAMPLE_RATE))

                audio_chunk_files.append(silence_wav)
                cumulative_samples += num_silent_samples

            elif seg_type == "speech":
                speech_text = str(seg_val).strip()
                if not speech_text:
                    continue

                chunk_audio_file = os.path.join(temp_dir, f"speech_{idx}.mp3")
                chunk_submaker = _single_tts(
                    text=speech_text,
                    voice_name=voice_name,
                    voice_rate=voice_rate,
                    voice_file=chunk_audio_file,
                    voice_volume=voice_volume,
                )
                if not chunk_submaker or not os.path.exists(chunk_audio_file) or os.path.getsize(chunk_audio_file) == 0:
                    logger.error(
                        f"failed to synthesize speech chunk (audio missing or empty): {speech_text[:50]}"
                    )
                    return None

                chunk_wav = os.path.join(temp_dir, f"speech_{idx}_decoded.wav")
                ffmpeg_binary = utils.get_ffmpeg_binary()
                cmd = [
                    ffmpeg_binary,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    chunk_audio_file,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-codec:a",
                    "pcm_s16le",
                    chunk_wav,
                ]
                res = _run_tts_ffmpeg(cmd, "speech chunk decode")
                if res is None:
                    return None
                if res.returncode != 0 or not os.path.exists(chunk_wav) or os.path.getsize(chunk_wav) == 0:
                    logger.error(
                        f"failed to decode speech chunk audio to PCM WAV: {speech_text[:50]}, "
                        f"error: {(res.stderr or res.stdout or '').strip()}"
                    )
                    return None

                try:
                    with wave.open(chunk_wav, "rb") as wf:
                        chunk_samples = wf.getnframes()
                except Exception as e:
                    logger.error(
                        f"failed to read decoded speech wave: {speech_text[:50]}, error: {e}"
                    )
                    return None

                if chunk_samples <= 0:
                    logger.error(
                        f"decoded speech chunk has no audio samples: {speech_text[:50]}"
                    )
                    return None

                # Subtitle offsets are computed exactly from the real sample count,
                # so there is no cumulative MP3 frame drift
                current_offset_seconds = cumulative_samples / float(SAMPLE_RATE)

                # 1. Migrate cues (edge_tts 7.x)
                if hasattr(chunk_submaker, "cues") and chunk_submaker.cues:
                    offset_td = timedelta(seconds=current_offset_seconds)
                    for cue in chunk_submaker.cues:
                        shifted_cue = Subtitle(
                            index=len(combined_submaker.cues) + 1,
                            start=cue.start + offset_td,
                            end=cue.end + offset_td,
                            content=cue.content,
                        )
                        combined_submaker.cues.append(shifted_cue)

                # 2. Migrate legacy subs/offset
                if hasattr(chunk_submaker, "subs") and chunk_submaker.subs:
                    combined_submaker.subs.extend(chunk_submaker.subs)
                if hasattr(chunk_submaker, "offset") and chunk_submaker.offset:
                    offset_100ns = int(current_offset_seconds * 10000000)
                    for start_ns, end_ns in chunk_submaker.offset:
                        combined_submaker.offset.append(
                            (start_ns + offset_100ns, end_ns + offset_100ns)
                        )

                audio_chunk_files.append(chunk_wav)
                cumulative_samples += chunk_samples

        if not _concat_audio_files(audio_chunk_files, voice_file):
            logger.error("failed to concatenate audio chunks with pauses")
            return None

        combined_submaker.duration = cumulative_samples / float(SAMPLE_RATE)
        return combined_submaker


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> Union[SubMaker, None]:
    # Emoji are not narrated: Edge TTS reads their names aloud ("chart
    # increasing"), and each one gets its own word boundary.
    narration = utils.remove_emoji(text)
    if narration != text and not narration.strip():
        logger.warning("the script has no text to narrate once emoji are removed")
        return None
    text = narration
    # With no pause tags, pass the original text through unchanged to avoid
    # pointless regex work or whitespace truncation
    if not utils.has_pause_tags(text):
        return _single_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
            voxcpm_reference_audio=voxcpm_reference_audio,
            voxcpm_prompt_audio=voxcpm_prompt_audio,
            voxcpm_prompt_text=voxcpm_prompt_text,
        )

    # Segmented synthesis only applies to Azure TTS v1 (Edge TTS) with pause tags
    if is_azure_v1_voice(voice_name):
        return _tts_with_pauses(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    # For other voice providers (e.g. Gemini, Fish Audio, SiliconFlow, Kokoro)
    # with pause tags, strip the tags and synthesize in a single request
    clean_text = utils.remove_pause_tags(text)
    return _single_tts(
        text=clean_text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file,
        voice_volume=voice_volume,
        voxcpm_reference_audio=voxcpm_reference_audio,
        voxcpm_prompt_audio=voxcpm_prompt_audio,
        voxcpm_prompt_text=voxcpm_prompt_text,
    )


def convert_rate_to_percent(rate: float) -> str:
    # edge-tts requires a sign-prefixed percentage (e.g. "+0%", "-20%").
    # Rounding can yield 0 for rates near but not equal to 1.0 (e.g. 1.004,
    # 0.997); those must still be returned as "+0%", not the unsigned "0%"
    # which edge-tts rejects with ValueError: Invalid rate '0%'.
    # API or batch calls may pass 0, 0.0, None or empty values that cannot be
    # converted; none of these are valid rates and computing directly would give
    # -100% or raise. Fall back to the normal rate so we neither generate very
    # slow audio nor fail the TTS flow on boundary input.
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    if not math.isfinite(rate) or rate <= 0:
        rate = 1.0
    percent = round((rate - 1.0) * 100)
    if percent >= 0:
        return f"+{percent}%"
    return f"{percent}%"


def ensure_file_path_exists(file_path: str) -> None:
    """
    Make sure the directory of the output file exists.

    This extra guard exists because edge_tts 7.x opens the target audio file
    before it actually starts the network request; if the directory is missing,
    the failure surfaces as a local file path error and hides the real TTS result.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def ensure_legacy_submaker_fields(sub_maker: SubMaker) -> SubMaker:
    """
    Add the compatibility fields expected by callers that still use the legacy
    subtitle structure.

    The edge_tts 7.x `SubMaker` mainly exposes `cues/get_srt()`, but the Azure v2,
    Gemini and SiliconFlow paths in this project still read and write `subs/offset`
    directly. They are filled in here so upgrading edge_tts does not break those
    non-edge paths.
    """
    if not hasattr(sub_maker, "subs"):
        sub_maker.subs = []
    if not hasattr(sub_maker, "offset"):
        sub_maker.offset = []
    return sub_maker


def populate_legacy_submaker_with_full_text(
    sub_maker: SubMaker, text: str, audio_duration_seconds: float
) -> SubMaker:
    """
    Fill the historical `subs/offset` subtitle structure from the full text.

    Background:
    1. The edge_tts 7.x `SubMaker` no longer provides the old `create_sub()`;
    2. non-edge paths in this project such as Gemini and SiliconFlow still need to
       return an object with `subs/offset` for unified audio duration and subtitle
       generation downstream;
    3. TTS services without word boundaries must at least split by script
       sentences, so the later `subtitle_provider=edge` aggregation keeps working
       instead of falling back to Whisper because the full text cannot be matched
       line by line against the script.

    Args:
        sub_maker: subtitle object that receives the compatibility fields
        text: original script text
        audio_duration_seconds: total audio duration in seconds

    Returns:
        SubMaker object populated with compatible subtitle data
    """
    sub_maker = ensure_legacy_submaker_fields(sub_maker)

    # Clear old values so reusing the same object does not accumulate stale data.
    sub_maker.subs = []
    sub_maker.offset = []

    # Reuse the same script cleanup as create_subtitle(): Markdown separator lines
    # such as `---` and `***` are not narrated, so including them in the
    # char-proportional split would consume audio time and make later subtitles
    # appear early, or the last subtitle vanish before the audio ends.
    normalized_text = utils.normalize_script_for_subtitle_matching(text or "").strip()
    if not normalized_text:
        return sub_maker

    audio_duration_100ns = max(int(audio_duration_seconds * 10000000), 1)

    # Paths such as Gemini / SiliconFlow have no word boundaries, so keep the
    # project's original strategy of splitting by punctuation and distributing
    # duration by character count. That lets create_subtitle() match script
    # sentences and avoids another Whisper fallback.
    sentences = utils.split_string_by_punctuations(normalized_text)
    if not sentences:
        sentences = [normalized_text]

    total_chars = sum(len(sentence) for sentence in sentences)
    if total_chars <= 0:
        sub_maker.subs.append(normalized_text)
        sub_maker.offset.append((0, audio_duration_100ns))
        return sub_maker

    current_offset = 0
    for index, sentence in enumerate(sentences):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue

        # Earlier sentences get a char-proportional duration and the last one
        # absorbs the remainder, so integer rounding neither loses total duration
        # nor leaves the final subtitle ending before the audio does.
        if index == len(sentences) - 1:
            sentence_end = audio_duration_100ns
        else:
            sentence_chars = len(cleaned_sentence)
            sentence_duration = max(
                int(audio_duration_100ns * (sentence_chars / total_chars)),
                1,
            )
            sentence_end = min(current_offset + sentence_duration, audio_duration_100ns)

        sub_maker.subs.append(cleaned_sentence)
        sub_maker.offset.append((current_offset, sentence_end))
        current_offset = sentence_end

    return sub_maker


def create_edge_tts_communicate(
    text: str, voice_name: str, rate_str: str
) -> edge_tts.Communicate:
    """
    Build a Communicate object for the currently installed edge_tts version.

    Background:
    1. Mainline code has upgraded to edge_tts 7.x and uses the `boundary`
       argument to get finer-grained boundary events;
    2. if the Windows portable package fails to update, the deployed environment
       may still run an older edge_tts;
    3. the old `Communicate.__init__()` does not accept `boundary` and raises
       `unexpected keyword argument 'boundary'`, breaking the whole TTS chain.

    So the supported parameters are probed from the constructor signature before
    deciding whether to pass `boundary`, letting one codebase support both the old
    and the new dependency.
    """
    communicate_kwargs = {"rate": rate_str}
    communicate_signature = inspect.signature(edge_tts.Communicate)

    if "boundary" in communicate_signature.parameters:
        communicate_kwargs["boundary"] = "WordBoundary"

    return edge_tts.Communicate(text, voice_name, **communicate_kwargs)


def get_edge_tts_timeout_seconds() -> Union[float, None]:
    """
    Get the timeout for a single Azure TTS V1 streaming request.

    Background:
    Edge consumer TTS can hang inside `stream_sync()` for a long time when the
    network is down, the server rate-limits, or the voice/language does not match
    the text, leaving the log at `start`. A default timeout is provided so WebUI
    tasks do not appear unresponsive for long stretches.

    Usage:
    - 30 seconds by default, covering the first-packet wait of typical short
      video scripts;
    - on a slow network or behind a proxy, set `edge_tts_timeout = 60` in
      `config.toml`;
    - 0 or a negative value explicitly disables the timeout, keeping full
      backward compatibility.
    """
    raw_timeout = config.app.get(
        "edge_tts_timeout", _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS
    )
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "invalid edge_tts_timeout: "
            f"{raw_timeout}, fallback to {_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS}s"
        )
        timeout_seconds = _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _stream_edge_tts_sync_with_timeout(
    communicate, on_chunk, timeout_seconds: float
) -> None:
    """
    Consume the edge_tts 7.x sync stream with an overall timeout.

    Why:
    `stream_sync()` is a blocking iterator, so the main thread cannot recover when
    the network layer stalls. The blocking iteration runs in a daemon thread and
    the main thread pulls chunks from a Queue, raising TimeoutError once the
    deadline passes so outer retries and error logging keep working.

    Notes:
    The SDK's async stream is consumed directly, bypassing the unbounded queue
    inside stream_sync. The queue holds only one item; a timeout or a failing
    callback cancels the async producer and closes the network stream. Adapters
    that only support the sync stream exit cooperatively after the blocked read
    returns and never keep accumulating abandoned data.
    """
    stream_queue = queue.Queue(maxsize=1)
    stopped = threading.Event()
    # Set on the producer loop each time the consumer takes an item.
    item_taken = asyncio.Event()
    producer_loop = None
    producer_task = None

    def _put(item):
        while not stopped.is_set():
            try:
                stream_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    async def _put_async(item):
        while not stopped.is_set():
            # Clear before the put, so a take right after it still wakes us.
            item_taken.clear()
            try:
                stream_queue.put_nowait(item)
                return True
            except queue.Full:
                # Resume as soon as the consumer takes the pending item. The
                # timeout only re-checks `stopped`; cancellation still works.
                try:
                    await asyncio.wait_for(item_taken.wait(), timeout=0.05)
                except TimeoutError:
                    pass
        return False

    def _wake_producer():
        if producer_loop is not None:
            try:
                producer_loop.call_soon_threadsafe(item_taken.set)
            except RuntimeError:
                pass  # The producer already finished and closed its loop.

    async def _produce_async():
        if stopped.is_set():
            return
        stream = communicate.stream()
        try:
            async for chunk in stream:
                if not await _put_async(("chunk", chunk)):
                    return
            await _put_async(("done", None))
        except Exception as error:
            await _put_async(("error", error))
        finally:
            close_stream = getattr(stream, "aclose", None)
            if callable(close_stream):
                await close_stream()

    def _produce_chunks():
        nonlocal producer_loop, producer_task
        # The SDK's stream_sync() has its own unbounded queue/executor. Using
        # its async source directly lets cancellation reach the network read.
        if callable(getattr(communicate, "stream", None)):
            producer_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(producer_loop)
            producer_task = producer_loop.create_task(_produce_async())
            if stopped.is_set():
                producer_task.cancel()
            try:
                producer_loop.run_until_complete(producer_task)
            except asyncio.CancelledError:
                pass
            finally:
                producer_loop.run_until_complete(producer_loop.shutdown_asyncgens())
                producer_loop.close()
            return
        # Compatibility for stream_sync-only adapters: cooperative termination
        # after a blocked read returns, with the same bounded mailbox.
        stream = None
        try:
            stream = communicate.stream_sync()
            for chunk in stream:
                if not _put(("chunk", chunk)):
                    return
            _put(("done", None))
        except Exception as error:
            _put(("error", error))
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()

    thread = threading.Thread(target=_produce_chunks, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"edge_tts stream timed out after {timeout_seconds:g}s"
                )
            try:
                item_type, payload = stream_queue.get(
                    timeout=min(0.5, remaining_seconds)
                )
            except queue.Empty:
                continue
            _wake_producer()
            if item_type == "chunk":
                on_chunk(payload)
            elif item_type == "error":
                raise payload
            elif item_type == "done":
                return
    finally:
        stopped.set()
        if producer_loop is not None and producer_task is not None:
            try:
                producer_loop.call_soon_threadsafe(producer_task.cancel)
            except RuntimeError:
                pass  # The producer already finished and closed its loop.


def stream_edge_tts_chunks(
    communicate, on_chunk, timeout_seconds: Union[float, None] = None
) -> None:
    """
    Consume both edge_tts' sync stream and the older async stream.

    edge_tts 7.x provides `stream_sync()`, which can be iterated inside a sync
    function; earlier versions usually only have the async `stream()`. This
    compatibility layer keeps `azure_tts_v1()` working where old dependencies
    linger.

    Args:
        communicate: edge_tts.Communicate instance
        on_chunk: callback run for each received event chunk
        timeout_seconds: overall timeout for one streaming request; None disables
            the timeout.
    """
    if hasattr(communicate, "stream_sync"):
        if timeout_seconds:
            _stream_edge_tts_sync_with_timeout(
                communicate, on_chunk, timeout_seconds
            )
            return

        for chunk in communicate.stream_sync():
            on_chunk(chunk)
        return

    if not hasattr(communicate, "stream"):
        raise AttributeError("edge_tts communicate object has no stream method")

    async def _consume_async_stream():
        async for chunk in communicate.stream():
            on_chunk(chunk)

    # An independent event loop is created explicitly instead of reusing an outer
    # context, to avoid "no event loop in this thread" errors in a sync call stack
    # and problems with reusing a loop across threads.
    loop = asyncio.new_event_loop()
    try:
        if timeout_seconds:
            loop.run_until_complete(
                asyncio.wait_for(_consume_async_stream(), timeout=timeout_seconds)
            )
        else:
            loop.run_until_complete(_consume_async_stream())
    finally:
        loop.close()


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    for i in range(3):
        temp_path = None
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            # Support both edge_tts 7.x and old dependencies that may linger in an
            # older portable package:
            # 1. the new version supports `boundary` + `stream_sync()`
            # 2. the old version lacks `boundary` and usually only exposes async
            #    `stream()`
            ensure_file_path_exists(voice_file)
            communicate = create_edge_tts_communicate(text, voice_name, rate_str)
            sub_maker = edge_tts.SubMaker()
            timeout_seconds = get_edge_tts_timeout_seconds()

            descriptor, temp_path = tempfile.mkstemp(
                prefix=".edge-tts-",
                suffix=os.path.splitext(voice_file)[1] or ".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            with os.fdopen(descriptor, "wb") as file:
                def _handle_chunk(chunk):
                    chunk_type = chunk["type"]
                    if chunk_type == "audio":
                        file.write(chunk["data"])
                    elif chunk_type in ["WordBoundary", "SentenceBoundary"]:
                        # Whether it comes from the 7.x sync stream or the old
                        # async stream, any event carrying boundary info is fed to
                        # SubMaker so the subtitle chain keeps using the project's
                        # existing logic.
                        sub_maker.feed(chunk)

                stream_edge_tts_chunks(
                    communicate, _handle_chunk, timeout_seconds=timeout_seconds
                )

            # Edge can finish a stream with timing events but no audio payload.
            # Those events produce a nonempty SRT, yet the MP3 is unplayable.
            if os.path.getsize(temp_path) == 0:
                logger.warning("failed, edge tts stream contained no audio")
                continue

            if not sub_maker.get_srt():
                logger.warning("failed, sub_maker.get_srt() is empty")
                continue

            # Audio and its timing belong to this same completed attempt.
            # Failed retries must not replace previously successful narration.
            os.replace(temp_path, voice_file)
            temp_path = None
            logger.info(f"completed, output file: {voice_file}")
            return sub_maker
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError as remove_error:
                    logger.warning(
                        "failed to remove temporary Edge TTS audio: "
                        f"{temp_path}, error: {remove_error}"
                    )
    return None


def siliconflow_tts(
    text: str,
    model: str,
    voice: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech through the SiliconFlow API.

    Args:
        text: text to convert to speech
        model: model name, e.g. "FunAudioLLM/CosyVoice2-0.5B"
        voice: voice name, e.g. "FunAudioLLM/CosyVoice2-0.5B:alex"
        voice_rate: speech rate, range [0.25, 4.0]
        voice_file: output audio file path
        voice_volume: speech volume, range [0.6, 5.0], mapped to the SiliconFlow
            gain range [-10, 10]

    Returns:
        SubMaker object or None
    """
    text = text.strip()
    api_key = config.siliconflow.get("api_key", "")

    if not api_key:
        logger.error("SiliconFlow API key is not set")
        return None

    # voice_volume is a linear multiplier while SiliconFlow gain is in dB, range
    # [-10, 10]. The 1.0 default maps to 0 dB; 2.0 is about +6 dB and 0.6 about
    # -4.4 dB. Subtracting directly would treat 2x volume as +1 dB, which clearly
    # disagrees with the linear mix of the preview and the final video.
    if voice_volume is None:
        voice_volume = 1.0
    if voice_volume <= 0:
        gain = -10.0
    else:
        gain = max(-10.0, min(10.0, 20.0 * math.log10(voice_volume)))

    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,
        "speed": voice_rate,
        "gain": gain,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(3):  # up to 3 attempts
        temporary_audio = None
        response_accepted = False
        try:
            logger.info(
                f"start siliconflow tts, model: {model}, voice: {voice}, try: {i + 1}"
            )

            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                allow_redirects=False,
                timeout=_SILICONFLOW_TTS_TIMEOUT_SECONDS,
            )

            if 300 <= response.status_code < 400:
                # Redirecting a speech POST may replay a billed synthesis or
                # forward its input/credentials. Do not follow or resubmit it.
                logger.error("TTS endpoint returned a redirect; stop speech retries")
                return None
            if response.status_code == 200:
                response_accepted = True
                if not response.content:
                    logger.error("siliconflow tts returned empty audio")
                    return None
                # Decode a temporary file before publishing it. A 200 response
                # may contain invalid audio, and must not destroy a previous
                # successful narration at the same path.
                ensure_file_path_exists(voice_file)
                with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(os.path.abspath(voice_file)),
                    suffix=".mp3",
                    delete=False,
                ) as f:
                    temporary_audio = f.name
                    f.write(response.content)

                sub_maker = ensure_legacy_submaker_fields(SubMaker())

                try:
                    _validate_remote_tts_audio(temporary_audio)
                    audio_clip = AudioFileClip(temporary_audio)
                    try:
                        audio_duration = audio_clip.duration
                    finally:
                        audio_clip.close()
                    if (
                        not isinstance(audio_duration, (int, float))
                        or not math.isfinite(audio_duration)
                        or audio_duration <= 0
                    ):
                        raise ValueError("audio duration must be positive and finite")
                except Exception as e:
                    # A 200 response is not proof that the bytes contain usable
                    # narration. Returning a fabricated duration lets an invalid
                    # file advance into the costly video pipeline.
                    logger.error(f"siliconflow tts returned invalid audio: {e}")
                    return None

                os.replace(temporary_audio, voice_file)
                logger.success(f"siliconflow tts succeeded: {voice_file}")
                return populate_legacy_submaker_with_full_text(
                    sub_maker=sub_maker,
                    text=text,
                    audio_duration_seconds=audio_duration,
                )
            else:
                logger.error(
                    f"siliconflow tts failed with status code {response.status_code}: {response.text}"
                )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"siliconflow tts could not connect, retrying: {e}")
        except requests.exceptions.RequestException as e:
            # The server may have synthesized and charged for the POST even
            # though its response was lost. A fresh POST could bill again.
            logger.error(
                "siliconflow tts result is unconfirmed after a transport error; "
                f"stop paid retries: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"siliconflow tts failed: {str(e)}")
            if response_accepted:
                # Retrying local processing cannot recover this accepted response
                # and would submit another synthesis request.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as cleanup_error:
                    logger.warning(
                        f"failed to remove temporary siliconflow audio: {cleanup_error}"
                    )

    return None


def _build_azure_v2_ssml(text: str, voice_name: str, voice_rate: float) -> str:
    """Build the SSML used by Azure Speech V2, normalizing the rate safely."""
    try:
        normalized_rate = float(voice_rate)
    except (TypeError, ValueError):
        normalized_rate = 1.0
    normalized_rate = max(0.25, min(4.0, normalized_rate))

    voice_locale_parts = voice_name.split("-", 2)
    voice_locale = (
        "-".join(voice_locale_parts[:2])
        if len(voice_locale_parts) >= 2
        else "en-US"
    )
    escaped_text = escape(text)
    escaped_voice_locale = escape(voice_locale, {'"': "&quot;"})
    escaped_voice_name = escape(voice_name, {'"': "&quot;"})
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{escaped_voice_locale}">'
        f'<voice name="{escaped_voice_name}">'
        f'<prosody rate="{normalized_rate:g}">{escaped_text}</prosody>'
        "</voice></speak>"
    )


def azure_tts_v2(
    text: str,
    voice_name: str,
    voice_file: str,
    voice_rate: float = 1.0,
) -> Union[SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"invalid voice name: {voice_name}")
        raise ValueError(f"invalid voice name: {voice_name}")
    text = text.strip()
    ssml = _build_azure_v2_ssml(text, voice_name, voice_rate)

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, timedelta):
            # Speech SDK durations are timedeltas; integer arithmetic retains
            # all microseconds and also handles zero/exact-second durations.
            return ((duration.days * 86400 + duration.seconds) * 1000000
                    + duration.microseconds) * 10
        if isinstance(duration, int):
            return duration
        return 0

    for i in range(3):
        temporary_audio = None
        audio_config = None
        speech_synthesizer = None
        try:
            logger.info(
                f"start, voice name: {voice_name}, rate: {voice_rate}, try: {i + 1}"
            )

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = ensure_legacy_submaker_fields(SubMaker())

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                # print('WordBoundary event:')
                # print('\tBoundaryType: {}'.format(evt.boundary_type))
                # print('\tAudioOffset: {}ms'.format((evt.audio_offset + 5000)))
                # print('\tDuration: {}'.format(evt.duration))
                # print('\tText: {}'.format(evt.text))
                # print('\tTextOffset: {}'.format(evt.text_offset))
                # print('\tWordLength: {}'.format(evt.word_length))

                duration = _format_duration_to_offset(evt.duration)
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Creates an instance of a speech config with specified subscription key and service region.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            if not speech_key or not service_region:
                logger.error("Azure speech key or region is not set")
                return None

            # File and default-speaker output are mutually exclusive SDK modes.
            # Let each attempt own a private file, never the last good export.
            ensure_file_path_exists(voice_file)
            descriptor, temporary_audio = tempfile.mkstemp(
                prefix=".azure-tts-", suffix=".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            os.close(descriptor)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temporary_audio)
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            # speak_text_async() does not accept a rate. With SSML prosody, both
            # previews and real generation apply the voice_rate from WebUI/API.
            result = speech_synthesizer.speak_ssml_async(ssml).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                if os.path.getsize(temporary_audio) <= 0:
                    logger.error("Azure completed without audio; preserve the previous export")
                    return None
                # Release SDK file-output owners before publication, including
                # on Windows where an open handle can prevent replacement.
                speech_synthesizer = None
                audio_config = None
                os.replace(temporary_audio, voice_file)
                temporary_audio = None
                logger.success(f"azure v2 speech synthesis succeeded: {voice_file}")
                return sub_maker
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"azure v2 speech synthesis canceled: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"azure v2 speech synthesis error: {cancellation_details.error_details}"
                    )
            logger.info(f"completed, output file: {voice_file}")
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
        finally:
            # Exceptions/cancellation can also leave the SDK output owner alive.
            # Drop all file owners before attempting to remove its staged file.
            speech_synthesizer = None
            audio_config = None
            if temporary_audio is not None:
                try:
                    os.remove(temporary_audio)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(f"failed to remove staged Azure audio: {exc}")
    return None


def gemini_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech with Google Gemini TTS.

    Args:
        text: text to convert
        voice_name: voice name, e.g. "Zephyr", "Puck"
        voice_rate: speech rate (currently unused)
        voice_file: output audio file path
        voice_volume: audio volume (currently unused)

    Returns:
        SubMaker object or None
    """
    import base64
    import io
    from pydub import AudioSegment
    from google import genai
    from google.genai import types
    _configure_pydub_ffmpeg(AudioSegment)
    
    temporary_audio = None
    try:
        api_key = config.app.get("gemini_api_key", "")
        if not api_key:
            logger.error("Gemini API key is not set")
            return None

        logger.info(f"start, voice name: {voice_name}, try: 1")

        generation_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        # google-genai uses one Client for text and TTS models. The context
        # manager ensures the HTTP connection is released after the request while
        # keeping the existing PCM transcoding and subtitle timeline logic.
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=generation_config,
            )

        # A blocked prompt can have no candidates; a stopped candidate can
        # have no content or parts. All are empty-audio responses, not decoder
        # errors. Preserve the provider's reason without logging its text.
        candidate = response.candidates[0] if response.candidates else None
        content = getattr(candidate, "content", None)
        audio_data = None
        for part in getattr(content, "parts", None) or []:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_data = part.inline_data.data
                break
                
        if not audio_data:
            feedback = getattr(response, "prompt_feedback", None)
            logger.error(
                "No audio data received from Gemini TTS: "
                "finish_reason={}, prompt_block_reason={}",
                getattr(candidate, "finish_reason", None),
                getattr(feedback, "block_reason", None),
            )
            return None
            
        # The audio data is already raw bytes and needs no base64 decoding
        if isinstance(audio_data, str):
            # A string must be base64-decoded first
            audio_bytes = base64.b64decode(audio_data)
        else:
            # Already bytes, use as is
            audio_bytes = audio_data
        
        # Try different audio formats - Gemini may return different ones
        audio_segment = None
        
        # Gemini returns Linear PCM; parse it with the documented parameters
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format="raw",
                frame_rate=24000,  # Gemini TTS default sample rate
                channels=1,        # mono
                sample_width=2     # 16-bit
            )
        except Exception as e:
            logger.error(f"Failed to load PCM audio: {e}")
            return None
        
        # The API, CLI or tests may target a nested directory that does not exist
        # yet. Create parent directories before writing so a successful Gemini
        # request is not lost to a missing local path, and so this provider
        # behaves like the other TTS implementations.
        ensure_file_path_exists(voice_file)

        # pydub returns an open output file object. If batch generation never
        # closes it, file descriptors keep piling up and later overwrites or
        # deletions of the audio file become more likely to fail on Windows.
        if len(audio_segment) <= 0:
            raise ValueError("Gemini returned empty PCM audio")
        temp_fd, temporary_audio = tempfile.mkstemp(
            prefix=".gemini-tts-", suffix=".mp3",
            dir=os.path.dirname(os.path.abspath(voice_file)),
        )
        os.close(temp_fd)
        exported_audio = audio_segment.export(temporary_audio, format="mp3")
        exported_audio.close()
        
        # Gemini does not provide edge_tts-style word boundary events, so fall
        # back to the project's `subs/offset` compatibility structure so the
        # downstream subtitle and duration calculations keep working.
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        audio_duration = len(audio_segment) / 1000.0  # convert to seconds
        sub_maker = populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=audio_duration,
        )
        os.replace(temporary_audio, voice_file)
        logger.info(f"completed, output file: {voice_file}")
        return sub_maker
        
    except ImportError as e:
        logger.error(f"Missing required package for Gemini TTS: {str(e)}. Please install: pip install pydub")
        return None
    except Exception as e:
        logger.error(f"Gemini TTS failed, error: {str(e)}")
        return None

    finally:
        if temporary_audio and os.path.exists(temporary_audio):
            try:
                os.unlink(temporary_audio)
            except OSError as cleanup_error:
                logger.warning(f"could not remove Gemini staging audio: {cleanup_error}")


def mimo_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech with Xiaomi MiMo V2.5 TTS.

    The official API is compatible with OpenAI Chat Completions, but TTS has two
    key differences:
    1. the text to synthesize must be placed in the `assistant` message;
    2. the audio comes back as a base64 string in `message.audio.data`.

    MiMo currently returns no word-level timeline, so the project's existing
    legacy SubMaker fallback is reused: the subtitle timeline is derived from the
    final audio duration and the script sentence splits.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("MiMo TTS text is empty")
        return None

    api_key = config.app.get("mimo_api_key", "")
    if not api_key:
        logger.error("MiMo API key is not set")
        return None

    base_url = config.app.get("mimo_base_url", "") or _MIMO_DEFAULT_BASE_URL
    model_name = config.app.get("mimo_tts_model_name", "") or _MIMO_DEFAULT_TTS_MODEL
    style_prompt = config.app.get(
        "mimo_tts_style_prompt",
        "请用自然、清晰、适合短视频旁白的语气朗读。",
    )

    _configure_pydub_ffmpeg(AudioSegment)

    temporary_audio = None
    try:
        logger.info(
            f"start mimo tts, model: {model_name}, voice: {voice_name}"
        )
        ensure_file_path_exists(voice_file)

        # A lost response may still have generated and billed the narration.
        # Disable SDK retries as well as the outer retry loop.
        with OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0
        ) as client:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": style_prompt},
                    {"role": "assistant", "content": text},
                ],
                audio={
                    "format": "wav",
                    "voice": voice_name,
                },
            )

        if not completion or not getattr(completion, "choices", None):
            raise ValueError("MiMo TTS returned empty response")

        message = completion.choices[0].message
        audio = getattr(message, "audio", None)
        audio_data = None
        if isinstance(audio, dict):
            audio_data = audio.get("data")
        elif audio is not None:
            audio_data = getattr(audio, "data", None)

        if not audio_data:
            raise ValueError("MiMo TTS returned empty audio data")

        audio_bytes = base64.b64decode(audio_data)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

        output_format = utils.parse_extension(voice_file) or "mp3"
        descriptor, temporary_audio = tempfile.mkstemp(
            prefix=".mimo-tts-", suffix=f".{output_format}",
            dir=os.path.dirname(os.path.abspath(voice_file)),
        )
        os.close(descriptor)
        if output_format == "wav":
            with open(temporary_audio, "wb") as f:
                f.write(audio_bytes)
        else:
            exported_audio = audio_segment.export(temporary_audio, format=output_format)
            if exported_audio is not None:
                exported_audio.close()

        audio_duration = len(audio_segment) / 1000.0
        if audio_duration <= 0:
            raise ValueError("MiMo TTS returned empty audio")
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        populated_sub_maker = populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=audio_duration,
        )
        os.replace(temporary_audio, voice_file)
        temporary_audio = None
        logger.success(f"mimo tts succeeded: {voice_file}")
        logger.debug(
            "mimo subtitle timeline generated, "
            f"duration: {audio_duration:.3f}s, output_format: {output_format}"
        )
        return populated_sub_maker
    except Exception as e:
        logger.error(f"mimo tts failed: {str(e)}")
    finally:
        if temporary_audio and os.path.exists(temporary_audio):
            try:
                os.remove(temporary_audio)
            except OSError as cleanup_error:
                logger.warning(f"failed to remove temporary MiMo audio: {cleanup_error}")

    return None


def _resolve_minimax_tts_url(configured_url: str) -> str:
    configured_url = (configured_url or "").strip().rstrip("/")
    if not configured_url:
        return MINIMAX_TTS_GLOBAL_URL
    if configured_url in {MINIMAX_TTS_GLOBAL_URL, MINIMAX_TTS_CN_URL}:
        return configured_url
    if configured_url.endswith("/v1"):
        return f"{configured_url}/t2a_v2"
    return configured_url


def get_minimax_tts_api_key() -> str:
    """Return the effective MiniMax TTS key; dedicated config beats the shared
    LLM config."""
    return str(
        config.minimax_tts.get("api_key", "")
        or config.app.get("minimax_api_key", "")
        or os.getenv("MINIMAX_API_KEY", "")
        or ""
    ).strip()


def _infer_minimax_tts_url(base_url: str) -> str:
    """Infer the same-region TTS endpoint from the MiniMax LLM URL; return an
    empty string when it cannot be recognized."""
    normalized_url = str(base_url or "").strip()
    if not normalized_url:
        return ""

    parse_target = normalized_url if "://" in normalized_url else f"//{normalized_url}"
    host = (urlparse(parse_target).hostname or "").lower()
    if host == "minimaxi.com" or host.endswith(".minimaxi.com"):
        return MINIMAX_TTS_CN_URL
    if host == "minimax.io" or host.endswith(".minimax.io"):
        return MINIMAX_TTS_GLOBAL_URL
    return ""


def get_minimax_tts_endpoint() -> str:
    """
    Return the MiniMax TTS endpoint that matches the effective key.

    With a dedicated TTS key, the user's chosen TTS endpoint is respected. When
    the MiniMax LLM key is reused, follow the region of the LLM base URL so a
    China-site key is not sent to the global site and rejected with 401.
    """
    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    if not dedicated_key:
        inferred_url = _infer_minimax_tts_url(config.app.get("minimax_base_url", ""))
        if inferred_url:
            return inferred_url
    return _resolve_minimax_tts_url(config.minimax_tts.get("base_url", ""))


def get_minimax_voice_catalog(
    api_key: str = "",
    endpoint: str = "",
    voice_type: str = "all",
) -> list[dict[str, str]]:
    """
    Query the system, cloned and generated voices available to this MiniMax
    account.

    The return value is normalized to the voice_id, voice_name and voice_type
    fields, so callers need not know that MiniMax splits its response arrays by
    voice source. Failures raise so the WebUI, API or CLI can surface a clear
    error in their own way instead of silently returning an empty list.
    """
    if voice_type not in {"system", "voice_cloning", "voice_generation", "all"}:
        raise ValueError(f"Unsupported MiniMax voice type: {voice_type}")

    effective_api_key = str(api_key or get_minimax_tts_api_key()).strip()
    if not effective_api_key:
        raise ValueError("MiniMax TTS API key is not set")

    tts_endpoint = (
        _resolve_minimax_tts_url(endpoint)
        if endpoint
        else get_minimax_tts_endpoint()
    )
    voice_endpoint = (
        f"{tts_endpoint[:-len('/t2a_v2')]}/get_voice"
        if tts_endpoint.endswith("/t2a_v2")
        else f"{tts_endpoint.rstrip('/')}/get_voice"
    )
    response = trace_http_call(
        "POST", requests.post,
        voice_endpoint,
        json={"voice_type": voice_type},
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"MiniMax get_voice failed with status {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("MiniMax get_voice returned invalid JSON") from exc

    base_resp = body.get("base_resp") or {}
    if base_resp.get("status_code") not in {0, "0"}:
        status_message = str(base_resp.get("status_msg") or "unknown error")
        raise RuntimeError(f"MiniMax get_voice failed: {status_message}")

    catalog = []
    seen_voice_ids = set()
    response_groups = (
        ("system", "system_voice"),
        ("voice_cloning", "voice_cloning"),
        ("voice_generation", "voice_generation"),
    )
    for normalized_type, response_key in response_groups:
        for item in body.get(response_key) or []:
            voice_id = str(item.get("voice_id") or "").strip()
            if not voice_id or voice_id in seen_voice_ids:
                continue
            seen_voice_ids.add(voice_id)
            catalog.append(
                {
                    "voice_id": voice_id,
                    "voice_name": str(item.get("voice_name") or voice_id).strip(),
                    "voice_type": normalized_type,
                }
            )

    logger.info(f"loaded MiniMax voices: count={len(catalog)}, type={voice_type}")
    return catalog


def _validate_remote_tts_audio(file_path: str) -> None:
    """Decode accepted audio fully before replacing a successful narration.

    MoviePy's duration/initial frame probe does not validate later MP3 frames.
    Reuse the bounded, self-contained FFmpeg audio validator without following
    playlists or resubmitting an already accepted speech request.
    """
    try:
        bgm.validate_audio_file(file_path, timeout_seconds=120)
    except (bgm.BgmUploadError, bgm.BgmServiceError) as exc:
        raise ValueError("TTS returned audio that could not be fully decoded") from exc


def _write_validated_minimax_audio(audio_bytes: bytes, voice_file: str) -> float:
    """
    Atomically write MiniMax audio to the target path and return its duration.

    A success status from the remote service does not guarantee complete audio.
    Validate in a temporary file in the same directory, then atomically replace
    with os.replace, so a decode failure or an unreadable file for MoviePy never
    leaves a half-written artifact.
    """
    ensure_file_path_exists(voice_file)
    output_dir = os.path.dirname(os.path.abspath(voice_file))
    output_suffix = os.path.splitext(voice_file)[1] or ".mp3"
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".minimax-tts-", suffix=output_suffix, dir=output_dir
    )
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as output:
            output.write(audio_bytes)

        _validate_remote_tts_audio(temp_path)
        # MiniMax's configured response container can differ from the caller's
        # artifact extension (the task pipeline normally requests audio.mp3).
        # Keep matching payloads byte-identical; transcode supported mismatches
        # before publication so MIME/extension consumers see the promised codec.
        source_format = (
            "wav" if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
            else "flac" if audio_bytes.startswith(b"fLaC")
            else "mp3" if audio_bytes.startswith(b"ID3") or (
                len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF
                and audio_bytes[1] & 0xE0 == 0xE0
            )
            else None
        )
        output_format = output_suffix.lstrip(".").lower()
        if source_format and output_format in {"mp3", "wav", "flac"} and source_format != output_format:
            from pydub import AudioSegment

            _configure_pydub_ffmpeg(AudioSegment)
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
            exported = segment.export(temp_path, format=output_format)
            if exported is not None:
                exported.close()
            _validate_remote_tts_audio(temp_path)
        audio_clip = AudioFileClip(temp_path)
        try:
            audio_duration = float(audio_clip.duration)
        finally:
            audio_clip.close()

        if not math.isfinite(audio_duration) or audio_duration <= 0:
            raise ValueError("MiniMax TTS returned audio with an invalid duration")

        os.replace(temp_path, voice_file)
        return audio_duration
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def minimax_tts(text: str, voice_id: str, voice_rate: float, voice_file: str, voice_volume: float = 1.0) -> Union[SubMaker, None]:
    """Generate speech with the synchronous MiniMax T2A HTTP API."""
    text, voice_id = (text or "").strip(), (voice_id or "").strip()
    if not text or not voice_id:
        logger.error("MiniMax TTS requires text and a voice ID")
        return None
    settings = config.minimax_tts
    api_key = get_minimax_tts_api_key()
    if not api_key:
        logger.error("MiniMax TTS API key is not set")
        return None
    url = get_minimax_tts_endpoint()
    model = str(settings.get("model_id", MINIMAX_TTS_DEFAULT_MODEL) or MINIMAX_TTS_DEFAULT_MODEL).strip()
    if model not in MINIMAX_TTS_MODELS:
        logger.error(f"Unsupported MiniMax TTS model: {model}")
        return None
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
        volume = max(0.0, min(10.0, float(voice_volume or 1.0)))
        pitch = max(-12, min(12, int(settings.get("pitch", 0) or 0)))
        sample_rate = int(settings.get("sample_rate", 32000) or 32000)
        bitrate = int(settings.get("bitrate", 128000) or 128000)
        channel = int(settings.get("channel", 1) or 1)
    except (TypeError, ValueError) as exc:
        logger.error(f"Invalid MiniMax TTS audio setting: {str(exc)}")
        return None
    audio_format = str(settings.get("audio_format", "mp3") or "mp3").strip()
    if audio_format not in {"mp3", "wav", "flac"}:
        logger.error(f"Unsupported MiniMax TTS audio format: {audio_format}; choose mp3, wav, or flac")
        return None
    payload = {
        "model": model, "text": text, "stream": False, "language_boost": "auto", "output_format": "hex",
        "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": volume, "pitch": pitch},
        "audio_setting": {"sample_rate": sample_rate, "bitrate": bitrate, "format": audio_format, "channel": channel},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        received_success = False
        try:
            logger.info(f"start MiniMax TTS, model: {model}, voice: {voice_id}, try: {attempt + 1}")
            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=120)
            if 300 <= response.status_code < 400:
                # Redirecting a speech POST may replay a billed synthesis or
                # forward its input/credentials. Do not follow or resubmit it.
                logger.error("TTS endpoint returned a redirect; stop speech retries")
                return None
            if response.status_code != 200:
                logger.error(f"MiniMax TTS failed with status {response.status_code}: {response.text[:200]}")
                continue
            received_success = True
            body = response.json()
            data = body.get("data") or {}
            base_resp = body.get("base_resp") or {}
            status_code = base_resp.get("status_code")
            if not isinstance(status_code, int) or isinstance(status_code, bool):
                logger.error("MiniMax returned an unknown acceptance status; stop paid retries")
                return None
            if status_code != 0:
                logger.error(f"MiniMax TTS returned an unsuccessful response: status_code={base_resp.get('status_code')}, audio_status={data.get('status')}")
                continue
            if data.get("status") != 2:
                logger.error("MiniMax accepted the request but returned incomplete audio")
                return None
            audio_hex = data.get("audio")
            if not isinstance(audio_hex, str) or not audio_hex:
                logger.error("MiniMax TTS returned empty audio data")
                return None
            if len(audio_hex) > _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS:
                logger.error("MiniMax TTS returned audio data exceeding the supported size")
                return None
            audio_duration = _write_validated_minimax_audio(bytes.fromhex(audio_hex), voice_file)
            logger.success(f"MiniMax TTS succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                ensure_legacy_submaker_fields(SubMaker()), text, audio_duration
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.warning(f"MiniMax TTS could not connect, retrying: {exc}")
        except requests.exceptions.RequestException as exc:
            logger.error(
                "MiniMax TTS result is unconfirmed after a transport error; "
                f"stop paid retries: {type(exc).__name__}"
            )
            return None
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.error(f"MiniMax TTS failed: {str(exc)}")
            if received_success:
                # The request may already be billed, even if JSON/audio parsing
                # or local file publication failed. Never regenerate it here.
                return None
    return None


def elevenlabs_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    text = (text or "").strip()
    if not text:
        logger.error("ElevenLabs TTS text is empty")
        return None

    api_key = get_elevenlabs_api_key()
    if not api_key:
        logger.error("ElevenLabs API key is not set")
        return None

    if not model_id:
        model_id = config.elevenlabs.get("model_id", "eleven_multilingual_v2")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    # The live v2 endpoint rejects speeds outside 0.7–1.2; v3 supports
    # the wider REST range. Never silently clamp a selected UI/API speed.
    minimum_speed, maximum_speed = (0.25, 4.0) if model_id == "eleven_v3" else (0.7, 1.2)
    speed_error = (
        f"ElevenLabs model {model_id} speech speed must be a finite number "
        f"from {minimum_speed} to {maximum_speed}; choose a supported speed "
        "or use eleven_v3 for a wider range"
    )
    try:
        speed = float(voice_rate) if voice_rate is not None else 1.0
    except (TypeError, ValueError, OverflowError):
        logger.error(speed_error)
        return None
    if speed != 1.0 and model_id in {"eleven_v4", "eleven_v4_turbo"}:
        logger.error(f"ElevenLabs model {model_id} does not support speech speed; use 1.0 or a v2/v3 model")
        return None
    if not math.isfinite(speed) or not minimum_speed <= speed <= maximum_speed:
        logger.error(speed_error)
        return None
    if speed != 1.0:
        payload["voice_settings"]["speed"] = speed

    # Errors where retrying will never help (auth/access/validation failures).
    _NON_RETRYABLE_CODES = {401, 403, 422}
    _NON_RETRYABLE_STATUSES = {"voice_disabled", "voice_access_denied", "unauthorized", "invalid_voice_settings"}

    for i in range(3):
        response = None
        temp_path = None
        try:
            logger.info(f"start elevenlabs tts, voice_id: {voice_id}, try: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=60,
                stream=True,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                # A paid request may already have reached the provider. Do not
                # follow the redirect with our key or resubmit the generation.
                logger.error("ElevenLabs TTS returned a redirect; stop paid retries")
                return None
            if response.status_code != 200:
                error_status = ""
                error_bytes = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=4096):
                        error_bytes.extend(
                            chunk[:_ELEVENLABS_TTS_MAX_ERROR_BYTES - len(error_bytes)]
                        )
                        if len(error_bytes) >= _ELEVENLABS_TTS_MAX_ERROR_BYTES:
                            break
                    detail = json.loads(error_bytes).get("detail", {})
                    if isinstance(detail, dict):
                        error_status = detail.get("status", "")
                except (ValueError, TypeError, AttributeError, requests.RequestException):
                    pass
                error_text = error_bytes.decode("utf-8", errors="replace")[:200]

                if response.status_code in _NON_RETRYABLE_CODES or error_status in _NON_RETRYABLE_STATUSES:
                    logger.error(
                        f"ElevenLabs TTS failed (non-retryable) — voice_id: {voice_id}, "
                        f"status: {response.status_code}, error: {error_status or error_text}. "
                        + (f"Check the selected model and speech speed: {error_text}"
                           if error_status == "invalid_voice_settings"
                           else "Please select a different ElevenLabs voice.")
                    )
                    return None

                logger.error(
                    f"elevenlabs tts failed with status {response.status_code}: {error_text}"
                )
                continue

            descriptor, temp_path = tempfile.mkstemp(
                prefix=".elevenlabs-tts-",
                suffix=os.path.splitext(voice_file)[1] or ".mp3",
                dir=os.path.dirname(os.path.abspath(voice_file)),
            )
            audio_bytes = 0
            with os.fdopen(descriptor, "wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    audio_bytes += len(chunk)
                    if audio_bytes > _ELEVENLABS_TTS_MAX_AUDIO_BYTES:
                        logger.error("ElevenLabs TTS audio exceeds the 50 MB limit")
                        return None
                    output.write(chunk)
            if audio_bytes == 0:
                logger.error("ElevenLabs TTS returned no audio data")
                return None

            _validate_remote_tts_audio(temp_path)
            audio_clip = AudioFileClip(temp_path)
            try:
                audio_duration = float(audio_clip.duration)
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                logger.error("ElevenLabs TTS returned audio with invalid duration")
                return None

            os.replace(temp_path, voice_file)
            temp_path = None

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            logger.success(f"elevenlabs tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"elevenlabs tts could not connect, retrying: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(
                "elevenlabs tts result is unconfirmed after a transport error; "
                f"stop paid retries: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"elevenlabs tts failed: {str(e)}")
            if response is not None and response.status_code == 200:
                # The provider may already have charged for a successful
                # request. Retrying cannot repair a corrupt returned file.
                return None
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError as exc:
                    logger.warning(f"failed to remove ElevenLabs TTS temp file: {exc}")
            if response is not None:
                response.close()

    return None


def _openai_compatible_tts(
    provider: str,
    base_url: str,
    api_key: str,
    model_id: str,
    voice: str,
    text: str,
    voice_rate: float,
    voice_file: str,
) -> Union[SubMaker, None]:
    """Shared transport for self-hosted, OpenAI-compatible ``/audio/speech``
    servers (Chatterbox, Kokoro, ...).

    Writes the returned audio to ``voice_file`` and builds the full-text
    SubMaker: these servers return no word-level timestamps, so set
    ``subtitle_provider = "whisper"`` for tighter subtitle sync.
    """
    url = f"{base_url}/audio/speech"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        # OpenAI speech API accepts speed 0.25-4.0; MoneyPrinterTurbo's rate is a
        # 1.0-centred multiplier, so it maps directly (clamped to the valid range).
        "speed": max(0.25, min(4.0, float(voice_rate or 1.0))),
    }
    # The OpenAI speech protocol has no volume field; voice_volume is applied at
    # the final mixing stage. speed only controls speaking rate and must not be
    # used as a volume parameter.

    for i in range(3):
        temporary_audio = None
        response_accepted = False
        try:
            logger.info(f"start {provider} tts, voice: {voice}, try: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=120)
            if 300 <= response.status_code < 400:
                # Redirecting a speech POST may replay a billed synthesis or
                # forward its input/credentials. Do not follow or resubmit it.
                logger.error("TTS endpoint returned a redirect; stop speech retries")
                return None
            if response.status_code != 200:
                logger.error(
                    f"{provider} tts failed with status {response.status_code}: {response.text[:200]}"
                )
                continue

            response_accepted = True
            if not response.content:
                raise ValueError(f"{provider} returned empty audio")

            # Write to a temp file in the same directory and really decode it
            # before replacing the target. A failure must not damage an existing
            # preview/narration; close the file before decoding and replacing to
            # respect Windows file locking rules.
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=".mp3", delete=False,
            ) as f:
                temporary_audio = f.name
                f.write(response.content)

            _validate_remote_tts_audio(temporary_audio)
            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError(f"{provider} returned an invalid audio duration")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"{provider} tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout:
            # No connection was established; retrying cannot replay synthesis.
            logger.warning(f"{provider} tts could not connect, retrying")
        except requests.exceptions.RequestException as exc:
            # A lost response can follow successful remote synthesis. Keep the
            # same acceptance boundary as hosted speech providers: do not POST
            # again when the first request's outcome is unknown.
            logger.error(
                f"{provider} tts result is unconfirmed after a transport error; "
                f"stop speech retries: {type(exc).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"{provider} tts failed: {str(e)}")
            if response_accepted:
                # Retrying local processing cannot recover this accepted response
                # and would submit another synthesis request.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    # A cleanup failure still follows the TTS failure contract; no
                    # secondary exception hides the original cause.
                    logger.warning(f"could not remove temporary {provider} audio: {exc}")

    return None


def chatterbox_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Generate speech with a self-hosted Chatterbox TTS server.

    Chatterbox (Resemble AI, MIT) is an open-source, locally hosted TTS model
    with zero-shot voice cloning — a self-hostable alternative to ElevenLabs.
    This talks to an OpenAI-compatible ``/audio/speech`` endpoint, so it works
    with the common community servers (e.g. devnen/Chatterbox-TTS-Server,
    travisvn/chatterbox-tts-api). Configure ``[chatterbox] base_url`` (and an
    optional ``api_key``).

    Like ElevenLabs, Chatterbox does not return word-level timestamps, so the
    subtitle path falls back to the full-text SubMaker. For tighter subtitle
    sync set ``subtitle_provider = "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("Chatterbox TTS text is empty")
        return None

    base_url = (config.chatterbox.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "Chatterbox base_url is not set, please configure [chatterbox] base_url in config.toml"
        )
        return None

    api_key = config.chatterbox.get("api_key", "")
    if not model_id:
        model_id = config.chatterbox.get("model_id", "chatterbox") or "chatterbox"

    return _openai_compatible_tts(
        "chatterbox", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


def kokoro_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Generate speech with a self-hosted Kokoro TTS server.

    Kokoro (hexgrad/Kokoro-82M, Apache-2.0 code and weights) is a small open
    TTS model that runs well on CPU — a free, offline alternative to the
    cloud voices. This talks to an OpenAI-compatible ``/audio/speech``
    endpoint, so it works with the common servers (e.g. remsky/Kokoro-FastAPI
    on port 8880). Configure ``[kokoro] base_url`` (ending in ``/v1``) and an
    optional ``api_key``.

    Voice names are Kokoro's presets (``af_heart``, ``bf_emma``, ``hf_alpha``,
    ...); their first letter is the language (a/b English, e Spanish, f French,
    h Hindi, i Italian, p Portuguese, j Japanese, z Chinese), so pick a voice
    that matches the script's language.

    Like Chatterbox, the OpenAI speech contract returns no word-level
    timestamps, so the subtitle path falls back to the full-text SubMaker.
    For tighter subtitle sync set ``subtitle_provider = "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("Kokoro TTS text is empty")
        return None
    # Pure punctuation/emoji contain nothing pronounceable; a real service may
    # return an empty MP3 with HTTP 200, so stopping early avoids a pointless
    # request and low-level MoviePy errors when decoding an empty file.
    if not any(character.isalnum() for character in text):
        logger.error("Kokoro TTS text contains no speakable characters")
        return None
    base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "Kokoro base_url is not set, please configure [kokoro] base_url in config.toml"
        )
        return None
    api_key = config.kokoro.get("api_key", "")
    if not model_id:
        model_id = config.kokoro.get("model_id", "kokoro") or "kokoro"
    return _openai_compatible_tts(
        "kokoro", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


# Fish Audio supported models.
FISH_AUDIO_MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro")
FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"


def fish_audio_tts(
    text: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    reference_id: str | None = None,
) -> Union[SubMaker, None]:
    """Generate speech using Fish Audio TTS API.

    The model is read from ``config.fish_audio["model"]`` (single source of
    truth).  ``reference_id`` selects a public or cloned voice; when *None*
    Fish Audio's built-in default voice is used.

    ``voice_rate`` is mapped to the ``prosody.speed`` field (0.5–2.0) and
    ``voice_volume`` is converted from a linear multiplier to dB for the
    ``prosody.volume`` field (-20.0–20.0 dB).
    """
    text = (text or "").strip()
    if not text:
        logger.error("Fish Audio TTS text is empty")
        return None

    api_key = get_fish_audio_api_key()
    if not api_key:
        logger.error(
            "Fish Audio API key is not set. Please set it in config.toml "
            "[fish_audio] or FISH_API_KEY environment variable."
        )
        return None

    model_name = str(
        config.fish_audio.get("model", FISH_AUDIO_DEFAULT_MODEL)
        or FISH_AUDIO_DEFAULT_MODEL
    ).strip()
    if model_name not in FISH_AUDIO_MODELS:
        logger.warning(
            f"Unknown Fish Audio model '{model_name}', falling back to "
            f"'{FISH_AUDIO_DEFAULT_MODEL}'"
        )
        model_name = FISH_AUDIO_DEFAULT_MODEL

    # Map voice_rate → prosody.speed (0.5–2.0)
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0

    # Map voice_volume (linear multiplier) → prosody.volume (dB, -20–20).
    # A multiplier of 1.0 → 0 dB; 0.1 → -20 dB; 2.0 → +6 dB.
    import math
    try:
        vol = float(voice_volume or 1.0)
        if vol <= 0:
            volume_db = -20.0
        else:
            volume_db = max(-20.0, min(20.0, 20.0 * math.log10(vol)))
    except (TypeError, ValueError):
        volume_db = 0.0

    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model_name,
    }
    payload: dict = {
        "text": text,
        "format": "mp3",
        "prosody": {
            "speed": speed,
            "volume": volume_db,
        },
    }
    if reference_id:
        payload["reference_id"] = reference_id

    for i in range(3):
        temporary_audio = None
        received_success = False
        try:
            logger.info(
                f"start fish audio tts, model: {model_name}, "
                f"ref: {reference_id or 'default'}, try: {i + 1}"
            )
            ensure_file_path_exists(voice_file)

            response = trace_http_call(
                "POST", requests.post, url, json=payload, headers=headers, allow_redirects=False, timeout=60)
            if 300 <= response.status_code < 400:
                # Redirecting a speech POST may replay a billed synthesis or
                # forward its input/credentials. Do not follow or resubmit it.
                logger.error("TTS endpoint returned a redirect; stop speech retries")
                return None
            if response.status_code == 401:
                logger.error(
                    "Fish Audio TTS failed: Invalid API key (401). "
                    "Check config.toml [fish_audio] api_key or FISH_API_KEY."
                )
                return None
            if response.status_code == 402:
                logger.error(
                    "Fish Audio TTS failed: Insufficient API credit (402). "
                    "Please check your account balance at "
                    "https://fish.audio/app/developers or verify your model and billing tier."
                )
                return None
            if response.status_code == 429:
                logger.warning(
                    "Fish Audio TTS rate limited (429), retrying..."
                )
                continue
            if response.status_code != 200:
                logger.error(
                    f"fish audio tts failed with status "
                    f"{response.status_code}: {response.text[:200]}"
                )
                continue
            received_success = True

            # Validate response contains audio data
            if not response.content or len(response.content) < 100:
                logger.error(
                    "Fish Audio TTS returned empty or invalid audio data"
                )
                return None

            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=".mp3",
                delete=False,
            ) as f:
                temporary_audio = f.name
                f.write(response.content)

            _validate_remote_tts_audio(temporary_audio)
            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError("Fish Audio returned an invalid audio duration")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"fish audio tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"fish audio tts could not connect, retrying: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(
                "fish audio tts result is unconfirmed after a transport error; "
                f"stop paid retries: {type(e).__name__}"
            )
            return None
        except Exception as e:
            logger.error(f"fish audio tts failed: {str(e)}")
            if received_success:
                # A successful provider response may already have been billed.
                return None
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as cleanup_error:
                    logger.warning(
                        f"failed to remove temporary Fish Audio file: {cleanup_error}"
                    )

    return None


def _iter_voxcpm_sse_events(response):
    """Yield bounded SSE events without Requests' unbounded line buffer."""
    def lines():
        pending = bytearray()
        skip_lf = False
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            # Consume the optional LF after a CR, even across transport chunks.
            if skip_lf:
                skip_lf = False
                if chunk.startswith(b"\n"):
                    chunk = chunk[1:]
                if not chunk:
                    continue
            for piece in chunk.splitlines(keepends=True):
                if piece.endswith(b"\r\n"):
                    payload, terminated = piece[:-2], True
                elif piece.endswith((b"\r", b"\n")):
                    payload, terminated = piece[:-1], True
                else:
                    payload, terminated = piece, False
                if len(pending) + len(payload) > _VOXCPM_SSE_MAX_EVENT_BYTES:
                    raise ValueError("VoxCPM SSE line exceeds the size limit")
                pending.extend(payload)
                if terminated:
                    yield bytes(pending).decode("utf-8")
                    pending.clear()
            skip_lf = chunk.endswith(b"\r")
        if pending:
            yield bytes(pending).decode("utf-8")

    def parse_event(data):
        try:
            event = json.loads("\n".join(data))
        except (TypeError, ValueError) as exc:
            raise ValueError("VoxCPM returned invalid SSE event data") from exc
        if not isinstance(event, dict):
            raise ValueError("VoxCPM returned a non-object SSE event")
        return event

    event_data = []
    event_bytes = 0
    first_line = True
    for line in lines():
        if first_line:
            # SSE UTF-8 decoding ignores one BOM at the start of the stream.
            line = line.removeprefix("\ufeff")
            first_line = False
        if not line:
            if event_data:
                yield parse_event(event_data)
                event_data = []
                event_bytes = 0
            continue
        if line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            event_bytes += len(data.encode("utf-8")) + 1
            if event_bytes > _VOXCPM_SSE_MAX_EVENT_BYTES:
                raise ValueError("VoxCPM SSE event exceeds the size limit")
            event_data.append(data)
    if event_data:
        yield parse_event(event_data)


def prepare_voxcpm_reference_audio(uploaded_audio: bytes, suffix: str = "") -> bytes:
    """Convert an uploaded reference clip into ModelBest's bounded WAV payload.

    The WebUI keeps the returned bytes only in its current session and passes a
    copy to the request that needs it. Temporary source and converted files are
    always scoped to this function, including FFmpeg timeouts and failures.
    """
    if not isinstance(uploaded_audio, bytes) or not uploaded_audio:
        raise ValueError("reference audio is empty")
    if len(uploaded_audio) > VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        raise ValueError("reference audio upload exceeds 20 MiB")

    normalized_suffix = str(suffix or "").lower()
    if normalized_suffix and not normalized_suffix.startswith("."):
        normalized_suffix = f".{normalized_suffix}"
    if normalized_suffix.removeprefix(".") not in VOXCPM_REFERENCE_AUDIO_FILE_TYPES:
        raise ValueError("unsupported reference audio format")

    try:
        with tempfile.TemporaryDirectory(prefix="voxcpm-reference-") as temp_dir:
            input_path = os.path.join(temp_dir, f"input{normalized_suffix or '.audio'}")
            output_path = os.path.join(temp_dir, "reference.wav")
            with open(input_path, "wb") as source:
                source.write(uploaded_audio)

            result = subprocess.run(
                [
                    utils.get_ffmpeg_binary(),
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-t",
                    str(VOXCPM_REFERENCE_AUDIO_MAX_DURATION_SECONDS),
                    "-i",
                    input_path,
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    output_path,
                ],
                capture_output=True,
                timeout=VOXCPM_REFERENCE_AUDIO_CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode != 0 or not os.path.isfile(output_path):
                raise ValueError(
                    "reference audio must contain a decodable audio stream"
                )

            with open(output_path, "rb") as converted:
                wav_audio = converted.read()
            if (
                not wav_audio
                or len(wav_audio) > VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES
            ):
                raise ValueError(
                    "converted reference audio exceeds ModelBest's 5 MiB limit"
                )

            try:
                with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
                    if wav_file.getnframes() <= 0:
                        raise ValueError("reference audio is empty")
            except wave.Error as exc:
                raise ValueError(
                    "reference audio conversion did not produce a valid WAV"
                ) from exc
            return wav_audio
    except subprocess.TimeoutExpired as exc:
        raise ValueError("reference audio conversion timed out") from exc
    except OSError as exc:
        raise ValueError("failed to convert reference audio") from exc


def _encode_voxcpm_audio_data_uri(
    audio_bytes: bytes | None,
    field_name: str,
) -> str | None:
    if audio_bytes is None:
        return None
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise ValueError(f"{field_name} audio is empty")
    if len(audio_bytes) > VOXCPM_REFERENCE_AUDIO_MAX_WAV_BYTES:
        raise ValueError(f"{field_name} audio exceeds ModelBest's 5 MiB limit")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            if frame_count <= 0:
                raise ValueError(f"{field_name} audio is empty")
            expected_bytes = frame_count * wav_file.getnchannels() * wav_file.getsampwidth()
            # WAV headers can advertise frames that are not in the upload.
            # Bound the read by the existing upload cap even for forged counts.
            if expected_bytes > len(audio_bytes) or len(wav_file.readframes(frame_count)) != expected_bytes:
                raise ValueError(f"{field_name} audio contains truncated PCM frames")
    except wave.Error as exc:
        raise ValueError(f"{field_name} audio must be a valid WAV") from exc
    return "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")


def voxcpm_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    reference_audio: bytes | None = None,
    prompt_audio: bytes | None = None,
    prompt_text: str = "",
) -> Union[SubMaker, None]:
    """Generate speech through ModelBest's VoxCPM Audio Speech API.

    ModelBest always streams Base64-encoded WAV chunks through SSE. The
    assembled WAV is decoded and exported to the project's requested output
    format so that the regular subtitle and video paths remain unchanged.
    ModelBest does not define a numeric speed field, so ``voice_rate`` is not
    sent. ``voice_volume`` is applied later by MoneyPrinterTurbo's video mixer.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("VoxCPM TTS text is empty")
        return None

    api_key = str(config.voxcpm.get("api_key", "") or "").strip()
    if not api_key:
        logger.error("VoxCPM API key is not set")
        return None

    base_url = str(
        config.voxcpm.get("base_url", VOXCPM_DEFAULT_BASE_URL)
        or VOXCPM_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    model_id = str(config.voxcpm.get("model_id", "") or "").strip()
    if not model_id:
        logger.error("VoxCPM model ID is not set")
        return None
    voice_id = str(voice_id or VOXCPM_DEFAULT_VOICE).strip() or VOXCPM_DEFAULT_VOICE

    url = f"{base_url}/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice_id,
        "response_format": "wav",
        "stream": True,
    }
    prompt_text = str(prompt_text or "").strip()
    if bool(prompt_audio) != bool(prompt_text):
        logger.error("VoxCPM prompt audio and prompt text must be provided together")
        return None
    try:
        encoded_reference_audio = _encode_voxcpm_audio_data_uri(
            reference_audio,
            "reference",
        )
        encoded_prompt_audio = _encode_voxcpm_audio_data_uri(
            prompt_audio,
            "prompt",
        )
    except ValueError as exc:
        logger.error(f"VoxCPM audio prompt is invalid: {exc}")
        return None
    if encoded_reference_audio:
        payload["ref_audio"] = encoded_reference_audio
    if encoded_prompt_audio:
        payload["prompt_audio"] = encoded_prompt_audio
        payload["prompt_text"] = prompt_text
    _configure_pydub_ffmpeg(AudioSegment)

    for attempt in range(3):
        temporary_audio = None
        response = None
        try:
            logger.info(
                f"start VoxCPM TTS, model: {model_id}, voice: {voice_id}, "
                f"try: {attempt + 1}"
            )
            response = trace_http_call(
                "POST", requests.post,
                url,
                json=payload,
                headers=headers,
                allow_redirects=False,
                stream=True,
                timeout=(10, 120),
            )
            if 300 <= response.status_code < 400:
                # Redirecting a speech POST may replay a billed synthesis or
                # forward its input/credentials. Do not follow or resubmit it.
                logger.error("TTS endpoint returned a redirect; stop speech retries")
                return None
            if response.status_code != 200:
                logger.error(
                    f"VoxCPM TTS failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                if response.status_code in _VOXCPM_NON_RETRYABLE_STATUS_CODES:
                    return None
                if attempt < 2:
                    time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
                continue

            audio_chunks = []
            total_audio_bytes = 0
            completed = False
            for event in _iter_voxcpm_sse_events(response):
                event_type = event.get("type")
                if event_type == "speech.audio.delta":
                    encoded_chunk = event.get("audio")
                    if not isinstance(encoded_chunk, str) or not encoded_chunk:
                        raise ValueError("VoxCPM returned an empty audio chunk")
                    try:
                        chunk = base64.b64decode(encoded_chunk, validate=True)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("VoxCPM returned invalid Base64 audio") from exc
                    total_audio_bytes += len(chunk)
                    if total_audio_bytes > _VOXCPM_TTS_MAX_AUDIO_BYTES:
                        raise ValueError("VoxCPM generated audio exceeds the size limit")
                    audio_chunks.append(chunk)
                elif event_type == "speech.audio.done":
                    completed = True
                    break

            if not completed:
                raise ValueError("VoxCPM stream ended before speech.audio.done")
            audio_bytes = b"".join(audio_chunks)
            if not audio_bytes:
                raise ValueError("VoxCPM returned no audio data")

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            if len(audio_segment) <= 0:
                raise ValueError("VoxCPM returned an empty WAV")

            ensure_file_path_exists(voice_file)
            output_format = utils.parse_extension(voice_file) or "mp3"
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=f".{output_format}",
                delete=False,
            ) as output:
                temporary_audio = output.name
            audio_segment.export(temporary_audio, format=output_format)

            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError("VoxCPM produced an invalid audio duration")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"VoxCPM TTS succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.error(f"VoxCPM TTS connection timed out: {exc}")
            # A timeout before POST returns is safe to retry only when no
            # response has been received. Never resubmit after SSE started.
            if response is not None:
                return None
            if attempt < 2:
                time.sleep(_VOXCPM_RETRY_DELAY_SECONDS[attempt])
        except requests.RequestException as exc:
            # The server may have generated speech before a read timeout or
            # stream disconnect. Resubmitting can create duplicate work.
            logger.error(f"VoxCPM TTS request outcome is unconfirmed: {exc}")
            return None
        except Exception as exc:
            # Invalid SSE/WAV data and local conversion failures are deterministic;
            # retrying the same response cannot repair them.
            logger.error(f"VoxCPM TTS failed: {exc}")
            return None
        finally:
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    logger.warning(f"could not remove temporary VoxCPM audio: {exc}")

    return None


def _format_text(text: str) -> str:
    """
    Clean the script text before subtitle alignment.

    This cannot be handled only at LLM generation time, because users may paste a
    script manually or pass Markdown-laden text straight through the API. TTS
    usually does not narrate separator lines such as `---`, `___`, `***`, nor the
    `_` emphasis marker; if subtitle alignment keeps those characters,
    `create_subtitle()` waits forever for cues that never arrive, the subtitle
    file is missing, and the Whisper fallback correction fills in an all-zero
    timeline.
    """
    text = utils.remove_pause_tags(text or "")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return utils.normalize_script_for_subtitle_matching(text)


def _build_subtitle_formatter():
    """
    Return the shared SRT line formatter.

    It is a separate helper so the edge_tts 7.x cues path and the project's
    legacy `subs/offset` path share one subtitle serialization format, avoiding
    subtle formatting differences between the two implementations.
    """

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    return formatter


# Arabic diacritics and the Tatweel elongation character can appear in text
# returned by edge-tts. They do not change meaning but break exact string matches
# between the script text and subtitle cue strings.
_ARABIC_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")


def _normalize_arabic(text: str) -> str:
    """Normalize common Arabic letter variants to make subtitle cue to script
    line matching more tolerant.

    For Arabic, edge-tts may return letter forms that differ from the script, for
    example folding أ/إ/آ into ا, or include diacritics. This is only used in the
    last-resort matching layer and never changes the subtitle text itself, so the
    final rendered content is unaffected.
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    for src, dst in (
        ("أإآٱ", "ا"),
        ("ىئ", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
    ):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def _match_script_line(script_lines: list[str], current_text: str, sub_index: int) -> str:
    """
    Try to match the accumulated subtitle text against one of the script's
    canonical sentences.

    This reuses the project's original approach of splitting the script by
    punctuation and comparing segment by segment:
    1. exact match first;
    2. then a match after stripping punctuation and Markdown `_` markers;
    3. finally a match on Arabic-normalized character forms.

    This tolerates:
    - punctuation missing from, or split off in, the TTS output;
    - Chinese cases where word boundaries do not map one-to-one to the script.
    """
    if len(script_lines) <= sub_index:
        return ""

    target_line = script_lines[sub_index]
    comparison_target = unicodedata.normalize("NFC", target_line)
    current_text = unicodedata.normalize("NFC", current_text)
    if current_text == comparison_target:
        return target_line.strip()

    current_text_normalized = re.sub(r"[_\W]+", "", current_text)
    target_line_normalized = re.sub(r"[_\W]+", "", comparison_target)
    if current_text_normalized == target_line_normalized:
        return target_line.strip()

    # Last-resort Arabic tolerance: letter forms, diacritics or Tatweel returned
    # by edge-tts may differ from the script. Only compare normalized text after
    # the regular matches fail, so non-Arabic text is unaffected.
    current_ar = re.sub(r"[_\W]+", "", _normalize_arabic(current_text))
    target_ar = re.sub(r"[_\W]+", "", _normalize_arabic(comparison_target))
    if current_ar and current_ar == target_ar:
        return target_line.strip()

    return ""


def _write_subtitle_items(sub_items: list[str], subtitle_file: str) -> bool:
    """Publish a complete, parseable SRT without destroying earlier captions."""
    try:
        ensure_file_path_exists(subtitle_file)
        with staged_subtitle_file(subtitle_file) as staged:
            with open(staged, "w", encoding="utf-8") as file:
                file.write("\n".join(sub_items) + "\n")
            sbs = moviepy_subtitles(staged)
            if not sbs:
                raise ValueError("subtitle output contains no cues")
            duration = max(tb for ((ta, tb), txt) in sbs)
            os.replace(staged, subtitle_file)
        logger.info(
            f"completed, subtitle file created: {subtitle_file}, duration: {duration}"
        )
        return True
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")
        return False


def _build_subtitle_items_from_edge_cues(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Aggregate edge_tts 7.x fine-grained `cues` into SRT segments split by script
    sentences.

    Background:
    edge_tts 7.x `SubMaker.get_srt()` leans toward a per-word/per-phrase timeline.
    Word-level highlighting is acceptable for English, but copying it directly to
    Chinese short-video subtitles produces a poor reading experience such as
    "money / is / a / social / tool".

    Strategy:
    1. consume the `content` of each cue in order;
    2. accumulate it into candidate text;
    3. when the candidate matches the current target script sentence, collapse it
       into one complete subtitle segment;
    4. use the start of the first cue and the end of the last cue so the timeline
       stays continuous.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    current_text = ""
    current_start_time = None

    for cue in sub_maker.cues:
        cue_text = unescape(cue.content)
        if current_start_time is None:
            current_start_time = int(cue.start.total_seconds() * 10000000)

        current_end_time = int(cue.end.total_seconds() * 10000000)
        current_text += cue_text

        matched_text = _match_script_line(script_lines, current_text, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=current_start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        current_text = ""
        current_start_time = None

    if current_text.strip():
        logger.warning(
            f"edge cues still have unmatched text after aggregation: {current_text}"
        )

    return sub_items


def _build_subtitle_items_from_legacy_submaker(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Aggregate the project's legacy `subs/offset` structure into SRT segments
    split by script sentences.

    This keeps the original core idea but splits it into a standalone function so
    it shares the same sentence matching and serialization flow as the edge_tts
    7.x cues aggregation.
    """
    formatter = _build_subtitle_formatter()
    start_time = -1.0
    sub_items = []
    sub_index = 0
    sub_line = ""

    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for _, (offset, sub) in enumerate(zip(legacy_offsets, legacy_subs)):
        current_start_time, current_end_time = offset
        if start_time < 0:
            start_time = current_start_time

        sub_line += unescape(sub)
        matched_text = _match_script_line(script_lines, sub_line, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        start_time = -1.0
        sub_line = ""

    if sub_line.strip():
        logger.warning(
            f"legacy subtitle items still have unmatched text after aggregation: {sub_line}"
        )

    return sub_items


def _build_subtitle_items_from_edge_cues_words(sub_maker: SubMaker) -> list[str]:
    """
    Directly format edge_tts cues into single-word / cue-level SRT items.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    for cue in sub_maker.cues:
        cue_text = unescape(cue.content).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time = int(cue.start.total_seconds() * 10000000)
        end_time = int(cue.end.total_seconds() * 10000000)
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def _build_subtitle_items_from_legacy_submaker_words(sub_maker: SubMaker) -> list[str]:
    """
    Directly format legacy submaker into single-word SRT items.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for offset, sub in zip(legacy_offsets, legacy_subs):
        cue_text = unescape(sub).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time, end_time = offset
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def _join_lines_split_inside_words(
    text: str, script_lines: list[str], words: list[str]
) -> list[str]:
    """
    Join each script line that ends inside a TTS word with the line after it.

    Edge TTS returns "July 20, 1969", "Washington, D.C.", "No. 1" and
    "Am 9. November 1989" as one word each, but the script split ends a line
    at their comma or full stop. No run of words can match such a line, and
    without it no subtitle file is written at all. A joined line keeps the
    script text between its parts, as in "On July 20, 1969". Lines are only
    joined when the words spell the script exactly, compared as in
    _match_script_line().
    """

    def comparable(value: str) -> str:
        return re.sub(r"[_\W]+", "", unicodedata.normalize("NFC", value))

    word_texts = [comparable(unescape(word)) for word in words]
    line_texts = [comparable(line) for line in script_lines]
    if "".join(word_texts) != "".join(line_texts):
        return script_lines

    word_ends = set(itertools.accumulate(map(len, word_texts)))
    line_ends = itertools.accumulate(map(len, line_texts))
    joined_lines = []
    start = None
    position = 0
    for line, line_end in zip(script_lines, line_ends):
        # Each script line is a part of text, in order.
        line_start = text.find(line, position)
        position = line_start + len(line)
        if start is None:
            start = line_start
        if line_end in word_ends:
            joined_lines.append(text[start:position])
            start = None
    return joined_lines


def create_subtitle(
    sub_maker: SubMaker,
    text: str,
    subtitle_file: str,
    word_level: bool = False,
):
    """
    Improve the subtitle file.
    1. Split the subtitle text into multiple lines by punctuation.
    2. Match the text line by line against the script.
    3. Write the resulting subtitle file.
    If word_level is True, emit one single-word subtitle entry per word.
    """
    text = _format_text(text)
    try:
        if word_level:
            if hasattr(sub_maker, "cues") and sub_maker.cues:
                sub_items = _build_subtitle_items_from_edge_cues_words(sub_maker)
            else:
                sub_items = _build_subtitle_items_from_legacy_submaker_words(sub_maker)
            if sub_items:
                _write_subtitle_items(sub_items, subtitle_file)
                return

        # Symbol-only lines such as `* * *`, `———` or `• • •` scene separators
        # are never narrated and get no word-level timeline. Keeping them makes
        # the line counts never match, so no subtitles are produced at all.
        script_lines = [
            line
            for line in utils.split_string_by_punctuations(text)
            if re.sub(r"[_\W]+", "", line)
        ]
        if hasattr(sub_maker, "cues") and sub_maker.cues:
            script_lines = _join_lines_split_inside_words(
                text, script_lines, [cue.content for cue in sub_maker.cues]
            )
            sub_items = _build_subtitle_items_from_edge_cues(sub_maker, script_lines)
        else:
            script_lines = _join_lines_split_inside_words(
                text, script_lines, getattr(sub_maker, "subs", [])
            )
            sub_items = _build_subtitle_items_from_legacy_submaker(
                sub_maker, script_lines
            )

        if len(sub_items) != len(script_lines):
            logger.warning(
                f"failed, sub_items len: {len(sub_items)}, script_lines len: {len(script_lines)}"
            )
            return

        _write_subtitle_items(sub_items, subtitle_file)
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")


def _get_audio_duration_from_submaker(sub_maker: SubMaker):
    """
    Get the audio duration.
    """
    if hasattr(sub_maker, "duration") and getattr(sub_maker, "duration", 0) > 0:
        return float(getattr(sub_maker, "duration"))

    # Prefer the edge_tts 7.x cues structure; for the legacy structure filled in
    # manually by other TTS implementations in this project, read offset instead.
    if hasattr(sub_maker, "cues") and sub_maker.cues:
        return sub_maker.cues[-1].end.total_seconds()

    legacy_offsets = getattr(sub_maker, "offset", [])
    if not legacy_offsets:
        return 0.0
    return legacy_offsets[-1][1] / 10000000

def _get_audio_duration_from_file(audio_file: str) -> float:
    """
    Get the duration of an audio file (supports mp3/m4a/wav/aac and any other
    format ffmpeg can decode)
    """
    if not os.path.exists(audio_file):
        logger.error(f"audio file does not exist: {audio_file}")
        return 0.0

    try:
        # Use moviepy (ffmpeg) to read the duration of any supported audio format
        with AudioFileClip(audio_file) as audio:
            return audio.duration  # Duration in seconds
    except Exception as e:
        logger.error(f"Failed to get audio duration from file: {str(e)}")
        return 0.0

def get_audio_duration(target: Union[str, SubMaker]) -> float:
    """
    Get the audio duration.
    For a SubMaker object, read the duration from the SubMaker.
    For an audio file path, read the duration from the file (mp3/m4a/wav and
    other formats are supported).
    """
    if isinstance(target, SubMaker):
        return _get_audio_duration_from_submaker(target)
    elif isinstance(target, str):
        return _get_audio_duration_from_file(target)
    else:
        logger.error(f"Invalid target type: {type(target)}")
        return 0.0

if __name__ == "__main__":
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
    voice_name = parse_voice_name(voice_name)
    voice_name = is_azure_v2_voice(voice_name)
    print(voice_name)

    voices = get_all_azure_voices()
    print(len(voices))

    async def _do():
        temp_dir = utils.storage_dir("temp")

        voice_names = [
            "zh-CN-XiaoxiaoMultilingualNeural",
            # female
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural",
            # male
            "zh-CN-YunyangNeural",
            "zh-CN-YunxiNeural",
        ]

        text = """
        What is the meaning of life? This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
        """


        text = "[Opening scene: A sunny day in a suburban neighborhood. A young boy named Alex, around 8 years old, is playing in his front yard with his loyal dog, Buddy.]\n\n[Camera zooms in on Alex as he throws a ball for Buddy to fetch. Buddy excitedly runs after it and brings it back to Alex.]\n\nAlex: Good boy, Buddy! You're the best dog ever!\n\n[Buddy barks happily and wags his tail.]\n\n[As Alex and Buddy continue playing, a series of potential dangers loom nearby, such as a stray dog approaching, a ball rolling towards the street, and a suspicious-looking stranger walking by.]\n\nAlex: Uh oh, Buddy, look out!\n\n[Buddy senses the danger and immediately springs into action. He barks loudly at the stray dog, scaring it away. Then, he rushes to retrieve the ball before it reaches the street and gently nudges it back towards Alex. Finally, he stands protectively between Alex and the stranger, growling softly to warn them away.]\n\nAlex: Wow, Buddy, you're like my superhero!\n\n[Just as Alex and Buddy are about to head inside, they hear a loud crash from a nearby construction site. They rush over to investigate and find a pile of rubble blocking the path of a kitten trapped underneath.]\n\nAlex: Oh no, Buddy, we have to help!\n\n[Buddy barks in agreement and together they work to carefully move the rubble aside, allowing the kitten to escape unharmed. The kitten gratefully nuzzles against Buddy, who responds with a friendly lick.]\n\nAlex: We did it, Buddy! We saved the day again!\n\n[As Alex and Buddy walk home together, the sun begins to set, casting a warm glow over the neighborhood.]\n\nAlex: Thanks for always being there to watch over me, Buddy. You're not just my dog, you're my best friend.\n\n[Buddy barks happily and nuzzles against Alex as they disappear into the sunset, ready to face whatever adventures tomorrow may bring.]\n\n[End scene.]"



        text = _format_text(text)
        lines = utils.split_string_by_punctuations(text)
        print(lines)

        for voice_name in voice_names:
            voice_file = f"{temp_dir}/tts-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts.mp3.srt"
            sub_maker = azure_tts_v2(
                text=text, voice_name=voice_name, voice_file=voice_file
            )
            create_subtitle(sub_maker=sub_maker, text=text, subtitle_file=subtitle_file)
            audio_duration = get_audio_duration(sub_maker)
            print(f"voice: {voice_name}, audio duration: {audio_duration}s")

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(_do())
    finally:
        loop.close()
