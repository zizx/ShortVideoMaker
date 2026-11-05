import base64
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import io
import math
import os
import random
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import ImageMotion, MaterialInfo, VideoAspect, VideoConcatMode
from app.services import (
    codex_image,
    generation_progress,
    infistar_video,
    jalapeno_video,
    material_cache,
    material_upload,
    metaso_minimax,
    muapi,
    ofox,
    task_artifacts,
    video,
    volcengine_seedance,
)
from app.utils import logging_utils, utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# A provider URL can point to an unexpectedly large object or never-ending stream.
# Short stock and generated clips should stay well below this conservative cap.
MAX_VIDEO_DOWNLOAD_BYTES = 512 * 1024 * 1024

# Serial by default, matching the legacy behavior; raise the stock material
# concurrency in the config when needed.
_DEFAULT_MATERIAL_CONCURRENCY = 1


def _get_material_concurrency() -> int:
    try:
        concurrency = int(config.app.get("material_concurrency", _DEFAULT_MATERIAL_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _DEFAULT_MATERIAL_CONCURRENCY
    return max(1, min(8, concurrency))


class _OpenAIImageDecodeError(ValueError):
    """The compatible API returned bytes that cannot be decoded as an image.

    Local file write failures are not covered by this error.
    """


class OpenAIImagePaidResultError(RuntimeError):
    """A paid image request cannot provide a usable local video material."""


class OpenAIImageUnconfirmedError(OpenAIImagePaidResultError):
    """A paid image request may have succeeded without returning a response."""


class CodexImageMaterialError(OpenAIImagePaidResultError):
    """Stop the image loop while retaining a translatable, sanitized error."""

    def __init__(self, error):
        self.key = error.key
        self.params = error.params
        messages = {
            "codex.image.save_failed": "Could not save the generated Codex image. Generation stopped without retrying.",
            "codex.image.render_failed": "Could not render the generated Codex image. Generation stopped without retrying.",
        }
        super().__init__(messages.get(self.key, str(error)))


def _safe_public_url(value: Any) -> str | None:
    """
    Keep only publicly shareable HTTP(S) page URLs, dropping query and credentials.

    Material download URLs may carry an API key, a signed JWT or a temporary
    token. The task manifest only needs to help users get back to the provider's
    public material page, so auth parameters must not be stored; URLs with user
    info are rejected as well, keeping ``https://user:pass@example.com``-style
    content off disk.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _creator_info(value: Any) -> dict[str, str] | None:
    """Extract unified public fields from the creator structures of providers."""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    Build a lightweight source record for a successfully downloaded material.

    ``source_info`` may come from a cache or even from an externally constructed
    ``MaterialInfo``, so it cannot be written as-is. Rebuild it from a whitelist
    here, keeping only public pages, business identifiers and dimensions, and
    record only the local file name so user directories or Docker mount paths do
    not leak into the task file.
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    Append the sources of the materials actually downloaded to the task manifest.

    Task recording is an auxiliary capability: it must not change the return
    value of the video download function, nor interrupt the main video pipeline
    when writing fails. ``patch_script_data`` handles atomic replacement and
    exception logging; here we only log the count on success so it is easy to
    confirm the task traceability data reached disk.
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts already degrades gracefully on failure, but keep a final
        # isolation layer so future implementation changes or directory
        # resolution errors cannot affect the material download return value.
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    # TLS certificate verification is on by default, preventing man-in-the-middle
    # tampering during material search and download. Only in scenarios that
    # clearly require it (corporate proxies, self-signed certificates) may users
    # temporarily disable it via an explicit `tls_verify = false` in `config.toml`.
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    Minimally redact exception text that is about to be written to the log.

    requests connection exceptions may contain the full request URL, and the
    Pixabay API key is passed as a query parameter. Replace both the raw value
    and its URL-encoded form here, keeping the network error information for
    troubleshooting while preventing the key from reaching the log file.
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    Keep network exception details for troubleshooting while removing API keys
    and proxy credentials.

    Logging only the exception type loses key context such as DNS, certificate
    and timeout failures; logging the raw exception may echo the full request
    URL. A single entry point gives all three material providers the same
    redaction rules.
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    Detect the HTML challenge returned by Cloudflare instead of treating it as
    Pixabay JSON.

    Cloudflare usually sets `cf-mitigated: challenge`; some deployments only
    return HTML containing "Just a moment" or challenge-platform, so keep a
    content-based fallback. The response body is inspected in memory only and
    never logged, avoiding large, useless HTML dumps.
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _stock_search_results(
    provider: str, response: Any, results_key: str
) -> list | None:
    """
    Return the result list of a Pexels or Coverr search response.

    When the search failed, log why and return None. As in the Pixabay search,
    the log gets the status and content type but not the response body, and a
    401 names the setting that holds the rejected key.
    """
    status_code = int(getattr(response, "status_code", 200))
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", "")) or "unknown"
    if status_code == 401:
        logger.error(
            f"{provider} rejected the API key: status=401. "
            f"Check {provider}_api_keys in config.toml."
        )
        return None
    if status_code == 429:
        logger.error(
            f"{provider} API rate limit exceeded: "
            f"status=429, retry_after={headers.get('retry-after') or 'unknown'}"
        )
        return None
    if status_code >= 400:
        logger.error(
            f"{provider} search request failed: "
            f"status={status_code}, content_type={content_type}"
        )
        return None
    try:
        body = response.json()
    except ValueError:
        logger.error(
            f"{provider} returned an unexpected non-JSON response: "
            f"status={status_code}, content_type={content_type}"
        )
        return None
    if not isinstance(body, dict) or not isinstance(body.get(results_key), list):
        logger.error(f"{provider} video search returned an unsupported response")
        return None
    return body[results_key]


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    Check whether a remote material matches the target frame orientation.

    Pexels, Pixabay and Coverr do not share response fields, so width and height
    are used first for a reliable decision; when some historical Coverr responses
    lack dimensions, fall back to the explicit ``is_vertical`` boolean. Materials
    whose orientation cannot be confirmed are skipped, preventing landscape clips
    from entering portrait tasks and causing black bars in the final video.
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (OverflowError, TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Re-validate orientation for cached results.

    The material search cache is kept for up to 24 hours, so entries written
    before an upgrade may contain materials with a mismatched orientation.
    Filtering at the shared cache entry point makes the fix effective
    immediately and guards against third-party providers or old caches that
    missed the remote filtering. Old entries without readable rendition
    dimensions are treated as unverified and skipped.
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay and Coverr rarely provide native square material. Square output
        # keeps the existing behavior: accept usable candidates and let the video
        # composition stage crop them, so 1:1 tasks still have material after the
        # upgrade.
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20}
    if aspect != VideoAspect.square:
        params["orientation"] = aspect.name
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        videos = _stock_search_results("pexels", r, "videos")
        if videos is None:
            return []
        video_items = []
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("video_files")
            if not isinstance(video_files, list):
                continue
            # Prefer the smallest rendition that can fill the output canvas.
            # Square output can crop either orientation, as other providers do.
            renditions = []
            for video in video_files:
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video.get("width"))
                    h = int(video.get("height"))
                except (OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("link")
                if not isinstance(video_url, str) or not video_url:
                    continue
                if (
                    (aspect == VideoAspect.square or _matches_video_aspect(w, h, aspect))
                    and w >= video_width
                    and h >= video_height
                ):
                    renditions.append((w * h, video, w, h, video_url))
            if renditions:
                _, video, w, h, video_url = min(renditions, key=lambda candidate: candidate[0])
                item = MaterialInfo()
                item.provider = "pexels"
                item.url = video_url
                item.duration = duration
                item.source_info = {
                    "provider": "pexels",
                    "search_term": search_term,
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("url")),
                    "creator": _creator_info(v.get("user")),
                    "rendition": {
                        "id": (
                            str(video.get("id"))
                            if video.get("id") is not None
                            else None
                        ),
                        "width": w,
                        "height": h,
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels/Coverr instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("videos")
            if not isinstance(video_files, dict):
                continue
            # loop through each url to determine the best quality
            for video_type, video in video_files.items():
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("url")
                if not isinstance(video_url, str) or not video_url:
                    continue
                # Pixabay rarely returns native square videos; 1:1 output keeps
                # accepting candidates that meet the resolution and lets the
                # composition stage crop them. Portrait and landscape must match
                # the target orientation strictly.
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width and h >= video_height:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video_url
                    item.duration = duration
                    item.source_info = {
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": (
                            str(v.get("id")) if v.get("id") is not None else None
                        ),
                        "source_page": _safe_public_url(v.get("pageURL")),
                        "creator": _creator_info(
                            {
                                "id": v.get("user_id"),
                                "name": v.get("user"),
                            }
                        ),
                        "rendition": {
                            "id": video_type,
                            "width": w,
                            "height": video.get("height"),
                        },
                    }
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - Auth: Authorization: Bearer <api_key>
      - Search endpoint: GET /videos?query=..., response shape {"hits": [...], ...}
      - Add ?urls=true to get direct mp4 links in the search response
      - The URL is a signed JWT (bound to the API key, no expiry)
      - Coverr can filter portrait/landscape material via
        filter=is_vertical:true/false; after the response returns, still validate
        locally using max_width/max_height or is_vertical
      - The duration field appears both as a number and as a string; this
        function accepts both

    This function uses the urls.mp4_download field as the download address --
    according to the official Coverr docs
    (https://api.coverr.co/docs/videos/#download-a-video), a GET on this URL is
    itself counted by Coverr as a legitimate download event, so there is no need
    to call PATCH /videos/:id/stats/downloads.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    # Server-side orientation filtering returns matching material directly from
    # the full search results, avoiding an empty portrait candidate list caused by
    # taking popular results first and filtering locally. Square material has no
    # corresponding boolean condition, so it still relies on local width/height
    # validation.
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        hits = _stock_search_results("coverr", r, "hits")
        if hits is None:
            return []
        video_items: List[MaterialInfo] = []

        for v in hits:
            if not isinstance(v, dict):
                continue
            # duration may be a number (11.625) or a string ("10.500000") in
            # different responses
            try:
                duration = int(float(v.get("duration") or 0))
            except (OverflowError, TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            urls = v.get("urls")
            if not isinstance(urls, dict):
                continue
            mp4_download_url = urls.get("mp4_download")
            if (
                not video_id
                or not isinstance(mp4_download_url, str)
                or not mp4_download_url
            ):
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


# WaveSpeed AI (https://wavespeed.ai) generates material directly from script
# keywords via text-to-video models, sharing the MaterialInfo result structure
# and the downstream download and editing pipeline with the three stock sources.
WAVESPEED_API_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_DEFAULT_T2V_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
WAVESPEED_POLL_INTERVAL_SECONDS = 2.0
WAVESPEED_RUN_TIMEOUT_SECONDS = 600.0
# The default model bytedance/seedance-2.0-fast/text-to-video only accepts 4-15
# seconds; requests outside that range are rejected by the API. The WebUI default
# clip duration is 3 seconds, so it must be clamped to the model-supported range
# before submission; any excess is trimmed by the existing editing pipeline.
WAVESPEED_MIN_DURATION_SECONDS = 4
WAVESPEED_MAX_DURATION_SECONDS = 15
# The three failure states differ in meaning (model error / user cancellation /
# platform timeout), but for the material pipeline they all mean this keyword
# produced nothing, so they are treated uniformly as an empty result and the
# caller skips the clip and continues generation.
WAVESPEED_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})
# Match the official WaveSpeed Python SDK / n8n node: 429 and 5xx are transient
# failures worth a limited number of backoff retries; 4xx is an explicit client
# error and fails fast.
WAVESPEED_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Consecutive transient failures allowed per polling run. One unlucky GET must
# not lose track of an already-billed task.
WAVESPEED_MAX_POLL_RETRIES = 5
# Linear backoff base; the n-th retry waits base * n seconds.
WAVESPEED_RETRY_BASE_SECONDS = 1.0
# Retry count for the same signed URL when the output download fails. The
# material is already paid for, so retry the original URL first; a single
# download hiccup must not trigger another paid generation task.
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2


class WaveSpeedUnconfirmedTaskError(RuntimeError):
    """
    A paid generation task was submitted, but its final state cannot be
    confirmed locally.

    This exception never means "the task failed and can be retried": the remote
    task may still be running or may have completed and been billed. The
    material pipeline must stop here, submit no new paid tasks for subsequent
    keywords, and pass the submitted prediction id to the task status so a human
    can recover it.
    """

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


class WaveSpeedDownloadError(RuntimeError):
    """A paid prediction completed, but its video could not be saved locally."""

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


def _wavespeed_status_code(response: Any) -> int:
    """Read the response status code; default to 200 when a test double or
    exception object lacks the field."""
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _is_wavespeed_retryable_error(error: Exception) -> bool:
    """
    Decide whether a polling exception is worth retrying.

    Network errors such as connection and timeout failures have no status code
    and are treated as transient; responses with a status code are retried only
    for 429 and 5xx, matching the official SDK's retry set.
    """
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    if response is not None:
        return _wavespeed_status_code(response) in WAVESPEED_RETRYABLE_STATUS_CODES
    return False


def _wavespeed_duration_bounds() -> tuple[int, int]:
    """
    Return the generation duration range (seconds) supported by the current model.

    The default range corresponds to the default Seedance model; when users
    switch to another text-to-video model they can adjust the range in the
    config. Any invalid configuration falls back to the defaults, and min <= max
    is guaranteed so user input never becomes a remote request bound to fail.
    """

    def read_bound(key: str, fallback: int) -> int:
        try:
            value = int(config.app.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value >= 1 else fallback

    min_duration = read_bound("wavespeed_min_duration", WAVESPEED_MIN_DURATION_SECONDS)
    max_duration = read_bound("wavespeed_max_duration", WAVESPEED_MAX_DURATION_SECONDS)
    return min_duration, max(max_duration, min_duration)


def generate_videos_wavespeed(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Generate one material clip for a script keyword with a WaveSpeed
    text-to-video model.

    Keeps the same signature and empty-list failure contract as the stock
    sources' search_videos_*, so it plugs directly into the generic download and
    duration accounting flow of ``download_videos``. In the generation context,
    ``minimum_duration`` is the target clip duration in seconds.
    """
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("wavespeed_api_keys")
    model_id = (
        str(
            config.app.get("wavespeed_text_to_video_model", "")
            or WAVESPEED_DEFAULT_T2V_MODEL
        )
        .strip()
        .strip("/")
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    requested_duration = max(int(minimum_duration), 1)
    min_duration, max_duration = _wavespeed_duration_bounds()
    duration = min(max(requested_duration, min_duration), max_duration)
    if duration != requested_duration:
        # Generating longer than requested does not affect the final video: the
        # editing pipeline still trims to the clip duration. Generating shorter
        # than requested only happens when the request exceeds the model limit,
        # in which case clamping to the limit is the only option.
        logger.info(
            f"wavespeed clip duration clamped to model-supported range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(supported {min_duration}-{max_duration}s)"
        )
    payload = {
        "prompt": search_term,
        "aspect_ratio": aspect.value,
        "duration": duration,
    }
    logger.info(
        f"generating video on wavespeed: model={model_id}, "
        f"term={search_term!r}, duration={duration}s"
    )

    # Never auto-retry the submission POST: the request may already have created
    # a paid task remotely, and resending would cause duplicate generation and
    # duplicate billing (matching the official SDK's submission policy).
    generation_progress.submitting("wavespeed")
    try:
        submit_response = trace_http_call(
            "POST", requests.post,
            f"{WAVESPEED_API_BASE_URL}/{model_id}",
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
    except Exception as e:
        # No response does not mean the task was not created. The state is
        # unknown, so the whole generation flow must stop instead of submitting
        # new paid tasks for the next keyword.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission did not return a response, the task may "
            "already exist remotely: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        ) from e

    submit_status = _wavespeed_status_code(submit_response)
    if submit_status >= 500:
        # A 5xx may occur after the task was created, so billing cannot be ruled
        # out.
        raise WaveSpeedUnconfirmedTaskError(
            f"wavespeed submission failed with HTTP {submit_status}, "
            "the task may already exist remotely"
        )
    try:
        submit_body = submit_response.json()
    except Exception as e:
        if 400 <= submit_status < 500:
            # A 4xx from a gateway or auth layer may be plain text/HTML. A 4xx is
            # still an explicit rejection and no remote task was created, so it
            # must not be misreported as "possibly billed" and abort the task.
            logger.error(
                "wavespeed video generation request rejected: "
                f"http_status={submit_status}, unreadable body: {type(e).__name__}"
            )
            return []
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission returned an unreadable response, the task "
            f"may already exist remotely: error={type(e).__name__}"
        ) from e

    submit_data = submit_body.get("data") if isinstance(submit_body, dict) else None
    if not isinstance(submit_body, dict) or submit_body.get("code") != 200:
        # A 4xx or a business error code is an explicit rejection and no remote
        # task was created, so there is no duplicate billing risk; return an
        # empty result per the existing material source contract and continue.
        logger.error(
            "wavespeed video generation request rejected: "
            f"http_status={submit_status}, "
            f"code={submit_body.get('code') if isinstance(submit_body, dict) else None}, "
            f"detail={_redact_secret(str((submit_body or {}).get('message') or ''), api_key)}"
        )
        return []
    prediction_id = (
        str(submit_data.get("id") or "") if isinstance(submit_data, dict) else ""
    )
    if not prediction_id:
        # The submission was accepted but no ID came back: the task may exist
        # without being trackable, so no further orders may be placed.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed accepted the submission without returning a prediction id"
        )
    # A successful submission already has a remote billing side effect, so log
    # the task ID first; even if polling later fails, users can still recover the
    # output in the WaveSpeed console with that ID.
    logger.info(f"wavespeed prediction created: id={prediction_id}")

    result_data = _wait_for_wavespeed_prediction(
        prediction_id=prediction_id,
        headers=headers,
        api_key=api_key,
    )
    if result_data is None:
        return []

    try:
        video_items = []
        outputs = result_data.get("outputs")
        for output in outputs if isinstance(outputs, list) else []:
            # The output URL is a signed temporary download address and must be
            # kept intact (query parameters cannot be stripped), so it is not
            # written to source_info and is used only for the immediate download.
            if not isinstance(output, str) or not output.startswith(
                ("http://", "https://")
            ):
                continue
            item = MaterialInfo()
            item.provider = "wavespeed"
            item.url = output
            item.duration = duration
            item.source_info = {
                "provider": "wavespeed",
                "search_term": search_term,
                "asset_id": prediction_id,
                "rendition": {
                    "id": None,
                    "width": video_width,
                    "height": video_height,
                },
            }
            video_items.append(item)
        if not video_items:
            logger.error(
                "wavespeed prediction completed without downloadable outputs: "
                f"id={prediction_id}"
            )
        return video_items
    except Exception as e:
        # The output was generated and billed, so an exception here can only come
        # from local parsing. Log it and return an empty result so the caller
        # skips this clip; the task state itself is known, so later clips can
        # continue.
        logger.error(
            "wavespeed output parsing failed: "
            f"id={prediction_id}, error={type(e).__name__}, "
            f"detail={_redact_request_error(e, api_key)}"
        )

    return []


def _wait_for_wavespeed_prediction(
    *,
    prediction_id: str,
    headers: dict,
    api_key: str,
) -> dict | None:
    """
    Poll the same prediction id until a definite result appears.

    Return the data of a ``completed`` prediction; return None when the remote
    side explicitly failed (failed / cancelled / timeout), meaning the task has
    ended and later clips can safely continue. Transient failures retry the same
    ID with linear backoff and never resubmit the task; when the state can never
    be confirmed, raise :class:`WaveSpeedUnconfirmedTaskError` so the caller
    stops the whole generation flow.
    """
    deadline = time.monotonic() + WAVESPEED_RUN_TIMEOUT_SECONDS
    generation_progress.remote("wavespeed", prediction_id)
    consecutive_failures = 0
    while True:
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{WAVESPEED_API_BASE_URL}/predictions/{prediction_id}/result",
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            status_code = _wavespeed_status_code(response)
            if status_code in WAVESPEED_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            result_body = response.json()
            result_data = (
                result_body.get("data") if isinstance(result_body, dict) else None
            )
            if not isinstance(result_body, dict) or result_body.get("code") != 200:
                # When polling is explicitly rejected (e.g. 4xx) the task state
                # is still unknown: the task was submitted, only the local lookup
                # failed, so no new paid tasks may be submitted either.
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction status is unknown: "
                    f"http_status={status_code}, "
                    f"code={result_body.get('code') if isinstance(result_body, dict) else None}, "
                    f"detail={_redact_secret(str((result_body or {}).get('message') or ''), api_key)}",
                    prediction_id=prediction_id,
                )
            if not isinstance(result_data, dict):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction result payload is malformed",
                    prediction_id=prediction_id,
                )
        except WaveSpeedUnconfirmedTaskError:
            raise
        except Exception as e:
            if not _is_wavespeed_retryable_error(e):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed and the task state is "
                    f"unknown: error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            consecutive_failures += 1
            generation_progress.remote("wavespeed", prediction_id, retrying=True)
            if consecutive_failures > WAVESPEED_MAX_POLL_RETRIES:
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed after "
                    f"{WAVESPEED_MAX_POLL_RETRIES + 1} attempts, the task may "
                    "still be running remotely: "
                    f"error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            delay = WAVESPEED_RETRY_BASE_SECONDS * consecutive_failures
            logger.warning(
                "wavespeed prediction polling hit a transient error, retry the "
                f"same task: id={prediction_id}, "
                f"attempt={consecutive_failures}/{WAVESPEED_MAX_POLL_RETRIES}, "
                f"error={type(e).__name__}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        # Reset the counter on any valid response; only consecutive failures
        # consume the retry budget.
        consecutive_failures = 0
        status = str(result_data.get("status") or "")
        generation_progress.remote("wavespeed", prediction_id, status)
        if status == "completed":
            return result_data
        if status in WAVESPEED_FAILURE_STATUSES:
            logger.error(
                "wavespeed prediction did not produce a video: "
                f"id={prediction_id}, status={status}, "
                f"detail={_redact_secret(str(result_data.get('error') or ''), api_key)}"
            )
            return None
        if time.monotonic() > deadline:
            # The remote task is still running and the final state cannot be
            # confirmed locally, so no further orders may be placed.
            raise WaveSpeedUnconfirmedTaskError(
                f"wavespeed prediction is still {status or 'pending'} after "
                f"{WAVESPEED_RUN_TIMEOUT_SECONDS:.0f}s of local waiting",
                prediction_id=prediction_id,
            )
        time.sleep(WAVESPEED_POLL_INTERVAL_SECONDS)


def _save_generated_video_with_retry(
    video_url: str, save_dir: str, provider: str
) -> str:
    """
    Download an already paid-for output, retrying the same URL first on failure.

    Regenerating a remote task costs another payment, so download hiccups must
    first be retried a limited number of times with backoff on the original URL;
    once retries are exhausted, the caller reports a recoverable paid-task
    failure.
    """
    for attempt in range(WAVESPEED_MAX_DOWNLOAD_RETRIES + 1):
        try:
            saved_video_path = save_video(video_url=video_url, save_dir=save_dir)
            if saved_video_path:
                return saved_video_path
            failure_detail = "empty result"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, video_url)}"
            )
        if attempt >= WAVESPEED_MAX_DOWNLOAD_RETRIES:
            break
        delay = WAVESPEED_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "failed to download generated video, retry the same url: "
            f"provider={provider}, "
            f"attempt={attempt + 1}/{WAVESPEED_MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay:.1f}s"
        )
        time.sleep(delay)
    logger.error(
        "failed to download generated video after "
        f"{WAVESPEED_MAX_DOWNLOAD_RETRIES + 1} attempts: "
        f"provider={provider}, {failure_detail}"
    )
    return ""


def _get_downloaded_video_duration(video_path: str) -> float:
    """Read the usable duration from the downloaded media file."""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0)
    except Exception as exc:
        raise ValueError(
            f"downloaded video duration could not be measured: {video_path}"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    "failed to close downloaded video after duration probe: "
                    f"path={video_path}, error={type(close_error).__name__}, "
                    f"detail={close_error}"
                )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"downloaded video duration is not positive and finite: {video_path}"
        )
    return duration


# requests produces no logs during a streaming download. On a slow network a
# single 4K material can take minutes, making the task look stuck. Log the
# downloaded size and speed when this interval passes without completion; small
# files that finish within the interval produce no extra logs, avoiding spam.
_DOWNLOAD_HEARTBEAT_SECONDS = 15.0
_BYTES_PER_MEGABYTE = 1024 * 1024


def _describe_download_progress(
    downloaded_bytes: int, declared_size: int, elapsed_seconds: float
) -> str:
    """Format download progress as log text; omit the percentage when the
    server did not declare a size."""
    downloaded_mb = downloaded_bytes / _BYTES_PER_MEGABYTE
    speed = downloaded_mb / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if declared_size > 0:
        percent = min(100, round(downloaded_bytes * 100 / declared_size))
        return (
            f"{downloaded_mb:.1f} of {declared_size / _BYTES_PER_MEGABYTE:.1f} MB "
            f"({percent}%), {speed:.2f} MB/s"
        )
    return f"{downloaded_mb:.1f} MB, {speed:.2f} MB/s"


def _is_valid_video_file(video_path: str) -> bool:
    """Return whether a video satisfies the cache publication contract."""
    clip = None
    try:
        # Decode the complete stream first. Unlike MoviePy's metadata probe,
        # this validator distinguishes confirmed bad media from an FFmpeg
        # execution failure through MaterialUploadError/MaterialServiceError.
        material_upload._validate_video(video_path)
    except material_upload.MaterialServiceError:
        raise
    except material_upload.MaterialUploadError as exc:
        logger.warning(f"invalid video file: {video_path} => {str(exc)}")
        return False
    except Exception as exc:
        raise material_upload.MaterialServiceError(
            "unexpected failure while validating video content"
        ) from exc

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        return bool(duration > 0 and fps > 0)
    except Exception as exc:
        # Complete decoding already succeeded. A MoviePy failure at this point
        # is an unavailable metadata probe (for example, its configured FFmpeg
        # executable is missing), not proof that the cached bytes are corrupt.
        raise material_upload.MaterialServiceError(
            "failed to probe video metadata"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    f"failed to close video clip: {video_path}, "
                    f"error: {str(close_error)}"
                )


def save_video(video_url: str, save_dir: str = "") -> str:
    generation_progress.downloading()
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    os.makedirs(save_dir, exist_ok=True)

    # Query parameters can identify the asset itself (for example,
    # /download?file_id=123). Dropping the query makes unrelated paid videos
    # share one cache entry and silently reuse the first scene. Fragments are
    # not sent in HTTP requests, so they do not affect the downloaded bytes.
    url_hash = utils.md5(video_url.split("#", 1)[0])
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # A nonempty path can still be a truncated file left by an older release,
    # a manual copy, or storage corruption. Validate it before declaring a hit
    # so the cache can repair itself instead of failing on every later render.
    try:
        cached_stat = os.stat(video_path)
    except FileNotFoundError:
        cached_stat = None
    if cached_stat is not None and cached_stat.st_size > 0:
        generation_progress.verifying()
        try:
            cached_video_is_valid = _is_valid_video_file(video_path)
        except material_upload.MaterialServiceError as exc:
            # Do not destroy the only local copy because the validation service
            # is temporarily unavailable. Reusing the existing nonempty cache
            # preserves the pre-validation behavior and avoids an unrecoverable
            # download attempt when a signed source URL has expired.
            logger.warning(
                f"video cache validation unavailable; preserving existing file: "
                f"{video_path} => {str(exc)}"
            )
            return video_path
        if cached_video_is_valid:
            logger.info(f"video already exists: {video_path}")
            return video_path

        logger.warning(f"discarding invalid cached video: {video_path}")
        try:
            current_stat = os.stat(video_path)
            cached_identity = (
                cached_stat.st_dev,
                cached_stat.st_ino,
                cached_stat.st_size,
                cached_stat.st_mtime_ns,
            )
            current_identity = (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_size,
                current_stat.st_mtime_ns,
            )
            # Do not delete a valid replacement published by another worker
            # while this process was validating the stale entry.
            if current_identity == cached_identity:
                os.remove(video_path)
        except FileNotFoundError:
            pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # A nonempty file is treated as a cache hit above. Keep the final path
    # unpublished until the complete download has passed media validation, so a
    # failed download cannot poison this or another concurrent task's cache.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{video_id}-",
            suffix=".mp4",
            dir=save_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            with trace_http_call(
                "GET", requests.get,
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
                stream=True,
            ) as response:
                response.raise_for_status()
                headers = getattr(response, "headers", {}) or {}
                try:
                    declared_size = int(headers.get("Content-Length", ""))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > MAX_VIDEO_DOWNLOAD_BYTES:
                    raise ValueError("video download exceeds 512 MB limit")

                downloaded_bytes = 0
                started_at = time.monotonic()
                last_report_at = started_at
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_VIDEO_DOWNLOAD_BYTES:
                            raise ValueError("video download exceeds 512 MB limit")
                        temp_file.write(chunk)
                        now = time.monotonic()
                        if now - last_report_at >= _DOWNLOAD_HEARTBEAT_SECONDS:
                            last_report_at = now
                            generation_progress.downloading(downloaded_bytes, declared_size)
                            # Log only the cache file name: the download URL may
                            # carry a signature or a key.
                            logger.info(
                                f"downloading video {video_id}.mp4: "
                                + _describe_download_progress(
                                    downloaded_bytes,
                                    declared_size,
                                    now - started_at,
                                )
                            )

        if os.path.getsize(temp_path) == 0:
            return ""

        generation_progress.verifying()
        if not _is_valid_video_file(temp_path):
            return ""

        os.replace(temp_path, video_path)
        temp_path = ""
        return video_path
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except OSError as remove_error:
                logger.warning(
                    f"failed to remove temporary video file: {temp_path}, "
                    f"error: {str(remove_error)}"
                )


# OpenAI-compatible text-to-image (Issue #1274) generates image material for
# script keywords via the /images/generations protocol; it can target a local
# ComfyUI/SD gateway or any OpenAI-protocol relay service. Generated images are
# immediately rendered into the same "slow zoom" mp4 clips as local material,
# fully transparent to the downstream editing pipeline.
OPENAI_IMAGE_ENDPOINT_PATH = "images/generations"
# The official OpenAI image API only accepts model-defined sizes and cannot take
# a video resolution directly (e.g. 1080x1920). When openai_image_size is empty,
# use the following compatible defaults per aspect; local gateways can override
# them explicitly in the config.
OPENAI_IMAGE_DEFAULT_SIZES = {
    VideoAspect.portrait: "1024x1536",
    VideoAspect.landscape: "1536x1024",
    VideoAspect.square: "1024x1024",
}
# Same retry policy as WaveSpeed: 429 and 5xx are transient failures and get a
# limited number of backoff retries.
OPENAI_IMAGE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 401/403 means the current key was explicitly rejected. get_api_key rotates keys
# on every call, so with multiple keys configured a retry automatically switches
# keys; with only one key, fail fast instead of retrying pointlessly.
OPENAI_IMAGE_KEY_ERROR_STATUS_CODES = frozenset({401, 403})
OPENAI_IMAGE_MAX_ATTEMPTS = 3
OPENAI_IMAGE_MAX_BYTES = 25 * 1024 * 1024
OPENAI_IMAGE_MAX_PIXELS = 50_000_000
# Serial image generation plus linear backoff, matching the rate-limit recovery
# window common to relay services.
OPENAI_IMAGE_RETRY_BACKOFF_SECONDS = (5, 15, 30)
# A synchronous generation endpoint may take tens of seconds to return an image,
# so allow a generous read timeout.
OPENAI_IMAGE_REQUEST_TIMEOUT = (30, 300)
# Download retries after an image has been billed per image: retry the original
# URL first instead of regenerating the same image.
OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS = 3
OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2


def is_openai_image_enabled(app_config: dict | None = None) -> bool:
    """
    Check whether the OpenAI-compatible image source has minimal configuration.

    The API key may be empty: a fully local ComfyUI/SD gateway usually needs no
    auth, and an empty key means requests carry no Authorization header. Used by
    task preflight and the WebUI to reject tasks with missing configuration
    before consuming LLM and TTS quota.
    """
    app_config = config.app if app_config is None else app_config
    return bool(
        str(app_config.get("openai_image_base_url", "") or "").strip()
        and str(app_config.get("openai_image_model", "") or "").strip()
    )


def _openai_image_endpoint() -> tuple[str, str]:
    """
    Read the text-to-image endpoint and model name, raising an error with
    configuration guidance when they are missing.
    """
    base_url = (
        str(config.app.get("openai_image_base_url", "") or "").strip().rstrip("/")
    )
    model = str(config.app.get("openai_image_model", "") or "").strip()
    if not base_url:
        raise ValueError(
            "\n\n##### openai_image_base_url is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    if not model:
        raise ValueError(
            "\n\n##### openai_image_model is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    return f"{base_url}/{OPENAI_IMAGE_ENDPOINT_PATH}", model


def _openai_image_size(video_aspect: VideoAspect) -> str:
    """
    Resolve the requested image size.

    The official OpenAI API only accepts model-defined sizes (e.g. 1024x1536);
    passing a video resolution (e.g. 1080x1920) returns 400. By default a
    compatible size is chosen per aspect; ``openai_image_size`` can override it
    explicitly for local gateways that support arbitrary resolutions (such as
    SD WebUI).
    """
    configured = str(config.app.get("openai_image_size", "") or "").strip()
    if configured:
        return configured
    return OPENAI_IMAGE_DEFAULT_SIZES.get(VideoAspect(video_aspect), "1024x1024")


def _openai_image_prompt(search_term: str) -> str:
    """
    Wrap a script keyword into the final prompt.

    The optional ``openai_image_prompt_template`` config supports a ``{term}``
    placeholder for uniformly appending style modifiers (such as quality,
    composition and camera language) to improve text-image matching:

    .. code-block:: toml

        openai_image_prompt_template = "cinematic photo of {term}, photorealistic"

    When empty or without the placeholder, fall back to the raw keyword, exactly
    as older versions behaved. If placeholder substitution fails (e.g. the
    template contains invalid format syntax), also fall back to the raw keyword
    so a configuration error does not abort the whole generation task.
    """
    template = str(config.app.get("openai_image_prompt_template", "") or "").strip()
    if not template or "{term}" not in template:
        return search_term
    try:
        return template.replace("{term}", search_term)
    except Exception:
        return search_term


def _response_json_safely(response: Any) -> Any:
    """Read the response JSON; return None when a test double or a malformed
    response fails to parse."""
    try:
        return response.json()
    except Exception:
        return None


def _openai_image_response_message(body: Any) -> str:
    """
    Extract a readable error description from an OpenAI-compatible response.

    The standard shape is ``{"error": {"message": ...}}``; relay services often
    degrade to ``{"message": ...}`` or a bare string. When none is available,
    return an empty string and let the caller decide whether to fall back to the
    response body.
    """
    if not isinstance(body, dict):
        return str(body or "")[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if error is not None:
        return str(error)[:300]
    return str(body.get("message") or "")[:300]


def _openai_image_http_failure(response: Any, status: int, api_key: str) -> str:
    """Turn an HTTP error response into a redacted, log-readable description."""
    message = _openai_image_response_message(_response_json_safely(response))
    if not message:
        message = str(getattr(response, "text", "") or "")[:300]
    return f"HTTP {status}: {_redact_secret(message, api_key)}"


def _openai_image_download_bytes(
    image_url: str,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Download the temporary URL of an already generated image.

    The image has already been billed per image, so on download failure retry
    the original URL first instead of falling back to regeneration, avoiding
    paying twice for the same image. When all retries fail, raise so the caller
    does not buy the next image.
    """
    failure_detail = "no download attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS + 1):
        response = None
        try:
            response = trace_http_call(
                "GET", requests.get,
                image_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                },
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code == 200:
                try:
                    declared_size = int(response.headers.get("Content-Length", 0))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > OPENAI_IMAGE_MAX_BYTES:
                    return None, "generated image exceeds the 25 MB download limit"
                image_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if len(image_bytes) + len(chunk) > OPENAI_IMAGE_MAX_BYTES:
                        return None, "generated image exceeds the 25 MB download limit"
                    image_bytes.extend(chunk)
                if image_bytes:
                    return bytes(image_bytes), ""
                failure_detail = "generated image download was empty"
            else:
                failure_detail = f"HTTP {response.status_code} while downloading image"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, api_key, image_url)}"
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception as exc:
                    logger.warning(
                        "failed to close generated image download response: "
                        f"error={type(exc).__name__}"
                    )
        if attempt < OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "generated image download failed, retrying the same url: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS}, "
                f"{failure_detail}"
            )
            time.sleep(OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS)
    raise OpenAIImagePaidResultError(
        "generated image could not be downloaded after retries"
    )


def _parse_openai_image_response(
    response: Any,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Parse the /images/generations response and retrieve url or b64_json data.

    A parse failure is either an explicit business rejection (e.g. content
    policy) or a malformed response format; return the error description
    directly without backoff retries -- resending the same request would only
    produce the same result.
    """
    body = _response_json_safely(response)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None, _redact_secret(_openai_image_response_message(body), api_key)

    entry = data[0]
    if not isinstance(entry, dict):
        return None, "invalid image data entry"

    b64_payload = entry.get("b64_json")
    if b64_payload:
        if (
            not isinstance(b64_payload, str)
            or len(b64_payload) > 4 * ((OPENAI_IMAGE_MAX_BYTES + 2) // 3)
        ):
            return None, "generated image exceeds the 25 MB response limit"
        try:
            image_bytes = base64.b64decode(b64_payload)
        except Exception as e:
            return None, f"invalid b64_json payload: {type(e).__name__}"
        if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
            return None, "generated image exceeds the 25 MB response limit"
        return image_bytes, ""

    image_url = entry.get("url")
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return _openai_image_download_bytes(image_url, api_key)

    return None, "image response has neither url nor b64_json"


def _request_openai_image(endpoint: str, payload: dict) -> tuple[bytes | None, str]:
    """
    Call the OpenAI-compatible /images/generations endpoint with backoff retries
    and key rotation.

    429/5xx are retried with backoff as transient failures; 401/403 are retried
    only when multiple keys are configured (using get_api_key's rotation to
    switch keys); other 4xx are explicit rejections, so fail fast and let the
    caller skip this keyword.

    Billing safety: a read timeout or connection drop on the POST is treated as
    "unconfirmed" -- the server may already have generated and billed the image
    without returning a response, and automatic resubmission could cause
    duplicate generation and duplicate billing, so a dedicated exception stops
    the local task. Only a connect-phase timeout (ConnectTimeout, where the
    request definitely never reached the server) confirms no generation task was
    created and can be safely retried.

    The API key may be empty: a fully local ComfyUI/SD gateway usually needs no
    auth, and an empty key means no Authorization header is sent.
    """
    api_keys = config.app.get("openai_image_api_keys")
    if isinstance(api_keys, (list, tuple)):
        configured_keys = [k for k in api_keys if str(k or "").strip()]
    elif str(api_keys or "").strip():
        configured_keys = [api_keys]
    else:
        configured_keys = []

    failure_detail = "no request attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_ATTEMPTS + 1):
        api_key = get_api_key("openai_image_api_keys") if configured_keys else ""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        retryable = False
        try:
            response = trace_http_call(
                "POST", requests.post,
                endpoint,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.exceptions.ConnectTimeout as e:
            # Connect-phase timeout: the request definitely never reached the
            # server and no generation task was created, so retrying is safe.
            failure_detail = (
                f"connect timeout: detail={_redact_request_error(e, api_key)}"
            )
            retryable = True
        except Exception as e:
            # Read timeouts and connection drops are "unconfirmed": the server
            # may already have accepted and billed the request, so automatic
            # resubmission or processing later keywords could double-bill; the
            # local task must stop.
            raise OpenAIImageUnconfirmedError(
                "unconfirmed image request (no retry to avoid double billing): "
                f"{type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            ) from e
        else:
            status = int(getattr(response, "status_code", 200) or 200)
            if 300 <= status < 400:
                response.close()
                raise OpenAIImageUnconfirmedError(
                    "image submission returned a redirect; no redirected request "
                    "or retry was sent because the paid outcome is unconfirmed"
                )
            if status in OPENAI_IMAGE_KEY_ERROR_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                # Only with multiple keys configured can a retry rotate to a
                # usable key.
                retryable = len(configured_keys) > 1
            elif status in OPENAI_IMAGE_RETRYABLE_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = True
            elif status >= 400:
                return None, _openai_image_http_failure(response, status, api_key)
            else:
                image_bytes, parse_error = _parse_openai_image_response(
                    response, api_key
                )
                if image_bytes is not None:
                    return image_bytes, ""
                failure_detail = parse_error

        if retryable and attempt < OPENAI_IMAGE_MAX_ATTEMPTS:
            backoff_seconds = OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(OPENAI_IMAGE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "openai image request failed, retrying: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_ATTEMPTS}, "
                f"next_retry_in={backoff_seconds}s, detail={failure_detail}"
            )
            time.sleep(backoff_seconds)
            continue
        return None, failure_detail

    return None, failure_detail


def _save_openai_image_file(
    image_bytes: bytes,
    save_dir: str,
) -> tuple[str, int, int]:
    """
    Normalize the generated result to PNG on disk, returning (path, width, height).

    Converting everything to PNG avoids two classes of problems: relay services
    returning WebP/JPEG without a reliable extension, and images with unusual
    metadata breaking MoviePy parsing (mirroring the sanitization logic for local
    material, normalization happens here at the write stage).
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, f"openai-image-{uuid.uuid4().hex[:12]}.png")
    if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
        raise _OpenAIImageDecodeError("generated image exceeds the 25 MB limit")

    # An image decode failure can degrade to "skip this keyword", but directory
    # permission, disk space and file write failures must keep propagating;
    # otherwise the on-demand generation loop would keep creating paid tasks
    # while local files cannot be saved. Image.open only reads in-memory bytes,
    # so an OSError here means format detection failed; an OSError from
    # image.load corresponds to truncated or corrupt image data.
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > OPENAI_IMAGE_MAX_PIXELS:
                raise ValueError("generated image exceeds the 50 million pixel limit")
            image.load()
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        if image is not None:
            image.close()
        raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc

    with image:
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        # save is not inside the decode exception guard: a write error means the
        # runtime environment is persistently unavailable and the whole task
        # should stop immediately, so later keywords do not keep producing paid
        # images that cannot be written to disk.
        image.save(image_path, format="PNG")
        width, height = image.size
    return image_path, width, height


def generate_images_openai(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Generate one image for a script keyword via the OpenAI-compatible
    text-to-image API and save it locally.

    Keeps the same signature and empty-list failure contract as
    generate_videos_wavespeed. Images have no native duration, so ``duration``
    records the target clip duration (seconds) for the on-demand download flow to
    check whether the voiceover duration is covered. The actual size returned by
    the API may differ from the request, so the real image dimensions are written
    to rendition rather than relying on request parameters.
    """
    aspect = VideoAspect(video_aspect)
    clip_duration = max(int(minimum_duration), 1)
    endpoint, model = _openai_image_endpoint()
    image_size = _openai_image_size(aspect)
    payload = {
        "model": model,
        "prompt": _openai_image_prompt(search_term),
        "n": 1,
        "size": image_size,
    }
    logger.info(
        f"generating image via openai-compatible endpoint: model={model}, "
        f"term={search_term!r}, size={image_size}"
    )
    generation_progress.submitting("openai_image")
    image_bytes, failure_detail = _request_openai_image(endpoint, payload)
    if image_bytes is None:
        logger.error(
            f"openai image generation failed: term={search_term!r}, "
            f"detail={failure_detail}"
        )
        return []

    try:
        image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
    except _OpenAIImageDecodeError as e:
        # The compatibility layer may return 200 with a non-image body (e.g. an
        # HTML error page disguised as JSON, or a gateway fallback page). An
        # undecodable image means "this generation failed", so return an empty
        # list per the material source contract and let the caller skip this
        # keyword instead of aborting the whole task with an exception.
        logger.error(
            "openai image response is not a decodable image, skipping term: "
            f"term={search_term!r}, error={type(e).__name__}, detail={e}"
        )
        return []
    item = MaterialInfo()
    item.provider = "openai_image"
    item.url = image_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "openai_image",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]


def generate_images_codex(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
    *,
    context=None,
) -> List[MaterialInfo]:
    """Obtain one Codex image using the existing PNG material representation."""
    aspect = VideoAspect(video_aspect)
    if minimum_duration <= 0:
        return []
    try:
        context = context if context is not None else codex_image.capture_context()
        codex_image.assert_current(context)
        # Independent of the OpenAI-compatible provider's template and settings.
        prompt = (
            f"Create one image depicting: {search_term}. "
            f"Compose the image for a {aspect.value} video frame."
        )
        generation_progress.submitting("codex_image")
        image_bytes = codex_image.generate_image(prompt, context)
        codex_image.assert_current(context)
        try:
            image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
        except _OpenAIImageDecodeError:
            raise codex_image.CodexImageError("codex.image.invalid_result") from None
        except OSError:
            raise codex_image.CodexImageError("codex.image.save_failed") from None
        codex_image.assert_current(context)
    except codex_image.CodexImageError as error:
        raise CodexImageMaterialError(error) from None
    item = MaterialInfo()
    item.provider = "codex_image"
    item.url = image_path
    item.duration = max(int(minimum_duration), 1)
    item.source_info = {
        "provider": "codex_image",
        "search_term": search_term,
        "rendition": {"id": None, "width": width, "height": height},
    }
    return [item]


def _render_openai_image_video(
    image_path: str, clip_duration: int, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Render a generated image into an mp4 clip, reusing the local material
    "image -> motion clip" pipeline.

    Return an empty string on render failure so the caller stops further paid
    requests.
    """
    try:
        return video.render_image_zoom_video(
            image_path, clip_duration, image_motion=image_motion
        )
    except Exception as e:
        logger.error(
            "failed to render generated image as a video clip: "
            f"image={image_path}, error={type(e).__name__}, detail={e}"
        )
        return ""


def _download_videos_openai_image_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    image_motion: ImageMotion = ImageMotion.zoom,
    image_generator: Callable[..., List[MaterialInfo]] | None = None,
    provider: str = "openai_image",
    codex_image_context=None,
) -> List[str]:
    """
    Generate OpenAI-compatible image material one by one in script clip order,
    stopping as soon as the required total duration is covered.

    Same billing-safety semantics as WaveSpeed on-demand generation: text-to-image
    is billed per image, and generating everything up front before selecting
    would pay for frames that are never used. Each generated image is immediately
    rendered into an mp4 clip and its effective duration accumulated (capped at
    the clip duration, as in the stock flow); once the required voiceover duration
    is reached, no new paid requests are made. A single explicitly rejected image
    may be skipped; an unknown request outcome, a failed download of a paid
    result, or a local render failure must stop the task so later keywords are not
    billed again.
    """
    # Validate even when the material service is called directly, before any paid
    # request, so an invalid effect parameter is not discovered after generation.
    image_motion = ImageMotion(image_motion)
    if not material_directory:
        # Generated images are billed per task and cannot be reused, so default
        # to the task directory for traceability.
        material_directory = utils.task_dir(task_id)

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0

    # A non-positive voiceover duration makes the "accumulated required duration"
    # check meaningless, so return empty-handed and avoid paying per image for a
    # task that can never be satisfied (matching the Seedance preflight semantics).
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError):
        required_duration = 0.0
    if required_duration <= 0:
        logger.warning(
            "skip openai image generation because required audio duration is "
            f"not positive: duration={audio_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    if image_generator is None:
        image_generator = generate_images_openai
    generation_options = {}
    if provider == "codex_image":
        try:
            codex_image_context = (
                codex_image_context
                if codex_image_context is not None
                else codex_image.capture_context()
            )
            codex_image.assert_current(codex_image_context)
        except codex_image.CodexImageError as error:
            raise CodexImageMaterialError(error) from None
        generation_options["context"] = codex_image_context

    for search_term in search_terms:
        try:
            items = image_generator(
                search_term=search_term,
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
                save_dir=material_directory,
                **generation_options,
            )
        except OpenAIImagePaidResultError:
            _persist_material_sources(task_id, material_sources)
            raise
        for item in items:
            video_file = _render_openai_image_video(
                item.url, max_clip_duration, image_motion=image_motion
            )
            if not video_file:
                _persist_material_sources(task_id, material_sources)
                if provider == "codex_image":
                    raise CodexImageMaterialError(
                        codex_image.CodexImageError("codex.image.render_failed")
                    )
                raise OpenAIImagePaidResultError(
                    "generated image could not be rendered locally"
                )
            if provider == "codex_image":
                try:
                    codex_image.assert_current(codex_image_context)
                except codex_image.CodexImageError as error:
                    _persist_material_sources(task_id, material_sources)
                    raise CodexImageMaterialError(error) from None
            logger.info(f"image material rendered: {video_file}")
            video_paths.append(video_file)
            try:
                material_sources.append(_material_source_record(item, video_file))
            except Exception as source_error:
                # As with stock sources: a source-record error must not treat an
                # already paid-for and successfully rendered material as a
                # failure, let alone block video generation.
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider={provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            # Use >= as with WaveSpeed: generating one more image after exactly
            # covering the duration would pay an extra fee. The inner and outer
            # checks must keep the same semantics.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated image materials cover the required duration, stop "
                f"generating more images: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and rendered {len(video_paths)} image materials")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Shared 24-hour search cache for the three online material sources.

    The cache wraps only the search API and does not change the downstream video
    download and deduplication logic. An empty list from the remote side is not
    cached, because the existing provider interface uses an empty list for both
    "no results" and "request failed"; until the two are split into explicit
    result types, retrying next time is preferable to caching a transient
    failure for a day.
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # The cache is an optional optimization; any cache implementation
            # error must be treated as a miss and must not block normal remote
            # searches for Pexels, Pixabay or Coverr.
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # Caches from older versions may contain material of other
            # orientations. Even if a few usable entries remain, refresh the full
            # candidate set; otherwise the same small batch of videos would be
            # reused for the whole cache lifetime.
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # Re-read after waiting for threads with the same search criteria, so
        # multiple API tasks do not hit the remote side simultaneously on a first
        # cache miss, reducing third-party rate limiting and risk-control triggers.
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Providers normally write the current keyword, but test doubles,
        # third-party extensions or old implementations may omit it or carry a
        # wrong value. Cache reads restore this field from the cache key, so
        # remote results are corrected at the same entry point, keeping task
        # source records consistent between first searches and cache hits.
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_terms_in_parallel(
    search_terms: List[str],
    search_videos: Callable[[str, int, VideoAspect], List[MaterialInfo]],
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[tuple[str, List[MaterialInfo]]]:
    """Search keywords in parallel; a failed keyword yields an empty result and
    does not block other material."""
    if not search_terms:
        return []

    workers = min(_get_material_concurrency(), len(search_terms))
    if workers == 1:
        results = []
        for search_term in search_terms:
            try:
                items = search_videos(
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-search",
    ) as executor:
        for search_term in search_terms:
            futures[executor.submit(
                copy_context().run,
                logging_utils.bind_log_scope(search_videos),
                search_term,
                minimum_duration,
                video_aspect,
            )] = search_term

        results = []
        for future in futures:
            search_term = futures[future]
            try:
                items = future.result()
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results


def _report_material_downloaded(
    position: int,
    total: int,
    item: MaterialInfo,
    saved_video_path: str,
    on_downloaded: Callable[[MaterialInfo], None] | None,
) -> None:
    """
    Record that one material finished downloading and notify the caller to update
    progress.

    Always called on the thread that started the download, so per-file logging
    and progress updates do not depend on the download concurrency.
    """
    description = os.path.basename(saved_video_path)
    try:
        size_mb = os.path.getsize(saved_video_path) / _BYTES_PER_MEGABYTE
        description += f" ({size_mb:.1f} MB)"
    except OSError:
        # File size is auxiliary information; failing to read it does not affect
        # the download result.
        pass
    logger.info(f"downloaded material {position}/{total}: {description}")
    if on_downloaded is None:
        return
    try:
        on_downloaded(item)
    except Exception as exc:
        # Progress is display-only. A callback failure (e.g. the status backend
        # is temporarily unavailable) must not invalidate an already downloaded
        # material, let alone interrupt the whole task.
        logger.warning(
            "failed to report material download progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _covered_duration_reporter(
    progress_callback: Callable[[float], None] | None,
    required_duration: float,
    max_clip_duration: int,
) -> Callable[[MaterialInfo], None] | None:
    """
    Build a per-material callback that converts "covered voiceover duration" into
    a 0-1 download progress value.

    The measure matches the download loop's stop condition: each material
    contributes at most one clip duration, and covering the required duration
    completes the work, so any excess is capped at 1.0. Return None when there is
    no progress callback.
    """
    if progress_callback is None:
        return None
    covered_duration = 0.0

    def report(item: MaterialInfo) -> None:
        nonlocal covered_duration
        covered_duration += min(max_clip_duration, item.duration)
        fraction = 1.0
        if required_duration > 0:
            fraction = min(1.0, covered_duration / required_duration)
        progress_callback(fraction)

    return report


def _download_materials_in_parallel(
    materials: List[tuple[str, MaterialInfo]],
    material_directory: str,
    on_downloaded: Callable[[MaterialInfo], None] | None = None,
) -> list[tuple[str, MaterialInfo, str]]:
    """
    Download one round of materials in parallel, preserving the caller's
    candidate order.

    After each successful download, log one line and call ``on_downloaded`` so
    the caller can update task progress. Both the callback and the logging happen
    on the thread that calls this function.
    """
    if not materials:
        return []

    total = len(materials)
    workers = min(_get_material_concurrency(), total)
    if workers == 1:
        downloaded = []
        for position, (search_term, item) in enumerate(materials, start=1):
            try:
                saved_video_path = save_video(
                    video_url=item.url,
                    save_dir=material_directory,
                )
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-download",
    ) as executor:
        for search_term, item in materials:
            futures[executor.submit(
                copy_context().run,
                logging_utils.bind_log_scope(save_video),
                item.url,
                material_directory,
            )] = (search_term, item)

        downloaded = []
        for position, future in enumerate(futures, start=1):
            search_term, item = futures[future]
            try:
                saved_video_path = future.result()
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded


def _select_materials_until_duration(
    materials: List[tuple[str, MaterialInfo]],
    max_clip_duration: int,
    audio_duration: float,
    current_duration: float = 0.0,
) -> List[tuple[str, MaterialInfo]]:
    """
    Select one round of materials in candidate order, avoiding downloads of
    candidates beyond the voiceover duration when running concurrently.

    Return a prefix subset of ``materials``: accumulate from the first candidate
    and stop when the accumulated duration first exceeds the voiceover duration
    (including the one that exceeds it). The caller uses the prefix property to
    determine which candidates were actually attempted this round.
    """
    selected = []
    total_duration = current_duration
    for material in materials:
        selected.append(material)
        total_duration += min(max_clip_duration, material[1].duration)
        if total_duration > audio_duration:
            break
    return selected


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    progress_callback: Callable[[float], None] | None = None,
    image_motion: ImageMotion = ImageMotion.zoom,
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context=None,
) -> List[str]:
    """
    Search and download the material needed to cover the voiceover duration,
    returning local file paths.

    ``progress_callback`` is optional: for stock material (Pexels, Pixabay,
    Coverr) it is called once per finished file with a 0-1 completion ratio so
    the task layer can advance the progress bar. On-demand paid generation
    sources do not report download progress.
    """
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if source == "infistar":
        return _download_videos_infistar_on_demand(
            task_id=task_id, search_terms=search_terms, video_aspect=video_aspect,
            audio_duration=audio_duration, max_clip_duration=max_clip_duration,
        )
    if source == "wavespeed":
        # AI generation is billed per clip, so the stock flow of "fetch candidates
        # for all keywords first, then select" cannot be reused; it would pay for
        # unused clips. Generation sources instead produce clips on demand and
        # stop as soon as the required duration is covered. They also skip the
        # 24-hour search cache: output URLs are signed and expire, and reusing the
        # cache would give different tasks the same generated video repeatedly.
        return _download_videos_wavespeed_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source in {"volcengine_seedance", "jalapeno"}:
        # As with WaveSpeed, the official Ark API creates asynchronous paid tasks.
        # Generate clip by clip on demand and buy only the material the current
        # voiceover duration actually needs.
        return _download_videos_seedance_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            source=source,
            endpoint_snapshot=jalapeno_video_endpoint,
        )
    if source == "ofox":
        # Same on-demand billing semantics as WaveSpeed/Ark: the OFox gateway's
        # /v1/videos creates asynchronous paid tasks, so generate clip by clip and
        # stop as soon as the required duration is covered. Output addresses are
        # temporary direct links that expire, so they also skip the 24-hour search
        # cache.
        return _download_videos_ofox_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "muapi":
        # MuAPI model endpoints create asynchronous paid tasks. Submit clip by
        # clip and stop once the voiceover duration is covered, so unused script
        # keywords are not sent to the remote side.
        return _download_videos_muapi_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "metaso_minimax":
        # Metaso MiniMax is also billed per remote asynchronous task. Its request
        # body resembles Volcengine Ark, but the task query path and response
        # structure differ, so only the local on-demand generation semantics are
        # shared; the provider client is not reused, keeping protocol differences
        # out of the material orchestration layer.
        return _download_videos_metaso_minimax_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source in {"openai_image", "codex_image"}:
        # Same on-demand billing semantics as WaveSpeed: text-to-image is billed
        # per image, so generate clip by clip and stop as soon as the required
        # duration is covered. Results are one-off local image files and also skip
        # the 24-hour search cache, which would give different tasks the same
        # image repeatedly.
        image_options = {}
        if source == "codex_image":
            image_options = {
                "image_generator": generate_images_codex,
                "provider": source,
                "codex_image_context": codex_image_context,
            }
        return _download_videos_openai_image_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            image_motion=image_motion,
            **image_options,
        )

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            progress_callback=progress_callback,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    pending_items = list(valid_video_items)
    # The default/random path is also governed by the material concurrency
    # config: preselect each round's candidates using the serial stop condition
    # (accumulate in order, stop at the first excess over the voiceover duration).
    # When all downloads succeed, each round's set matches the serial logic
    # exactly; failed candidates are made up by later candidates in the next round.
    while pending_items and total_duration <= audio_duration:
        batch = []
        projected_duration = total_duration
        for item in pending_items:
            batch.append(item)
            projected_duration += min(max_clip_duration, item.duration)
            if projected_duration > audio_duration:
                break
        for item in batch:
            source_info = item.source_info if isinstance(item.source_info, dict) else {}
            logger.info(
                f"downloading {item.provider} video: "
                f"asset_id={source_info.get('asset_id') or 'unknown'}"
            )
        downloaded_materials = _download_materials_in_parallel(
            materials=[("", item) for item in batch],
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        pending_items = pending_items[len(batch):]
        for _, item, saved_video_path in downloaded_materials:
            try:
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        # A source-record error must not treat a successfully
                        # downloaded material as a download failure, let alone
                        # block video generation; keep the provider and exception
                        # type for later diagnosis.
                        logger.warning(
                            "failed to prepare material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, detail={source_error}"
                        )
                    seconds = min(max_clip_duration, item.duration)
                    total_duration += seconds
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_wavespeed_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """
    Generate WaveSpeed material clip by clip in script clip order, stopping as
    soon as the required total duration is covered.

    Each keyword naturally maps to one script clip, and generation is billed
    immediately: generating everything first and selecting later would pay for
    unused clips. Here each generated clip is downloaded right away and its
    effective duration accumulated (capped at the clip duration, as in the stock
    flow); once the required voiceover duration is covered, no new generation
    requests are triggered. Clips the remote side explicitly failed can be
    skipped; a paid task with an unknown state or a failed download must keep its
    ID and stop the local task.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed audio duration must be finite") from exc
    if not math.isfinite(required_duration):
        raise ValueError("WaveSpeed audio duration must be finite")
    if required_duration <= 0:
        logger.warning(
            "skip WaveSpeed paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ValueError("WaveSpeed clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = generate_videos_wavespeed(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except WaveSpeedUnconfirmedTaskError as e:
            # The submitted paid task has an unknown state: the remote side may
            # still be running or may have completed and billed. Ordering more for
            # later keywords would cause duplicate generation and duplicate
            # billing, so stop here and hand the prediction id to the task layer
            # for the failure state so users can recover the output.
            logger.error(
                "stop submitting new wavespeed tasks, the last submitted task "
                f"is unconfirmed: prediction_id={e.prediction_id or 'unknown'}, "
                f"detail={e}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "wavespeed"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                prediction_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise WaveSpeedDownloadError(
                    "WaveSpeed generated a paid video but the result could not be "
                    f"downloaded: id={prediction_id or 'unknown'}",
                    prediction_id=prediction_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                # As with stock sources: a source-record error must not treat an
                # already paid-for and successfully downloaded material as a
                # failure, let alone block video generation.
                logger.warning(
                    "failed to prepare material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            # Use >=: when the accumulated duration exactly equals the required
            # duration it is already enough, and generating more would pay an
            # extra fee. The inner and outer checks must keep the same semantics.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated materials cover the required duration, stop "
                f"generating more clips: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_seedance_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    source: str = "volcengine_seedance",
    endpoint_snapshot: volcengine_seedance.SeedanceEndpoint | None = None,
) -> List[str]:
    """Generate Seedance material sequentially, stopping paid orders as soon as
    the voiceover duration is covered."""
    service = jalapeno_video if source == "jalapeno" else volcengine_seedance
    generation_options = (
        {"endpoint_snapshot": endpoint_snapshot}
        if source == "jalapeno" and endpoint_snapshot is not None else {}
    )
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # The paid generation loop must validate the two durations that control the
    # loop count first. NaN/Infinity makes ``total_duration >= audio_duration``
    # never true, and a non-positive clip duration prevents the accumulated value
    # from growing; either could create useless paid tasks for every keyword.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Seedance paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = service.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
                **generation_options,
            )
        except volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError as exc:
            # The remote paid task may still succeed. Stop ordering immediately
            # and keep the task ID so users can later confirm or recover the
            # result in the Ark console.
            logger.error(
                "stop submitting new Seedance tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            logger.error(f"Seedance generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, source
            )
            if not saved_video_path:
                # The remote task completed and incurred a charge, so a local
                # download failure must carry the remote task ID back to the task
                # state so users can recover the result in the Ark console. Raise
                # the dedicated error here, which also stops later keywords from
                # creating new paid tasks.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                    "Seedance generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            if source == "jalapeno":
                # Both are 480p, but different gateways may actually return 854 or
                # 864 width. Measure dimensions and duration from the downloaded
                # video so source records stay accurate and coverage is not
                # overestimated.
                try:
                    with VideoFileClip(saved_video_path) as downloaded:
                        actual_duration = float(downloaded.duration)
                        width, height = downloaded.size
                    if not math.isfinite(actual_duration) or actual_duration <= 0:
                        raise ValueError("invalid generated video duration")
                except Exception as exc:
                    remote_task_id = str((item.source_info or {}).get("asset_id") or "")
                    _persist_material_sources(task_id, material_sources)
                    raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                        "Jalapeno Cloud generated a paid video but its local media could not be read",
                        task_id=remote_task_id,
                    ) from exc
                item.duration = actual_duration
                item.source_info = dict(item.source_info or {})
                item.source_info["rendition"] = {
                    **item.source_info.get("rendition", {}),
                    "width": width, "height": height,
                }
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider={source}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Seedance materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(
        f"generated and downloaded {len(video_paths)} {source} videos"
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_ofox_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Generate OFox material sequentially, stopping paid orders as soon as the
    voiceover duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # The paid generation loop must validate the two durations that control the
    # loop count first. NaN/Infinity makes ``total_duration >= audio_duration``
    # never true, and a non-positive clip duration prevents the accumulated value
    # from growing; either could create useless paid tasks for every keyword.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise ofox.OFoxError("OFox audio duration must be a finite number") from exc
    if not math.isfinite(required_duration):
        raise ofox.OFoxError("OFox audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip OFox paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ofox.OFoxError("OFox clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ofox.OFoxError("OFox clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = ofox.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except ofox.OFoxUnconfirmedTaskError as exc:
            # The remote paid task may still succeed. Stop ordering immediately
            # and keep the task ID so users can later confirm or recover the
            # result in the OFox console.
            logger.error(
                "stop submitting new OFox tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except ofox.OFoxError as exc:
            logger.error(f"OFox generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        # A single keyword explicitly failed by the remote side (e.g. content
        # moderation) yields an empty list: the task has ended with no billing
        # ambiguity, so skip this clip and continue with later keywords.
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "ofox"
            )
            if not saved_video_path:
                # The remote task completed and incurred a charge, so a local
                # download failure must carry the remote task ID back to the task
                # state so users can recover the result in the OFox console. Raise
                # the dedicated error here, which also stops later keywords from
                # creating new paid tasks.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise ofox.OFoxDownloadError(
                    "OFox generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=ofox, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated OFox materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} OFox videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_muapi_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Generate MuAPI materials until the narration duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise muapi.MuAPIError(
            "MuAPI audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise muapi.MuAPIError("MuAPI audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip MuAPI paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise muapi.MuAPIError(
            "MuAPI clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise muapi.MuAPIError("MuAPI clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = muapi.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except muapi.MuAPIUnconfirmedTaskError as exc:
            # An ambiguous paid submission must stop the loop: retrying with the
            # next keyword could create a duplicate charge.  The task ID is kept
            # for task-service recovery when the provider returned one.
            logger.error(
                "stop submitting new MuAPI tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except muapi.MuAPIError as exc:
            logger.error(f"MuAPI generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "muapi"
            )
            if not saved_video_path:
                # The remote job completed and may already have incurred a
                # charge.  Do not submit a replacement task after download
                # failure; return the remote ID for manual recovery instead.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=muapi, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            try:
                downloaded_duration = _get_downloaded_video_duration(saved_video_path)
            except Exception as duration_error:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but its downloaded duration "
                    "could not be measured: "
                    f"id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                ) from duration_error
            total_duration += min(clip_duration, downloaded_duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated MuAPI materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} MuAPI videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_metaso_minimax_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Generate Metaso MiniMax material sequentially, stopping paid orders as
    soon as the voiceover duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # The remote side generates at least 4 seconds, but locally we still trim and
    # accumulate by the user's clip duration. Validate the loop control parameters
    # up front so NaN, Infinity or non-positive values cannot make the stop
    # condition unsatisfiable and submit every keyword as a paid task.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Metaso MiniMax paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = metaso_minimax.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except metaso_minimax.MetasoMiniMaxUnconfirmedTaskError as exc:
            # When the request or polling state is unknown, the remote task may
            # still succeed and be billed. Stop the whole generation loop
            # immediately to prevent further orders for later keywords, and hand
            # the task ID to the task service for storage.
            logger.error(
                "stop submitting new Metaso MiniMax tasks because the last paid "
                f"task is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except metaso_minimax.MetasoMiniMaxError as exc:
            logger.error(f"Metaso MiniMax generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "metaso_minimax"
            )
            if not saved_video_path:
                # Successful generation already incurred a charge, so a download
                # failure must not create a replacement task. Raise the dedicated
                # error carrying the remote ID for task status and manual recovery.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise metaso_minimax.MetasoMiniMaxDownloadError(
                    "Metaso MiniMax generated a paid video but the result could "
                    f"not be downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=metaso_minimax, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )

            # The local video only uses the user-selected clip length; even if H3
            # generated longer material due to its minimum-duration constraint,
            # the unused part must not count toward coverage and cause necessary
            # frames to be under-generated.
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Metaso MiniMax materials cover the required duration; "
                f"stop submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} Metaso MiniMax videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    progress_callback: Callable[[float], None] | None = None,
) -> List[str]:
    """
    Download material in script text order.

    The default download logic merges candidates from all keywords into one big
    list; if the first keyword returns many results, the final download may keep
    consuming that keyword's material and later script topics never make it onto
    the timeline. Here candidates are grouped by keyword and downloaded
    round-robin: round 1 takes each keyword's first candidate, round 2 takes each
    keyword's second candidate. This keeps material order close to the script
    order without rewriting the video composition engine.
    """
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    # Advance the candidate index independently per keyword: only candidates
    # actually selected for download this round advance, while unselected ones
    # stay for the next round and are not skipped by a uniform round index.
    next_candidate_indices = [0] * len(candidate_groups)
    while candidate_groups and total_duration <= audio_duration:
        round_materials = [
            (
                group_position,
                search_term,
                term_items[next_candidate_indices[group_position]],
            )
            for group_position, (search_term, term_items) in enumerate(
                candidate_groups
            )
            if next_candidate_indices[group_position] < len(term_items)
        ]
        if not round_materials:
            break

        selected_materials = _select_materials_until_duration(
            materials=[
                (search_term, item) for _, search_term, item in round_materials
            ],
            max_clip_duration=max_clip_duration,
            audio_duration=audio_duration,
            current_duration=total_duration,
        )
        # _select_materials_until_duration returns a prefix subset of the
        # candidate list, so the first len(selected_materials) entries are the
        # candidates actually attempted this round; advance only the indices of
        # the groups those candidates belong to.
        for group_position, _, _ in round_materials[: len(selected_materials)]:
            next_candidate_indices[group_position] += 1

        downloaded_materials = _download_materials_in_parallel(
            materials=selected_materials,
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        for search_term, item, saved_video_path in downloaded_materials:
            try:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                logger.info(
                    f"downloading ordered {item.provider} video for {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare ordered material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download ordered material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

    logger.success(f"downloaded {len(video_paths)} ordered videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )


def _download_videos_infistar_on_demand(
    *, task_id, search_terms, video_aspect, audio_duration, max_clip_duration,
):
    """Order clip by clip; stop once material is sufficient, and do not buy the
    next clip after an unknown state or a download failure."""
    try:
        required = float(audio_duration)
        clip_duration = int(max_clip_duration)
    except (ValueError, TypeError, OverflowError) as exc:
        raise infistar_video.InfistarVideoError("Infistar duration must be finite and positive") from exc
    if not math.isfinite(required) or clip_duration <= 0:
        raise infistar_video.InfistarVideoError("Infistar duration must be finite and positive")
    paths = []
    sources = []
    covered = 0.0
    try:
        for term in search_terms:
            if covered >= required:
                break
            item = infistar_video.generate_video(
                search_term=term, video_aspect=video_aspect,
                output_dir=utils.task_dir(task_id),
            )
            paths.append(item.url)
            sources.append(_material_source_record(item, item.url))
            covered += min(clip_duration, item.duration)
    finally:
        _persist_material_sources(task_id, sources)
    return paths
