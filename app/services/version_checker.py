"""Check whether a newer stable release of MoneyPrinterTurbo is available."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from packaging.version import InvalidVersion, Version


LATEST_RELEASE_API_URL: Final = (
    "https://api.github.com/repos/harry0703/MoneyPrinterTurbo/releases/latest"
)
LATEST_RELEASE_PAGE_URL: Final = (
    "https://github.com/harry0703/MoneyPrinterTurbo/releases/latest"
)
# Update checks are auxiliary: network problems must not noticeably slow down the
# local WebUI. Connect and read timeouts are bounded separately, letting GitHub
# respond on ordinary networks while avoiding long waits in offline environments.
RELEASE_CHECK_TIMEOUT: Final = (1.0, 2.0)
RELEASE_CHECK_HEADERS: Final = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MoneyPrinterTurbo-Version-Checker",
}
UPDATE_CHECK_CACHE_TTL_SECONDS: Final = 12 * 60 * 60


def _parse_version(value: str) -> Version:
    """Accept GitHub's common ``v1.2.3`` tags and convert them into comparable versions."""
    normalized = str(value or "").strip()
    if normalized.lower().startswith("v"):
        normalized = normalized[1:]
    return Version(normalized)


def get_available_update(current_version: str) -> str | None:
    """
    Return the latest stable release if it is newer than the current version;
    return ``None`` when up to date or when the check fails.

    GitHub's ``releases/latest`` endpoint already excludes drafts and pre-releases,
    so release status filtering is not reimplemented here. The WebUI calls this
    function in the background through ``AsyncUpdateChecker``; network, response
    format or version tag errors are only logged and degrade to "show no
    notification", leaving core features such as video generation untouched.
    """
    try:
        installed_version = _parse_version(current_version)
    except InvalidVersion:
        logger.warning(
            f"skip update check because current version is invalid: {current_version!r}"
        )
        return None

    try:
        response = trace_http_call(
            "GET", requests.get,
            LATEST_RELEASE_API_URL,
            headers=RELEASE_CHECK_HEADERS,
            timeout=RELEASE_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # A failed update check is a recoverable, non-core exception. Keep the
        # exception type and message so proxy, DNS, GitHub rate-limit or corrupted
        # response issues can be diagnosed, without bothering regular WebUI users.
        logger.debug(
            "GitHub release check failed: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None

    if not isinstance(payload, dict):
        logger.debug(
            "GitHub release check returned an invalid payload: "
            f"payload_type={type(payload).__name__}"
        )
        return None

    tag_name = payload.get("tag_name", "")
    try:
        latest_version = _parse_version(tag_name)
    except InvalidVersion:
        logger.warning(
            f"skip update notification because release tag is invalid: {tag_name!r}"
        )
        return None

    if latest_version <= installed_version:
        return None

    normalized_latest_version = str(latest_version)
    logger.info(
        "MoneyPrinterTurbo update available: "
        f"current={installed_version}, latest={normalized_latest_version}"
    )
    return normalized_latest_version


@dataclass(frozen=True)
class UpdateCheckSnapshot:
    """Instant status of the background version check, read by the WebUI without blocking."""

    complete: bool
    available_version: str | None = None


class AsyncUpdateChecker:
    """
    Run the version check in a background thread and cache the latest result.

    Streamlit re-executes the page script after any widget interaction. Hitting
    GitHub directly in the title area would block the whole page on first open or
    when the cache expires. The network request therefore runs in a daemon thread
    while the page only reads the current snapshot; once the check finishes, a
    short-lived WebUI fragment refreshes the result once.

    Results are cached whether an update was found or not (no update / network
    failure), so an unreachable GitHub is not re-requested on every rerun. The
    lock only guards in-memory state and never wraps the network request, so other
    sessions can still read the check status without blocking.
    """

    def __init__(
        self,
        check: Callable[[str], str | None] = get_available_update,
        ttl_seconds: float = UPDATE_CHECK_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current_version: str | None = None
        self._available_version: str | None = None
        self._completed_at: float | None = None
        self._checking = False
        self._request_id = 0

    def poll(self, current_version: str) -> UpdateCheckSnapshot:
        """Return the check snapshot immediately; start a new background check when stale."""
        normalized_current_version = str(current_version or "").strip()
        now = self._clock()

        with self._lock:
            cache_is_fresh = (
                self._current_version == normalized_current_version
                and self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return UpdateCheckSnapshot(
                    complete=True,
                    available_version=self._available_version,
                )

            if (
                self._checking
                and self._current_version == normalized_current_version
            ):
                return UpdateCheckSnapshot(complete=False)

            # When the version changed or the cache expired, the old result must not
            # be shown any more. Clear the state before starting the new thread so
            # callers get an explicit pending snapshot during the check.
            self._current_version = normalized_current_version
            self._available_version = None
            self._completed_at = None
            self._checking = True
            self._request_id += 1

            try:
                worker = threading.Thread(
                    target=self._run_check,
                    args=(normalized_current_version, self._request_id),
                    name="mpt-version-check",
                    daemon=True,
                )
                worker.start()
            except RuntimeError as exc:
                # Thread exhaustion must not leave every subsequent poll pending.
                logger.warning(f"could not start update check thread: {exc}")
                self._checking = False
                self._completed_at = now
                return UpdateCheckSnapshot(complete=True)

        return UpdateCheckSnapshot(complete=False)

    def _run_check(self, current_version: str, request_id: int) -> None:
        try:
            available_version = self._check(current_version)
        except Exception:
            # get_available_update already handles expected network and data errors.
            # This is the last safety boundary of the background thread, so the full
            # stack must be logged to avoid an unexpected failure silently ending in a
            # permanent pending state.
            logger.exception(
                "unexpected error while checking for a MoneyPrinterTurbo update"
            )
            available_version = None

        with self._lock:
            # A -> B -> A can launch two workers for the same version. Only the
            # latest request may complete the current snapshot.
            if self._request_id != request_id:
                return
            self._available_version = available_version
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_UPDATE_CHECKER = AsyncUpdateChecker()


def poll_available_update(current_version: str) -> UpdateCheckSnapshot:
    """Read the global background checker state so Streamlit sessions do not re-hit GitHub."""
    return _ASYNC_UPDATE_CHECKER.poll(current_version)
