"""Statistics, preview and cleanup service for the video material cache."""

from __future__ import annotations

import os
import re
import stat
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# Online material uses the URL's MD5 as a stable file name. Cache management
# only accepts that naming format so videos, readme files or other business
# files a user accidentally drops into the directory are not deleted as cache.
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")
# save_video writes to this same-directory name before atomically publishing a
# validated clip. A forced process exit can leave one behind, so include only
# aged temporary files in cache cleanup and never touch an active download.
_VIDEO_CACHE_TEMP_FILE_PATTERN = re.compile(
    r"^\.vid-[0-9a-f]{32}-[a-z0-9_]+\.mp4$"
)
_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class VideoCacheStats:
    """Lightweight stats for the cache directory, based only on filesystem
    metadata."""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None


@dataclass(frozen=True)
class VideoCacheCleanupResult:
    """Result of one cleanup run, allowing some files to fail deletion."""

    deleted_count: int = 0
    deleted_size: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class _VideoCacheEntry:
    """Minimal file information captured during the scan, so cleanup never
    opens or parses videos."""

    path: str
    name: str
    size: int
    mtime: float
    device: int
    inode: int
    mtime_ns: int


def video_cache_dir() -> str:
    """Return the default video cache directory managed by the project."""

    return os.path.realpath(utils.storage_dir("cache_videos"))


def _iter_video_cache_entries(include_temp: bool = False) -> Iterator[_VideoCacheEntry]:
    """
    Sequentially scan the first level of the default cache directory.

    ``os.scandir`` is used so that with tens of thousands of cached files the
    metadata returned by the directory walk is reused, avoiding the extra
    file-type lookup after ``Path.iterdir``. Nothing is recursive, no video is
    opened and FFmpeg is never invoked, so the cost scales linearly with the
    number of files rather than with the total video size.
    """

    cache_dir = video_cache_dir()
    try:
        entries = os.scandir(cache_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            f"failed to scan video cache directory: path={cache_dir}, error={exc}"
        )
        return

    with entries:
        for entry in entries:
            if not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name) and not (
                include_temp and _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
            ):
                continue

            try:
                # Do not follow symlinks, so cleanup can never escape the
                # default cache directory boundary.
                if not entry.is_file(follow_symlinks=False):
                    continue
                # DirEntry.stat() reports st_dev/st_ino as 0 on Windows, which
                # would never match the os.stat() identity check before deletion.
                stat_result = os.stat(entry.path, follow_symlinks=False)
            except OSError as exc:
                logger.warning(
                    f"failed to inspect video cache file: file={entry.name}, error={exc}"
                )
                continue

            yield _VideoCacheEntry(
                path=entry.path,
                name=entry.name,
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
                device=stat_result.st_dev,
                inode=stat_result.st_ino,
                mtime_ns=stat_result.st_mtime_ns,
            )


def _is_cleanup_candidate(
    entry: _VideoCacheEntry,
    max_age_days: int | None,
    now: float,
) -> bool:
    if (
        _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
        and entry.mtime >= now - _SECONDS_PER_DAY
    ):
        return False
    if max_age_days is None:
        return True
    return entry.mtime < now - max_age_days * _SECONDS_PER_DAY


def _validate_max_age_days(max_age_days: int | None) -> None:
    """Invalid cleanup parameters must be rejected consistently even when the
    cache directory is empty."""
    if max_age_days is None:
        return
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, int)
        or max_age_days <= 0
    ):
        raise ValueError("max_age_days must be a positive integer or None")


def get_video_cache_stats(max_age_days: int | None = None) -> VideoCacheStats:
    """
    Count all cached files, or preview the cleanable ones whose mtime is older
    than the given number of days.

    ``max_age_days=None`` means all cached files. Stats only read directory
    entry size and mtime, never video contents, so even a very large cache
    produces no I/O proportional to its total size.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    file_count = 0
    total_size = 0
    oldest_mtime = None
    newest_mtime = None

    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        file_count += 1
        total_size += entry.size
        oldest_mtime = (
            entry.mtime if oldest_mtime is None else min(oldest_mtime, entry.mtime)
        )
        newest_mtime = (
            entry.mtime if newest_mtime is None else max(newest_mtime, entry.mtime)
        )

    return VideoCacheStats(
        file_count=file_count,
        total_size=total_size,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
    )


def clean_video_cache(max_age_days: int | None = None) -> VideoCacheCleanupResult:
    """
    Clean the default video cache and return a summary suitable for the user.

    A page preview and an actual cleanup click may be far apart in time, so the
    cleanup must rescan and re-evaluate instead of reusing an old candidate
    list. Deletion is fault-tolerant per file: a locked or permission-denied
    file is logged as a warning and skipped, so one bad file among hundreds
    does not fail the whole cleanup.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    logger.info(
        f"start cleaning video cache: max_age_days={max_age_days}"
    )

    candidate_count = 0
    candidate_size = 0
    deleted_count = 0
    deleted_size = 0
    failed_count = 0
    cache_dir = video_cache_dir()

    # Scan and delete in one pass without keeping a full candidate list in
    # memory. Even with hundreds of thousands of files, cleanup memory stays
    # constant; a single `now` is used so the cutoff does not drift during a
    # long cleanup and produce an unpredictable candidate range.
    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        candidate_count += 1
        candidate_size += entry.size
        try:
            # entry.path comes from a first-level scandir of the default
            # directory; re-validate the parent directory and file name before
            # deleting so future changes to the scan logic cannot widen the
            # deletable range by accident.
            if (
                os.path.realpath(os.path.dirname(entry.path)) != cache_dir
                or not (
                    _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name)
                    or _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
                )
                or os.path.islink(entry.path)
            ):
                raise ValueError("cache file is outside the managed directory")
            try:
                current = os.stat(entry.path, follow_symlinks=False)
            except FileNotFoundError:
                # Another cleanup already removed the scanned candidate.
                continue
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
                current.st_mtime_ns,
                current.st_size,
            ) != (entry.device, entry.inode, entry.mtime_ns, entry.size):
                # A downloader may atomically publish a new file at this name
                # after the scan. Do not delete the fresh replacement.
                continue
            os.unlink(entry.path)
            deleted_count += 1
            deleted_size += entry.size
        except (OSError, ValueError) as exc:
            failed_count += 1
            logger.warning(
                f"failed to delete video cache file: file={entry.name}, error={exc}"
            )

    logger.info(
        "finished cleaning video cache: "
        f"candidates={candidate_count}, candidate_bytes={candidate_size}, "
        f"deleted={deleted_count}, deleted_bytes={deleted_size}, failed={failed_count}"
    )
    return VideoCacheCleanupResult(
        deleted_count=deleted_count,
        deleted_size=deleted_size,
        failed_count=failed_count,
    )
