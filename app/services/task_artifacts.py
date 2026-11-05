"""Safe reads and writes of persisted files in the task directory."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


_SCRIPT_LOCKS = tuple(threading.RLock() for _ in range(256))


def _script_lock(task_id: str):
    return _SCRIPT_LOCKS[hash(task_id) % len(_SCRIPT_LOCKS)]


def _script_file(task_id: str) -> Path:
    """Return the task script manifest path, reusing the shared task directory logic."""
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically write JSON inside the target directory so an interrupted process
    never leaves a half-written file.

    The temporary file and the target file must live in the same directory for
    ``os.replace`` to keep its atomic replacement semantics on common local
    filesystems and Docker mount directories. Existing files are not modified
    until the write succeeds; on error only the temporary file created by this
    call is removed, and the error is left to the caller to decide whether it
    should affect the main flow.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=4,
                default=lambda value: value.__dict__,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """Create or fully replace the task's ``script.json`` manifest."""
    with _script_lock(task_id):
        _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    Merge fields into the task manifest while keeping existing ones, returning
    ``False`` on failure.

    Material sources are auxiliary diagnostic information, so file permissions,
    transient disk errors or a corrupted legacy file must not block video
    generation. This entry point therefore logs the full exception and degrades
    gracefully; creating the manifest for the first time still uses
    ``write_script_data``, leaving the main flow to decide how a failed write of
    the base task data is handled.
    """
    try:
        with _script_lock(task_id):
            target = _script_file(task_id)
            with target.open("r", encoding="utf-8") as script_file:
                payload = json.load(script_file)
            if not isinstance(payload, dict):
                raise ValueError("task script data must be a JSON object")

            payload.update(updates)
            _write_json_atomic(target, payload)
            return True
    except FileNotFoundError:
        # ``download_videos`` may also be called directly by tests, scripts or
        # third-party code; a missing task manifest is normal then, so do not emit
        # a warning or create a partial file just for auxiliary bookkeeping.
        logger.debug(
            f"skip task script update because script.json does not exist: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "failed to update task script data: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
