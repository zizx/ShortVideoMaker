import functools
import os
import re
import sys
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
_PROJECT_ROOT_IN_MESSAGE_RE = re.compile(re.escape(PROJECT_ROOT) + r"(?![\w.\-])")
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Loguru's default terminal handler has ID 0 at startup. Reloading the WebUI may
# only replace this base terminal output and must not call logger.remove() to
# clear all handlers, otherwise the temporary sinks collecting WebUI logs for
# running tasks would be removed too.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
# Only replace this immutable snapshot; logging threads never take the config lock
# held by generation tasks.
_runtime_logging = ("DEBUG", False)


def get_logging_options() -> tuple[str, bool]:
    return _runtime_logging


def set_logging_options(level: str, log_requests: bool = False) -> None:
    """Adjust this process's logging immediately, without rebuilding handlers or
    interrupting running tasks."""
    global _runtime_logging
    level = str(level).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"unsupported log level: {level}")
    if not isinstance(log_requests, bool):
        raise ValueError("log_requests must be a boolean")
    _runtime_logging = (level, log_requests)


def runtime_log_filter(record) -> bool:
    level, log_requests = _runtime_logging
    # Request tracing is a separate switch; users may enable diagnostics even at
    # ERROR level.
    if record["extra"].get("request_trace"):
        return log_requests
    return record["level"].no >= logger.level(level).no


# The WebUI filters logs by the task worker thread ID. Helper threads started for
# parallel downloads, clip encoding and heartbeats have their own thread IDs, so
# everything they log is dropped entirely, and the longest phase then shows no
# output in the WebUI at all. This maps "helper thread -> thread that started it"
# so the log filter can attribute helper records back to their task.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Return the root thread ID of the logging scope a thread belongs to; an
    unbound thread is its own root."""
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Wrap a function that will run on another thread so its logs are attributed to
    the thread calling this function.

    Must be called on the thread submitting the task: the scope is fixed when
    wrapping, and threads started by the helper thread are also attributed to the
    original task thread. The binding is removed after execution because the
    system reuses thread IDs, and a stale mapping would attribute later unrelated
    threads' logs to a task that already finished.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # Thread pools reuse threads: restore the previous attribution
                # instead of always deleting it, so nested wrappers do not clear
                # a binding the outer wrapper is still using.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Shorten an absolute path to a project-relative path starting with ``./`` and
    always using forward slashes.

    On Windows the project may be launched from a mapped network drive or a
    ``subst`` drive. Paths in the call stack are then still
    ``X:\\MoneyPrinterTurbo\\...`` while ``PROJECT_ROOT``, resolved through
    ``realpath``, lands on ``C:\\...``, so ``os.path.relpath`` raises
    ``ValueError``. loguru catches errors from format functions and drops the
    whole record, emptying both the terminal and the WebUI log panel, so the
    original path must be returned as a fallback. The same applies to files
    outside the project directory: prefixing ``./`` onto a ``..`` path only
    produces harder-to-read output.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # Windows relpath returns backslash-separated paths, so naive concatenation
    # gives mixed separators like ``./app\\utils``, inconsistent with log output on
    # other platforms.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Format terminal and WebUI logs consistently.

    Loguru passes the same record to multiple sinks. The first sink may already
    have converted an absolute path into a project-relative one, so both absolute
    paths and already-formatted paths starting with ``./`` are supported here.
    The WebUI sink disables colors, but time, level, call site and message content
    match the terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Log messages sometimes contain absolute paths of task files. Shortening
    # them to project-relative paths keeps the WebUI and terminal from showing two
    # different versions depending on the initialization entry point.
    # Only replace the complete project root; sibling directories such as
    # `<root>-backup` or `<root>2` share the prefix, and substring replacement
    # would turn them into wrong paths like `.-backup` and `.2`.
    record["message"] = _PROJECT_ROOT_IN_MESSAGE_RE.sub(".", record["message"])
    return LOG_RECORD_FORMAT


def force_utf8_console() -> None:
    """
    Switch stdout and stderr to UTF-8 before anything is logged or printed.

    When they go to a file or a pipe on Windows, as with
    ``webui.bat > webui.log 2>&1``, Python encodes them in the ANSI code page,
    such as cp1252 on English and cp936 on Chinese Windows. A log record with
    a character outside it (the circled digits of generate_video(), Chinese
    on cp1252, emoji on cp936) raises UnicodeEncodeError in the terminal sink,
    and Loguru drops the record and prints "Logging error in Loguru Handler"
    in its place. The CLI's final print() failed the same way after the video
    was written. Streams that cannot be switched, such as None under pythonw,
    are left as they are.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Safely replace the process-level terminal log handler while keeping
    task-specific handlers.

    Streamlit may re-run log initialization on hot reload or cache invalidation.
    Only the recorded handler ID is removed, so the WebUI logs that background
    tasks are currently writing are not interrupted. The lock protects the ID
    update when several browser sessions initialize at the same time.
    """
    global _terminal_handler_id

    set_logging_options(level, get_logging_options()[1])

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Tests or external entry points may already have removed this
                # handler. Carry on creating the new terminal output without
                # affecting the other log sinks that are still valid.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=0,
            filter=runtime_log_filter,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
